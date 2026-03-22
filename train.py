import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel, get_scheduler, AutoTokenizer
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Optional
from torch_geometric.nn import GCNConv, RGCNConv
from torch_geometric.utils import to_dense_batch


# 1. THE DOC-AMR ENCODER
# This module converts the graph structure into a format the Transformer understands.
class GraphAMREncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, num_relations: int = 100):
        super().__init__()
        # Upgraded to RGCN (Relational GCN) to handle specific AMR edge types (:arg0, :mod).
        # num_relations defines how many distinct edge types your dataset has.
        self.conv1 = RGCNConv(node_dim, hidden_dim, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_type, batch_index):
        """
        x: Node features (e.g., BERT embeddings of the AMR node labels) [total_nodes, node_dim]
        edge_index: Graph connectivity [2, total_edges]
        edge_type: The AMR relation indices (e.g., 0 for :ARG0, 1 for :mod) [total_edges]
        batch_index: Tells PyG which nodes belong to which graph in the batch [total_nodes]
        """
        # First Layer: Nodes gather info from neighbors based on Edge Type
        h = self.conv1(x, edge_index, edge_type)
        h = self.activation(h)
        
        # Second Layer: Nodes gather info from 2-hops away based on Edge Type
        h = self.conv2(h, edge_index, edge_type)
        h = self.layer_norm(h)
        
        # THE BRIDGE: Convert the irregular PyG graph batch into a dense 
        # rectangular tensor that Hugging Face Transformers can use for Cross-Attention.
        # dense_h shape: [batch_size, max_nodes_in_batch, hidden_dim]
        # mask shape: [batch_size, max_nodes_in_batch] (True for real nodes, False for padding)
        dense_h, attention_mask = to_dense_batch(h, batch_index)
        
        return dense_h, attention_mask

# 2. THE DENOISING TRANSFORMER (The Core Backbone)
# Diffusion network responsible for predicting the noise added to the text embeddings, 
#  conditioned on the AMR graph and the diffusion timestep.
class DiffusionDenoisingModel(nn.Module):
    def __init__(self, denoiser_config, node_dim, hidden_dim, num_relations=100, timesteps=1000):
        super().__init__()
        # The new Graph Encoder
        self.amr_encoder = GraphAMREncoder(node_dim, hidden_dim, num_relations=num_relations)
        
        # The Hugging Face Transformer Backbone
        self.transformer = AutoModel.from_config(denoiser_config)
        self.time_embed = nn.Embedding(timesteps, hidden_dim)

    def forward(self, x_t, t, amr_x, amr_edge_index, amr_edge_type, amr_batch_index):
        """
        x_t: Noisy text embeddings [batch_size, seq_len, hidden_dim]
        t: Timestep [batch_size]
        """
        # 1. Encode the Structural Graph
        # amr_context is now dynamically shaped based on the largest graph in the batch
        amr_context, amr_mask = self.amr_encoder(amr_x, amr_edge_index, amr_edge_type, amr_batch_index)
        
        # 2. Inject Timestep into noisy text
        t_emb = self.time_embed(t).unsqueeze(1)
        x_t = x_t + t_emb
        
        # 3. Cross-Attention Denoising
        # The Transformer looks at the text (x_t) and attends to the Graph (amr_context)
        # We pass the amr_mask so it doesn't attend to empty padded nodes!
        outputs = self.transformer(
            inputs_embeds=x_t,
            encoder_hidden_states=amr_context, 
            encoder_attention_mask=amr_mask, # CRITICAL: Ignore padded graph nodes
            return_dict=True
        )
        
        return outputs.last_hidden_state


def get_ddpm_schedule(timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod)
    }

def train_diffusion_model(
    model, 
    dataloader, 
    epochs=10, 
    lr=1e-4, 
    device="cuda",
    timesteps=1000,
    save_dir="./checkpoints",
    resume_from_checkpoint=None,
    max_checkpoints=5
):
    import os
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)

    # 0. Setup DDPM Noise Schedule
    schedule = get_ddpm_schedule(timesteps, device=device)
    sqrt_alphas_cumprod = schedule["sqrt_alphas_cumprod"]
    sqrt_one_minus_alphas_cumprod = schedule["sqrt_one_minus_alphas_cumprod"]

    # 1. Setup Optimizer and Scheduler
    # AdamW is standard for Transformers. We apply weight decay to prevent overfitting.
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    # A linear warmup scheduler helps stabilize early Transformer training
    num_training_steps = epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps), # 10% warmup
        num_training_steps=num_training_steps
    )

    # 2. Setup Loss Function and Mixed Precision Scaler
    criterion = nn.MSELoss()
    scaler = GradScaler() # Helps prevent underflow/overflow in fp16 training

    start_epoch = 0
    best_loss = float('inf')
    saved_checkpoints = []
    
    # Optional: Resume from a saved checkpoint
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Loading checkpoint from {resume_from_checkpoint}...")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        saved_checkpoints = checkpoint.get('saved_checkpoints', [])
        print(f"Resumed successfully at epoch {start_epoch+1}.")

    model.train()

    # 3. The Main Epoch Loop
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            # --- A. Extract Tensors and Move to Device ---
            # Text embeddings (Clean data, x_0)
            clean_text_embeds = batch['clean_embeds'].to(device) # Shape: [batch, seq_len, dim]
            
            # PyTorch Geometric Graph Data
            graph_batch = batch['graph_batch'].to(device)
            amr_x = graph_batch.x                   # Shape: [total_nodes, dim]
            amr_edge_index = graph_batch.edge_index # Shape: [2, total_edges]
            amr_edge_type = graph_batch.edge_type   # Shape: [total_edges]
            amr_batch_index = graph_batch.batch     # Shape: [total_nodes]

            batch_size = clean_text_embeds.shape[0]




            # --- B. Forward Diffusion Process
            # Sample a random timestep t for each item in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            
            # Sample random Gaussian noise
            noise = torch.randn_like(clean_text_embeds)
            
            # Extract the DDPM schedule values for the sampled timesteps
            # .view(-1, 1, 1) reshapes to [batch_size, 1, 1] for broadcasting over [batch, seq_len, dim]
            sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1, 1)
            sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
            
            # Create noisy text x_t using the accurate mathematical DDPM formulation
            # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
            x_t = sqrt_alpha_t * clean_text_embeds + sqrt_one_minus_alpha_t * noise 




            # --- C. Prediction and Loss
            # Passes the noisy text, timestep, and AMR structural graphs to the model to predict the original noise, 
            #   using a simple Mean Squared Error
            with autocast():
                # The model tries to predict the NOISE that was added, 
                # using the Structural Graph as guidance.
                predicted_noise = model(
                    x_t=x_t, 
                    t=t, 
                    amr_x=amr_x, 
                    amr_edge_index=amr_edge_index, 
                    amr_edge_type=amr_edge_type,
                    amr_batch_index=amr_batch_index
                )
                
                # Calculate Mean Squared Error
                loss = criterion(predicted_noise, noise)






            # --- D. Backward Pass and Weight Update ---
            # Scale the loss and call backward to compute gradients
            scaler.scale(loss).backward()
            
            # Gradient clipping prevents "exploding gradients", common in graphs and transformers
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights and learning rate
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()





            # --- E. Logging ---
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        checkpoint_path = os.path.join(save_dir, f"docamr_diffusion_epoch_{epoch+1}.pt")
        
        # Track best loss
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            
        # Prune older checkpoints to save storage
        saved_checkpoints.append(checkpoint_path)
        if len(saved_checkpoints) > max_checkpoints:
            oldest_ckpt = saved_checkpoints.pop(0)
            if os.path.exists(oldest_ckpt):
                os.remove(oldest_ckpt)
                print(f"Removed old checkpoint: {oldest_ckpt} to save space")

        state_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
            'saved_checkpoints': saved_checkpoints
        }
        
        torch.save(state_dict, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")
        
        # Save as the "best" model if it is the best so far
        if is_best:
            best_model_path = os.path.join(save_dir, "docamr_diffusion_best.pt")
            torch.save(state_dict, best_model_path)
            print(f"--> Saved new Best Model to {best_model_path} (Loss: {best_loss:.4f})")


@torch.no_grad() # Crucial: We do not track gradients during generation
def generate_text_from_amr(
    model, 
    base_model_name, 
    amr_batch, 
    seq_length=128, 
    num_timesteps=1000, 
    device="cuda"
):
    model.eval()
    model.to(device)
    
    # 1. Load Tokenizer and Word Embeddings for final decoding
    # Since the diffusion model operates completely in the continuous latent space (embeddings), 
    #   we need the base transformer's tokenizer and underlying word_embeddings matrix to map the final continuous 
    #   tensors back into discrete vocabulary IDs (actual words). 
    # It then extracts the AMR structure (Nodes, Edges, and Batch connectivity) exactly like the training loop.
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    word_embeddings = model.transformer.embeddings.word_embeddings.weight 

    # Extract the graph data
    graph_batch = amr_batch['graph_batch'].to(device)
    amr_x = graph_batch.x
    amr_edge_index = graph_batch.edge_index
    amr_edge_type = graph_batch.edge_type
    amr_batch_index = graph_batch.batch
    
    batch_size = amr_batch_index.max().item() + 1
    hidden_dim = model.amr_encoder.layer_norm.normalized_shape[0]




    # 2. Start with Pure Gaussian Noise (The "Blank Canvas")
    # Shape: [batch_size, seq_length, hidden_dim]
    x_t = torch.randn((batch_size, seq_length, hidden_dim), device=device)

    # Get the proper DDPM schedule
    schedule = get_ddpm_schedule(num_timesteps, device=device)
    beta = schedule["betas"]
    alpha = schedule["alphas"]
    alpha_bar = schedule["alphas_cumprod"]

    print("Starting reverse diffusion...")

   
    
    # 3. The Reverse Loop (from T down to 0)
    for t_step in reversed(range(num_timesteps)):
        # Create a tensor of the current timestep for the batch
        t_tensor = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
        
        # A. Predict the noise present in the current x_t
        predicted_noise = model(
            x_t=x_t, 
            t=t_tensor, 
            amr_x=amr_x, 
            amr_edge_index=amr_edge_index, 
            amr_edge_type=amr_edge_type,
            amr_batch_index=amr_batch_index
        )
        
        # B. The DDPM Reverse Step Math
        # We subtract a scaled version of the predicted noise to get a slightly cleaner x
        # Formula: x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (1 - alpha_t) / sqrt(1 - alpha_bar_t) * pred_noise)
        a_t = alpha[t_step]
        a_bar_t = alpha_bar[t_step]
        
        # Calculate the mean of the previous step
        mean = (1.0 / torch.sqrt(a_t)) * (x_t - ((1.0 - a_t) / torch.sqrt(1.0 - a_bar_t)) * predicted_noise)
        
        # Add a tiny bit of random variance back in (Langevin dynamics), unless it's the final step
        if t_step > 0:
            z = torch.randn_like(x_t)
            variance = torch.sqrt(beta[t_step]) * z
        else:
            variance = 0.0
            
        x_t = mean + variance



    # 4. Latent to Discrete Decoding (The "Rounding" Step)
    # x_t is now x_0 (our clean, continuous embeddings). 
    # We calculate the cosine similarity (or dot product) between our generated embeddings 
    # and every word in the BERT vocabulary to find the closest match.
    
    # x_t shape: [batch, seq_length, hidden_dim]
    # word_embeddings shape: [vocab_size, hidden_dim]
    # logits shape: [batch, seq_length, vocab_size]
    logits = torch.matmul(x_t, word_embeddings.T) 
    
    # Get the token ID with the highest score for each position
    predicted_token_ids = torch.argmax(logits, dim=-1)
    
    # 5. Convert IDs back to human-readable strings
    generated_texts = []
    for i in range(batch_size):
        tokens = predicted_token_ids[i].tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        generated_texts.append(text)
        
    return generated_texts





#################### Train section ####################

if __name__ == "__main__":
    from amr_dataset import DocAMRDataset, docamr_collate_fn
    from torch.utils.data import DataLoader
    from transformers import AutoConfig
    
    print("Starting training script directly...")
    
    # 1. Load the Dataset
    dataset = DocAMRDataset("/home/long/Master/Thesis/docAMR/doc_amr_baseline/output_doc_amr/docamr_docAMR.out", base_model_name="roberta-base", max_seq_len=128)
    
    # 2. Let the custom collator handle the PyG DataBatch shapes
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=docamr_collate_fn)

    # 3. Instantiate your Architecture
    denoiser_config = AutoConfig.from_pretrained("roberta-base")

    # We pass num_relations directly into the DenoisingModel
    model = DiffusionDenoisingModel(
        denoiser_config, 
        node_dim=768, 
        hidden_dim=768, 
        num_relations=dataset.num_relations
    )

    # 4. Train!
    train_diffusion_model(model, dataloader, epochs=3)