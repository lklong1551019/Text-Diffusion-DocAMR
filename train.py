import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForMaskedLM, AutoConfig, PreTrainedModel, get_scheduler, AutoTokenizer
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
        
        # The Hugging Face Transformer Backbone for Masked Language Modeling
        self.transformer = AutoModelForMaskedLM.from_config(denoiser_config)
        self.time_embed = nn.Embedding(timesteps, hidden_dim)

    def forward(self, x_t, t, amr_x, amr_edge_index, amr_edge_type, amr_batch_index):
        """
        x_t: Discrete token ids (some masked) [batch_size, seq_len]
        t: Timestep [batch_size]
        """
        # 1. Encode the Structural Graph
        amr_context, amr_mask = self.amr_encoder(amr_x, amr_edge_index, amr_edge_type, amr_batch_index)
        
        # 2. Get embeddings and inject Timestep
        inputs_embeds = self.transformer.get_input_embeddings()(x_t)
        t_emb = self.time_embed(t).unsqueeze(1)
        inputs_embeds = inputs_embeds + t_emb
        
        # 3. Cross-Attention Denoising
        # The model predicts the vocabulary logits for every token position
        outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=amr_context, 
            encoder_attention_mask=amr_mask,
            return_dict=True
        )
        
        return outputs.logits


def train_diffusion_model(
    model, 
    dataloader, 
    tokenizer,
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

    # 1. Setup Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    num_training_steps = epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    # 2. Setup Loss Function for Discrete Tokens
    # We ignore -100 which is the standard PyTorch ignore index for unmasked tokens
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = GradScaler() 

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
            # Discrete token ids (Clean data, x_0)
            clean_text_ids = batch['target_ids'].to(device) # Shape: [batch, seq_len]
            
            # PyTorch Geometric Graph Data
            graph_batch = batch['graph_batch'].to(device)
            amr_x = graph_batch.x                   # Shape: [total_nodes, dim]
            amr_edge_index = graph_batch.edge_index # Shape: [2, total_edges]
            amr_edge_type = graph_batch.edge_type   # Shape: [total_edges]
            amr_batch_index = graph_batch.batch     # Shape: [total_nodes]

            batch_size = clean_text_ids.shape[0]
            seq_len = clean_text_ids.shape[1]

            # --- B. Forward Diffusion Process (Discrete Masking)
            # Sample a random timestep t for each item in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            
            # Masking ratio r(t) = t / timesteps (0.0 to 1.0)
            ratios = (t.float() / timesteps).unsqueeze(1) # [batch_size, 1]
            
            rand = torch.rand((batch_size, seq_len), device=device)
            pad_token_id = tokenizer.pad_token_id
            mask_token_id = tokenizer.mask_token_id
            
            # Mask tokens if random value < ratio (we allow masking pad tokens so it learns variable sequence lengths!)
            mask_condition = (rand < ratios)
            
            x_t = clean_text_ids.clone()
            x_t[mask_condition] = mask_token_id

            # --- C. Prediction and Loss
            with autocast():
                # The model predicts the vocabulary logits for ALL tokens
                logits = model(
                    x_t=x_t, 
                    t=t, 
                    amr_x=amr_x, 
                    amr_edge_index=amr_edge_index, 
                    amr_edge_type=amr_edge_type,
                    amr_batch_index=amr_batch_index
                ) # [batch_size, seq_len, vocab_size]
                
                # We only calculate loss on the tokens we explicitly MASKED
                labels = clean_text_ids.clone()
                labels[~mask_condition] = -100 # Ignore unmasked tokens
                
                # CrossEntropy expects [N, C] and [N]
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))






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


@torch.no_grad()
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
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    mask_token_id = tokenizer.mask_token_id
    
    # Extract the graph data
    graph_batch = amr_batch['graph_batch'].to(device)
    amr_x = graph_batch.x
    amr_edge_index = graph_batch.edge_index
    amr_edge_type = graph_batch.edge_type
    amr_batch_index = graph_batch.batch
    
    batch_size = amr_batch_index.max().item() + 1

    # 2. Start with 100% Masked Sequence (The "Blank Canvas")
    x_t = torch.full((batch_size, seq_length), mask_token_id, device=device, dtype=torch.long)

    print("Starting discrete reverse diffusion (Confidence-based Decoding)...")

    # 3. The Reverse Loop (from T-1 down to 0)
    # At each step, we progressively unmask tokens based on model confidence.
    for t_step in reversed(range(0, num_timesteps)):
        t_tensor = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
        
        # A. Predict the logits for all tokens
        logits = model(
            x_t=x_t, 
            t=t_tensor, 
            amr_x=amr_x, 
            amr_edge_index=amr_edge_index, 
            amr_edge_type=amr_edge_type,
            amr_batch_index=amr_batch_index
        )
        
        # B. Get probabilities and most likely tokens
        probs = torch.softmax(logits, dim=-1)
        max_probs, predicted_ids = torch.max(probs, dim=-1)
        
        # We only evaluate tokens that are CURRENTLY masked
        is_masked = (x_t == mask_token_id)
        
        if not is_masked.any():
            break
            
        # C. Partially Unmask
        for i in range(batch_size):
            masked_indices = is_masked[i].nonzero(as_tuple=True)[0]
            num_masked = len(masked_indices)
            
            if num_masked == 0:
                continue
                
            # Unmask a fraction of the remaining tokens. 
            # If t_step=0 (last step), unmask everything remaining.
            steps_remaining = t_step + 1
            num_to_unmask = max(1, int(num_masked / steps_remaining))
            if t_step == 0:
                num_to_unmask = num_masked
                
            # Find the most confident predictions among the masked tokens
            masked_probs = max_probs[i, masked_indices]
            _, topk_relative_indices = torch.topk(masked_probs, k=num_to_unmask)
            topk_absolute_indices = masked_indices[topk_relative_indices]
            
            # Permanently unmask these tokens by replacing [MASK] with the predicted ID
            x_t[i, topk_absolute_indices] = predicted_ids[i, topk_absolute_indices]

    # 4. Convert IDs back to human-readable strings
    generated_texts = []
    for i in range(batch_size):
        tokens = x_t[i].tolist()
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

    # 4. Load Tokenizer & Train!
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    # Hint: Diffusion models require LOTS of epochs to learn. 3 epochs is not enough. Try 200+
    train_diffusion_model(model, dataloader, tokenizer, epochs=300)