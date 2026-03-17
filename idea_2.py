import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from typing import Optional
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch

# 1. THE DOC-AMR ENCODER
# This module converts the graph structure into a format the Transformer understands.
class GraphAMREncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        # We use Graph Convolutional Networks (GCN) for the baseline.
        # For your thesis defense, you might eventually upgrade this to an 
        # RGCN (Relational GCN) to handle specific AMR edge types (:arg0, :mod).
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, batch_index):
        """
        x: Node features (e.g., BERT embeddings of the AMR node labels) [total_nodes, node_dim]
        edge_index: Graph connectivity [2, total_edges]
        batch_index: Tells PyG which nodes belong to which graph in the batch [total_nodes]
        """
        # First GNN Layer: Nodes gather info from immediate neighbors
        h = self.conv1(x, edge_index)
        h = self.activation(h)
        
        # Second GNN Layer: Nodes gather info from 2-hops away
        h = self.conv2(h, edge_index)
        h = self.layer_norm(h)
        
        # THE BRIDGE: Convert the irregular PyG graph batch into a dense 
        # rectangular tensor that Hugging Face Transformers can use for Cross-Attention.
        # dense_h shape: [batch_size, max_nodes_in_batch, hidden_dim]
        # mask shape: [batch_size, max_nodes_in_batch] (True for real nodes, False for padding)
        dense_h, attention_mask = to_dense_batch(h, batch_index)
        
        return dense_h, attention_mask

# 2. THE DENOISING TRANSFORMER (The Core Backbone)
# This is where the magic happens: merging noisy text with AMR guidance.
class DiffusionDenoisingModel(nn.Module):
    def __init__(self, denoiser_config, node_dim, hidden_dim):
        super().__init__()
        # The new Graph Encoder
        self.amr_encoder = GraphAMREncoder(node_dim, hidden_dim)
        
        # The Hugging Face Transformer Backbone
        from transformers import AutoModel
        self.transformer = AutoModel.from_config(denoiser_config)
        self.time_embed = nn.Embedding(1000, hidden_dim)

    def forward(self, x_t, t, amr_x, amr_edge_index, amr_batch_index):
        """
        x_t: Noisy text embeddings [batch_size, seq_len, hidden_dim]
        t: Timestep [batch_size]
        """
        # 1. Encode the Structural Graph
        # amr_context is now dynamically shaped based on the largest graph in the batch
        amr_context, amr_mask = self.amr_encoder(amr_x, amr_edge_index, amr_batch_index)
        
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

# # 3. THE FULL SYSTEM PIPELINE
# class DocAMRDiffusionPipeline(nn.Module):
#     def __init__(self, model: DiffusionDenoisingModel, amr_encoder: DocAMREncoder):
#         super().__init__()
#         self.model = model
#         self.amr_encoder = amr_encoder
#         self.mse_loss = nn.MSELoss()

#     def get_noise_schedule(self, t, device):
#         # Simplification: Linear noise schedule logic would go here
#         return torch.randn_like(t).to(device)

#     def forward_diffusion(self, x_0, t, noise):
#         """Standard Gaussian diffusion: adds noise to clean embeddings."""
#         # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
#         # (This is a simplified representation)
#         return x_0 + noise 

#     def training_step(self, x_0, amr_graph, t):
#         # 1. Encode the AMR semantic skeleton
#         amr_features = self.amr_encoder(amr_graph)
        
#         # 2. Add noise to the clean text embeddings
#         noise = torch.randn_like(x_0)
#         x_t = self.forward_diffusion(x_0, t, noise)
        
#         # 3. Predict the noise using the DocAMR context
#         predicted_noise = self.model(x_t, t, amr_features)
        
#         # 4. Objective: How well did the model use the AMR to 'see through' the noise?
#         loss = self.mse_loss(predicted_noise, noise)
#         return loss




def train_diffusion_model(
    model, 
    dataloader, 
    epochs=10, 
    lr=1e-4, 
    device="cuda"
):
    model.to(device)
    model.train()

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

    # 3. The Main Epoch Loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            # --- A. Extract Tensors and Move to Device ---
            # Text embeddings (Clean data, x_0)
            clean_text_embeds = batch['clean_embeds'].to(device) # Shape: [batch, seq_len, dim]
            
            # PyTorch Geometric Graph Data
            amr_x = batch['amr_x'].to(device)                   # Shape: [total_nodes, dim]
            amr_edge_index = batch['amr_edge_index'].to(device) # Shape: [2, total_edges]
            amr_batch_index = batch['amr_batch_index'].to(device)# Shape: [total_nodes]

            batch_size = clean_text_embeds.shape[0]

            # --- B. Diffusion Process: Sample Time and Noise ---
            # Sample a random timestep t for each item in the batch
            # Assuming T=1000 total diffusion steps
            t = torch.randint(0, 1000, (batch_size,), device=device).long()
            
            # Sample random Gaussian noise
            noise = torch.randn_like(clean_text_embeds)
            
            # Create noisy text x_t 
            # (In a real DDPM, you'd multiply x_0 and noise by alpha/beta schedule constants)
            # For this baseline: x_t = clean_text + noise
            x_t = clean_text_embeds + noise 

            # --- C. Forward Pass with Automatic Mixed Precision ---
            with autocast():
                # The model tries to predict the NOISE that was added, 
                # using the Structural Graph as guidance.
                predicted_noise = model(
                    x_t=x_t, 
                    t=t, 
                    amr_x=amr_x, 
                    amr_edge_index=amr_edge_index, 
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

        print(f"Epoch {epoch+1} Complete. Average Loss: {epoch_loss / len(dataloader):.4f}")

# (Assuming model and dataloader are already defined)
# train_diffusion_model(pipeline, dataloader, epochs=20)
    
    
    
    
    
    
    
    
    
    
    
    