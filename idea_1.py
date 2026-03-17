import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# Assuming these are imported from your project files
# from models import SyntacticGemmaDenoiser
# from loss import DocAMRDiffusionLoss 
# from diffusion_utils import forward_diffusion, get_alphas_cumprod

def train_diffusion_model(
    dataloader: DataLoader, 
    epochs: int = 10, 
    total_timesteps: int = 1000, 
    device: str = 'cuda'
):
    # 1. Initialize your custom architecture
    # This model is designed to take noisy sequence embeddings AND graph data
    model = SyntacticGemmaDenoiser().to(device)
    
    # 2. Initialize your custom loss with the Sigmoid scheduler
    criterion = DocAMRDiffusionLoss(lambda_max=0.1, total_timesteps=total_timesteps).to(device)
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    alphas_cumprod = get_alphas_cumprod(total_timesteps).to(device)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()

            # --- A. UNPACK THE MULTIMODAL BATCH ---
            # Text is continuous, AMR is a graph, Adjacency is a matrix
            clean_text_embeddings = batch['text_embeddings'].to(device)
            doc_amr_graphs = batch['doc_amr_pyg_data'].to(device) # PyG Data (x, edge_index)
            true_adj_matrices = batch['true_adjacency'].to(device) 
            
            batch_size = clean_text_embeddings.shape[0]

            # --- B. THE FORWARD DIFFUSION PROCESS ---
            # Sample random timesteps uniformly across the batch
            t = torch.randint(0, total_timesteps, (batch_size,), device=device).long()
            
            # Add noise to the continuous text embeddings
            x_t, true_noise = forward_diffusion(clean_text_embeddings, t, alphas_cumprod)

            # --- C. THE REVERSE PROCESS (DENOISING + CROSS-ATTENTION) ---
            # The model predicts the noise while attending to the Doc-AMR structural prior.
            # It MUST return the cross-attention maps so we can compute the structural loss.
            noise_pred, cross_attn_maps = model(x_t, t, doc_amr_graphs)

            # --- D. THE CUSTOM DOC-AMR LOSS ---
            # This handles both the standard MSE denoising loss AND 
            # your scheduled structural penalty using the adjacency matrix.
            loss, loss_diff, loss_struct = criterion(
                noise_pred=noise_pred,
                true_noise=true_noise,
                x_t=x_t,
                t=t,
                alphas_cumprod=alphas_cumprod,
                attn_maps=cross_attn_maps,
                true_adj=true_adj_matrices
            )

            # --- E. OPTIMIZATION ---
            loss.backward()
            
            # Gradient clipping is highly recommended for diffusion models + GCNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Update progress bar metrics
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                "Total Loss": f"{loss.item():.4f}", 
                "Diff MSE": f"{loss_diff.item():.4f}", 
                "Struct Penalty": f"{loss_struct.item():.4f}"
            })
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader):.4f}")

    return model