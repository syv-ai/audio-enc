# train_mimi_hf_enhanced.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

from audiocraft.models import EncodecModel
from transformers import Wav2Vec2Model
from datasets import load_dataset
from einops import rearrange

import random
import os
from pathlib import Path

# --- Part 1: Helper Modules for Mimi (Identical to previous script) ---

class VectorQuantize(nn.Module):
    def __init__(self, dim, codebook_size, commitment_weight=0.25):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.embedding = nn.Embedding(codebook_size, dim)
        self.embedding.weight.data.uniform_(-1. / codebook_size, 1. / codebook_size)
    def forward(self, x):
        x_flat = rearrange(x, 'b s d -> (b s) d')
        distances = (torch.sum(x_flat**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(x_flat, self.embedding.weight.t()))
        indices = torch.argmin(distances, dim=1)
        indices = rearrange(indices, '(b s) -> b s', s=x.shape[1])
        quantized = self.embedding(indices)
        commitment_loss = F.mse_loss(x.detach(), quantized)
        embedding_loss = F.mse_loss(x, quantized.detach())
        loss = embedding_loss + self.commitment_weight * commitment_loss
        quantized = x + (quantized - x).detach()
        return quantized, indices, loss

class ResidualVectorQuantize(nn.Module):
    def __init__(self, dim, codebook_size, num_quantizers):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList([VectorQuantize(dim, codebook_size) for _ in range(num_quantizers)])
    def forward(self, x):
        residual = x
        all_quantized, all_indices, total_loss = [], [], 0
        for layer in self.layers:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            all_quantized.append(quantized)
            all_indices.append(indices)
            total_loss += loss
        quantized_sum = torch.sum(torch.stack(all_quantized), dim=0)
        indices_stack = torch.stack(all_indices, dim=-1)
        return quantized_sum, indices_stack, total_loss

# --- Part 2: The Mimi Model Architecture (Identical to previous script) ---

class Mimi(nn.Module):
    def __init__(self, encodec_model: EncodecModel, semantic_codebook_size=2048, num_acoustic_quantizers=7, transformer_layers=8, transformer_heads=8, wavlm_dim=1024):
        super().__init__()
        self.encoder = encodec_model.encoder
        self.decoder = encodec_model.decoder
        dim = self.encoder.output_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=transformer_heads, dim_feedforward=dim*4, batch_first=True, activation=F.gelu)
        self.pre_quant_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.semantic_quantizer = VectorQuantize(dim, semantic_codebook_size)
        self.acoustic_quantizer = ResidualVectorQuantize(dim, semantic_codebook_size, num_acoustic_quantizers)
        self.wavlm_projection = nn.Linear(dim, wavlm_dim)
        decoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=transformer_heads, dim_feedforward=dim*4, batch_first=True, activation=F.gelu)
        self.post_quant_transformer = nn.TransformerEncoder(decoder_layer, num_layers=transformer_layers)
    def forward(self, waveform):
        encoded_frames = self.encoder(waveform)
        transformed_frames = self.pre_quant_transformer(encoded_frames.transpose(1, 2)).transpose(1, 2)
        quantized_semantic, semantic_indices, vq_loss_semantic = self.semantic_quantizer(transformed_frames.transpose(1, 2))
        residual = transformed_frames.transpose(1, 2) - quantized_semantic.detach()
        quantized_acoustic, acoustic_indices, vq_loss_acoustic = self.acoustic_quantizer(residual)
        vq_loss = vq_loss_semantic + vq_loss_acoustic
        combined_quantized = quantized_semantic + quantized_acoustic
        decoder_input = self.post_quant_transformer(combined_quantized)
        reconstructed_waveform = self.decoder(decoder_input.transpose(1, 2))
        projected_semantic = self.wavlm_projection(quantized_semantic)
        return reconstructed_waveform, vq_loss, projected_semantic

# --- Part 3: Loss Functions (Identical to previous script) ---

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
        super().__init__()
        self.stft_losses = nn.ModuleList([STFTLoss(fs, hs, wl) for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)])
    def forward(self, x, y):
        total_loss = 0
        for loss_fn in self.stft_losses:
            total_loss += loss_fn(x, y)
        return total_loss

class STFTLoss(nn.Module):
    def __init__(self, fft_size, hop_size, win_length):
        super().__init__()
        self.fft_size, self.hop_size, self.win_length = fft_size, hop_size, win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
    def forward(self, x, y):
        x_stft = torch.stft(x.squeeze(1), self.fft_size, self.hop_size, self.win_length, self.window, return_complex=True)
        y_stft = torch.stft(y.squeeze(1), self.fft_size, self.hop_size, self.win_length, self.window, return_complex=True)
        sc_loss = F.l1_loss(torch.abs(x_stft), torch.abs(y_stft))
        mag_loss = F.l1_loss(torch.log(torch.abs(x_stft).clamp(min=1e-5)), torch.log(torch.abs(y_stft).clamp(min=1e-5)))
        return sc_loss + mag_loss

def calculate_distillation_loss(projected_semantic, waveform, wavlm_model, target_sr=16000, model_sr=24000):
    with torch.no_grad():
        if model_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=model_sr, new_freq=target_sr).to(waveform.device)
            wavlm_input = resampler(waveform.squeeze(1))
        else:
            wavlm_input = waveform.squeeze(1)
        wavlm_output = wavlm_model(wavlm_input).last_hidden_state
        target_len = projected_semantic.shape[1]
        wavlm_targets = F.interpolate(wavlm_output.transpose(1, 2), size=target_len).transpose(1, 2)
    return 1 - F.cosine_similarity(projected_semantic, wavlm_targets, dim=-1).mean()

# --- Part 4: Data Pipeline (Enhanced with caching) ---

class HuggingFaceAudioDataset(Dataset):
    def __init__(self, dataset_name, split, sample_rate, segment_duration, audio_column_name='audio'):
        """
        Args:
            dataset_name (str): Name of the dataset on Hugging Face Hub.
            split (str): The split to use (e.g., 'train', 'test').
            sample_rate (int): The target sample rate to resample all audio to.
            segment_duration (int): The duration of audio segments in seconds.
            audio_column_name (str): The name of the column containing audio data.
        """
        self.dataset = load_dataset(dataset_name, 'en-US', split=split)
        self.sample_rate = sample_rate
        self.segment_length = segment_duration * sample_rate
        self.audio_column_name = audio_column_name
        
        # Cache resamplers to avoid creating them repeatedly
        self.resamplers = {}
        
        print(f"Loaded {dataset_name} ({split}) with {len(self.dataset)} examples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            # Get audio data, which is a dictionary {'array':..., 'sampling_rate':...}
            audio_data = self.dataset[idx][self.audio_column_name]
            waveform_np, sr = audio_data['array'], audio_data['sampling_rate']
            
            # Convert to PyTorch tensor and ensure it's 2D (1, Time)
            waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)
            
            # Resample if necessary (with caching)
            if sr != self.sample_rate:
                if sr not in self.resamplers:
                    self.resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = self.resamplers[sr](waveform)
            
            # Convert to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Pad or extract a random segment to ensure fixed length
            if waveform.shape[1] < self.segment_length:
                padding = self.segment_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            else:
                start = random.randint(0, waveform.shape[1] - self.segment_length)
                waveform = waveform[:, start:start + self.segment_length]
                
            return waveform
        except Exception as e:
            print(f"Error processing item at index {idx}: {e}")
            # Return a silent waveform on error to avoid crashing the training loop
            return torch.zeros(1, self.segment_length)

# --- Part 5: Validation Function ---

def validate_model(mimi, val_dataloader, stft_loss_fn, wavlm_teacher, base_encodec, lambda_recon, lambda_vq, lambda_distill, device):
    """Run validation and return average losses."""
    mimi.eval()
    total_loss, total_recon_loss, total_vq_loss, total_distill_loss = 0, 0, 0, 0
    num_batches = 0
    
    with torch.no_grad():
        for waveform in val_dataloader:
            waveform = waveform.to(device)
            
            reconstructed_waveform, vq_loss, projected_semantic = mimi(waveform)
            
            # Calculate losses
            recon_loss = stft_loss_fn(reconstructed_waveform, waveform)
            distill_loss = calculate_distillation_loss(projected_semantic, waveform, wavlm_teacher, model_sr=base_encodec.sample_rate)
            
            batch_total_loss = (lambda_recon * recon_loss + lambda_vq * vq_loss + lambda_distill * distill_loss)
            
            total_loss += batch_total_loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_distill_loss += distill_loss.item()
            num_batches += 1
            
            # Don't validate on too many batches to save time
            if num_batches >= 20:
                break
    
    mimi.train()
    return {
        'total_loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'vq_loss': total_vq_loss / num_batches,
        'distill_loss': total_distill_loss / num_batches
    }

# --- Part 6: Enhanced Training Script ---

def train():
    # --- Config ---
    # <<< YOU CAN CHANGE THESE >>>
    DATASET_NAME = "CoRal-project/coral-tts"
    TRAIN_SPLIT = "train"
    VAL_SPLIT = "validation"  # Use validation split if available
    AUDIO_COLUMN_NAME = "audio"
    
    BATCH_SIZE = 4
    LEARNING_RATE = 3e-4
    NUM_TRAIN_STEPS = 10000
    SEGMENT_DURATION_S = 4
    
    # Training parameters
    GRADIENT_CLIP_NORM = 1.0
    VALIDATION_INTERVAL = 500
    CHECKPOINT_INTERVAL = 1000
    SAMPLE_INTERVAL = 500
    
    # Loss weights
    lambda_recon = 1.0
    lambda_vq = 1.0
    lambda_distill = 2.5 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path("mimi_training_outputs")
    output_dir.mkdir(exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # --- Models ---
    print("Loading models...")
    # Load EncodecModel with correct API
    try:
        # Try the standard way for 24kHz model
        base_encodec = EncodecModel.get_pretrained('facebook/encodec_24khz').to(device)
        print("Loaded 24kHz EncodecModel")
    except Exception as e:
        print(f"Failed to load 24kHz model: {e}")
        try:
            # Try 32kHz model as fallback
            base_encodec = EncodecModel.get_pretrained('facebook/encodec_32khz').to(device)
            print("Loaded 32kHz EncodecModel")
        except Exception as e2:
            print(f"Failed to load 32kHz model: {e2}")
            # Last resort - try loading without specifying model
            base_encodec = EncodecModel.get_pretrained().to(device)
            print("Loaded default EncodecModel")
    wavlm_teacher = Wav2Vec2Model.from_pretrained("microsoft/wavlm-large").to(device)
    wavlm_teacher.eval()
    mimi = Mimi(encodec_model=base_encodec, transformer_layers=4, transformer_heads=4).to(device)

    # --- Data ---
    print("Loading datasets...")
    train_dataset = HuggingFaceAudioDataset(
        dataset_name=DATASET_NAME,
        split=TRAIN_SPLIT,
        sample_rate=base_encodec.sample_rate,
        segment_duration=SEGMENT_DURATION_S,
        audio_column_name=AUDIO_COLUMN_NAME
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # Try to create validation dataset, fall back to train if validation split doesn't exist
    try:
        val_dataset = HuggingFaceAudioDataset(
            dataset_name=DATASET_NAME,
            split=VAL_SPLIT,
            sample_rate=base_encodec.sample_rate,
            segment_duration=SEGMENT_DURATION_S,
            audio_column_name=AUDIO_COLUMN_NAME
        )
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        print(f"Using separate validation split: {VAL_SPLIT}")
    except:
        print(f"Validation split '{VAL_SPLIT}' not found, using training data for validation")
        val_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # --- Optimizer & Loss ---
    optimizer = torch.optim.AdamW(mimi.parameters(), lr=LEARNING_RATE)
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)

    # --- Training Loop ---
    print("Starting training...")
    step = 0
    best_val_loss = float('inf')
    
    while step < NUM_TRAIN_STEPS:
        for waveform in train_dataloader:
            if step >= NUM_TRAIN_STEPS: 
                break
            
            waveform = waveform.to(device)
            
            # Forward pass
            reconstructed_waveform, vq_loss, projected_semantic = mimi(waveform)
            
            # Calculate losses
            recon_loss = stft_loss_fn(reconstructed_waveform, waveform)
            distill_loss = calculate_distillation_loss(projected_semantic, waveform, wavlm_teacher, model_sr=base_encodec.sample_rate)
            
            total_loss = (lambda_recon * recon_loss + lambda_vq * vq_loss + lambda_distill * distill_loss)
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(mimi.parameters(), max_norm=GRADIENT_CLIP_NORM)
            optimizer.step()
            
            # Logging
            if step % 10 == 0:
                print(f"Step {step}/{NUM_TRAIN_STEPS} | Total Loss: {total_loss.item():.4f} | Recon: {recon_loss.item():.4f} | VQ: {vq_loss.item():.4f} | Distill: {distill_loss.item():.4f}")
            
            # Validation
            if step > 0 and step % VALIDATION_INTERVAL == 0:
                print("Running validation...")
                val_losses = validate_model(mimi, val_dataloader, stft_loss_fn, wavlm_teacher, base_encodec, 
                                           lambda_recon, lambda_vq, lambda_distill, device)
                
                print(f"Validation at step {step}:")
                print(f"  Total Loss: {val_losses['total_loss']:.4f}")
                print(f"  Recon Loss: {val_losses['recon_loss']:.4f}")
                print(f"  VQ Loss: {val_losses['vq_loss']:.4f}")
                print(f"  Distill Loss: {val_losses['distill_loss']:.4f}")
                
                # Save best model
                if val_losses['total_loss'] < best_val_loss:
                    best_val_loss = val_losses['total_loss']
                    torch.save(mimi.state_dict(), output_dir / "best_model.pt")
                    print(f"New best model saved! Loss: {best_val_loss:.4f}")
            
            # Save audio samples
            if step > 0 and step % SAMPLE_INTERVAL == 0:
                torchaudio.save(samples_dir / f"original_{step}.wav", waveform[0].cpu(), base_encodec.sample_rate)
                torchaudio.save(samples_dir / f"reconstructed_{step}.wav", reconstructed_waveform[0].detach().cpu(), base_encodec.sample_rate)
                print(f"Saved audio samples at step {step}")
            
            # Save checkpoints
            if step > 0 and step % CHECKPOINT_INTERVAL == 0:
                checkpoint = {
                    'model_state_dict': mimi.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': step,
                    'best_val_loss': best_val_loss
                }
                torch.save(checkpoint, checkpoints_dir / f"checkpoint_{step}.pt")
                print(f"Saved checkpoint at step {step}")

            step += 1

    print("Training finished!")
    
    # Final validation
    print("Running final validation...")
    final_val_losses = validate_model(mimi, val_dataloader, stft_loss_fn, wavlm_teacher, base_encodec, 
                                    lambda_recon, lambda_vq, lambda_distill, device)
    print(f"Final validation loss: {final_val_losses['total_loss']:.4f}")
    
    # Save final model
    torch.save(mimi.state_dict(), output_dir / "final_model.pt")
    print(f"Final model saved to {output_dir / 'final_model.pt'}")

if __name__ == '__main__':
    train()