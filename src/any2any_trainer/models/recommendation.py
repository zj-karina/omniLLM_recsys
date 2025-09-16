"""
Recommendation model with semantic IDs using Qwen2.5-Omni.
Based on the experiment in omni_qwen.ipynb.
"""

import torch
import torch.nn as nn
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


class QwenTextEncoder(nn.Module):
    """Text encoder using Qwen2.5-Omni."""
    
    def __init__(self, ckpt="Qwen/Qwen2.5-Omni-7B", reduced_dim=1024, device='cpu'):
        super().__init__()
        self.device = device
        print(f"🔧 QwenTextEncoder: Creating on device {device}")
        
        # Load model on CPU first, then move to device
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            ckpt, torch_dtype=torch.float32  # Use float32 for stability
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(ckpt)
        self.hidden_size = self.model.config.text_config.hidden_size
        self.reduced_dim = reduced_dim
        
        # Move model to device
        self.model = self.model.to(device)
        print(f"✅ QwenTextEncoder: Model moved to {device}")
        
        self.pool_proj = nn.Linear(self.hidden_size, self.reduced_dim).to(device)
        
        # Initialize projection layer with smaller values
        nn.init.normal_(self.pool_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.pool_proj.bias)

        # Freeze the base model
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def forward(self, text):
        """Encode text to embeddings."""
        inputs = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden_state = outputs.hidden_states[-1]
        pooled = last_hidden_state.mean(dim=1)  # mean pooling
        projected = self.pool_proj(pooled)
        return projected


class FusionHead(nn.Module):
    """Fusion head for combining text and ID embeddings."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights properly
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.zeros_(self.dense1.bias)
        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.zeros_(self.dense2.bias)

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x
    
    def to(self, device):
        """Move model to device."""
        super().to(device)
        return self


class RecommendationModel(nn.Module):
    """
    Recommendation model with semantic IDs.
    
    Combines text embeddings from Qwen2.5-Omni with item ID embeddings
    to predict next item recommendations.
    """
    
    def __init__(
        self, 
        ckpt="Qwen/Qwen2.5-Omni-7B",
        id_vocab_size=1_000_000,
        id_dim=512,
        fusion_dim=1024,
        reduced_dim=1024,
        device='cpu'
    ):
        super().__init__()
        self.device = device
        print(f"🔧 RecommendationModel: Creating on device {device}")
        self.text_encoder = QwenTextEncoder(ckpt, reduced_dim, device)
        
        # ID embedding layer with proper initialization
        self.id_emb = nn.Embedding(id_vocab_size, id_dim).to(device)
        # Initialize embeddings with very small random values
        nn.init.normal_(self.id_emb.weight, mean=0.0, std=0.01)
        
        self.id_proj = nn.Linear(id_dim, self.text_encoder.reduced_dim).to(device)
        # Initialize projection layer with smaller values
        nn.init.normal_(self.id_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.id_proj.bias)
        
        # Fusion head for predicting next item ID
        self.fusion_head = FusionHead(
            input_dim=self.text_encoder.reduced_dim * 2,
            hidden_dim=fusion_dim,
            output_dim=id_vocab_size
        ).to(device)
        
        print(f"✅ RecommendationModel: All components moved to {device}")
        print(f"   Text encoder device: {next(self.text_encoder.parameters()).device}")
        print(f"   ID embedding device: {next(self.id_emb.parameters()).device}")
        print(f"   Fusion head device: {next(self.fusion_head.parameters()).device}")

    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.device = device
        # Move additional components
        if hasattr(self, 'id_emb'):
            self.id_emb = self.id_emb.to(device)
        if hasattr(self, 'id_proj'):
            self.id_proj = self.id_proj.to(device)
        if hasattr(self, 'fusion_head'):
            self.fusion_head = self.fusion_head.to(device)
        return self

    @classmethod
    def from_config(cls, config):
        """Create RecommendationModel from configuration."""
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        return cls(
            ckpt=config.model_name_or_path,
            id_vocab_size=getattr(config, 'id_vocab_size', 1_000_000),
            id_dim=getattr(config, 'id_dim', 512),
            fusion_dim=getattr(config, 'fusion_dim', 1024),
            reduced_dim=getattr(config, 'reduced_dim', 1024),
            device=device
        )

    def forward(self, text=None, id_ids=None, labels=None):
        """
        Forward pass for recommendation model.
        
        Args:
            text: List of text inputs
            id_ids: List of item ID sequences
            labels: Target item IDs for training
        """
        # Text embedding
        text_emb = self.text_encoder(text)  # (B, reduced_dim)
        
        # Check for NaN in text embeddings
        if torch.isnan(text_emb).any():
            print("⚠️ NaN detected in text embeddings, replacing with zeros")
            text_emb = torch.zeros_like(text_emb)
        
        # ID embedding
        if id_ids is not None:
            # id_ids are already tensors from collator
            device = next(self.parameters()).device
            
            # Move id_ids to device first
            id_ids_device = [ids.to(device) for ids in id_ids]
            
            ids_tensor = pad_sequence(
                id_ids_device,
                batch_first=True,
                padding_value=0
            )
            id_embeds = self.id_emb(ids_tensor).mean(dim=1)  # (B, id_dim)
            id_proj = self.id_proj(id_embeds)  # (B, reduced_dim)
            
            # Check for NaN in ID projections
            if torch.isnan(id_proj).any():
                print("⚠️ NaN detected in ID projections, replacing with zeros")
                id_proj = torch.zeros_like(id_proj)
        else:
            id_proj = torch.zeros_like(text_emb)

        # Fusion (concatenation)
        fused = torch.cat([text_emb, id_proj], dim=1)  # (B, reduced_dim*2)
        logits = self.fusion_head(fused)  # (B, id_vocab_size)
        
        # Check for NaN in logits
        if torch.isnan(logits).any():
            print("⚠️ NaN detected in logits, replacing with small values")
            logits = torch.randn_like(logits) * 0.01

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"⚠️ NaN loss detected! Logits stats: mean={logits.mean():.4f}, std={logits.std():.4f}")
                print(f"   Text emb stats: mean={text_emb.mean():.4f}, std={text_emb.std():.4f}")
                print(f"   ID proj stats: mean={id_proj.mean():.4f}, std={id_proj.std():.4f}")
                # Replace NaN with a small positive value
                loss = torch.tensor(1e-6, device=loss.device, requires_grad=True)
        
        return {"loss": loss, "logits": logits}
    
    def predict_next_item(self, text, id_ids=None, top_k=5):
        """
        Predict next item recommendations.
        
        Args:
            text: Input text
            id_ids: Optional item ID history
            top_k: Number of top recommendations to return
            
        Returns:
            Top-k item IDs and their probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(text, id_ids)
            logits = outputs["logits"]
            
            # Get top-k predictions
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
            
            return top_indices.cpu().numpy(), top_probs.cpu().numpy()


class VQVAEEncoder(nn.Module):
    """
    VQ-VAE encoder for creating semantic IDs.
    
    This is a placeholder for VQ-VAE implementation.
    In practice, you would implement a proper VQ-VAE here.
    """
    
    def __init__(self, input_dim, codebook_size, codebook_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, codebook_dim)
        )
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
    
    def forward(self, x):
        """Encode input to VQ-VAE codes."""
        z = self.encoder(x)
        
        # Find closest codebook entries
        distances = torch.cdist(z, self.codebook.weight)
        codes = torch.argmin(distances, dim=-1)
        
        # Quantize
        quantized = self.codebook(codes)
        
        return quantized, codes
    
    def decode(self, codes):
        """Decode codes back to embeddings."""
        return self.codebook(codes)


class SemanticIDRecommendationModel(RecommendationModel):
    """
    Enhanced recommendation model with VQ-VAE semantic IDs.
    
    Uses VQ-VAE to create semantic IDs instead of raw item IDs.
    """
    
    def __init__(
        self,
        ckpt="Qwen/Qwen2.5-Omni-7B",
        id_vocab_size=1_000_000,
        id_dim=512,
        fusion_dim=1024,
        reduced_dim=1024,
        vq_codebook_size=10000,
        vq_codebook_dim=256,
        device='cpu'
    ):
        super().__init__(ckpt, id_vocab_size, id_dim, fusion_dim, reduced_dim, device)
        
        # VQ-VAE for semantic IDs
        self.vq_vae = VQVAEEncoder(
            input_dim=id_dim,
            codebook_size=vq_codebook_size,
            codebook_dim=vq_codebook_dim
        )
        
        # Update ID projection to work with VQ-VAE codes
        self.id_proj = nn.Linear(vq_codebook_dim, self.text_encoder.reduced_dim)
    
    @classmethod
    def from_config(cls, config):
        """Create SemanticIDRecommendationModel from configuration."""
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        return cls(
            ckpt=config.model_name_or_path,
            id_vocab_size=getattr(config, 'id_vocab_size', 1_000_000),
            id_dim=getattr(config, 'id_dim', 512),
            fusion_dim=getattr(config, 'fusion_dim', 1024),
            reduced_dim=getattr(config, 'reduced_dim', 1024),
            vq_codebook_size=getattr(config, 'vq_codebook_size', 10000),
            vq_codebook_dim=getattr(config, 'vq_codebook_dim', 256),
            device=device
        )
    
    def forward(self, text=None, id_ids=None, labels=None, use_semantic_ids=True):
        """
        Forward pass with optional semantic ID processing.
        
        Args:
            text: List of text inputs
            id_ids: List of item ID sequences
            labels: Target item IDs for training
            use_semantic_ids: Whether to use VQ-VAE semantic IDs
        """
        # Text embedding
        text_emb = self.text_encoder(text)
        
        # ID embedding with optional VQ-VAE processing
        if id_ids is not None:
            ids_tensor = pad_sequence(
                [torch.tensor(ids, dtype=torch.long) for ids in id_ids],
                batch_first=True,
                padding_value=0
            ).to(text_emb.device)
            
            if use_semantic_ids:
                # Convert to semantic IDs using VQ-VAE
                id_embeds = self.id_emb(ids_tensor).mean(dim=1)  # (B, id_dim)
                quantized, codes = self.vq_vae(id_embeds)  # (B, vq_codebook_dim)
                id_proj = self.id_proj(quantized)  # (B, reduced_dim)
            else:
                # Use raw ID embeddings
                id_embeds = self.id_emb(ids_tensor).mean(dim=1)
                id_proj = self.id_proj(id_embeds)
        else:
            id_proj = torch.zeros_like(text_emb)

        # Fusion
        fused = torch.cat([text_emb, id_proj], dim=1)
        logits = self.fusion_head(fused)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {"loss": loss, "logits": logits}
