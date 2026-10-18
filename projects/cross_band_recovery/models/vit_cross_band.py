"""
projects.cross_band_recovery.models.vit_cross_band
==================================================
Phase 4 (P4) / Gate G4b: Cross-Band Vision Transformer (ViT-CrossBand).

Architecture:
1. Per-Band Patch Tokenization (16x16 patches -> 64 spatial tokens per band).
2. Learnable Spatial Positional Embedding (64 tokens) + Channel Embedding (9 bands: 7 Optical + 2 SAR).
3. Zero-Leakage Encoder: Masked bands are physically excluded from token generation.
   Ensures dLoss/d(x_masked) == 0 identically (CHK-G4b.2, CHK-G4b.3).
4. Variable Sequence Length Batching with Self-Attention Padding Mask (CHK-G4b.1, CHK-G4b.5, CHK-G4b.6).
5. Cross-Band Transformer Decoder with Mask Tokens + Channel-Conditioned Query Tokens.
6. Reconstruction Head projecting to 128x128 7-band optical surface reflectance.
7. FP32 and BF16 numerical stability (CHK-G4b.7).
8. Supports all 127 optical patterns x {S1_on, S1_off} (CHK-G4b.8).
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


OPTICAL_NAMES = ["B02", "B03", "B04", "B8A", "B08", "B11", "B12"]
SAR_NAMES = ["vv", "vh"]
NUM_OPTICAL = 7
NUM_SAR = 2
TOTAL_CHANNELS = NUM_OPTICAL + NUM_SAR  # 9


class CrossBandPatchEmbed(nn.Module):
    """
    Projects single-channel 2D spatial patches (16x16) into embedding space (D).
    Shared across spectral bands to align representations while preserving per-band independence.
    """
    def __init__(self, patch_size: int = 16, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )

    def forward(self, x_single_channel: torch.Tensor) -> torch.Tensor:
        """
        x_single_channel: (B, 1, H, W)
        Returns: (B, N_patches, D)
        """
        # (B, D, H/P, W/P)
        feat = self.proj(x_single_channel)
        # Flatten spatial: (B, D, N) -> (B, N, D)
        tokens = feat.flatten(2).transpose(1, 2)
        return tokens


class CrossBandViT(nn.Module):
    """
    Cross-Band Vision Transformer for Optical Satellite Band Reconstruction.
    """
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 4,
        decoder_num_heads: int = 4,
        dropout: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # 128 // 16 = 8
        self.num_patches = self.grid_size * self.grid_size  # 64
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        
        # 1. Patch Tokenizer
        self.patch_embed = CrossBandPatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        
        # 2. Learnable Positional & Channel Embeddings (Encoder)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        # 9 channel embeddings: [0..6] Optical, [7..8] SAR (VV, VH)
        self.channel_embed = nn.Parameter(torch.zeros(TOTAL_CHANNELS, embed_dim))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.encoder_norm = norm_layer(embed_dim)
        
        # 4. Latent -> Decoder Projection
        self.decoder_proj = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # 5. Decoder Positional & Channel Embeddings + Mask Token
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))
        self.decoder_channel_embed = nn.Parameter(torch.zeros(TOTAL_CHANNELS, decoder_embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # 6. Transformer Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_num_heads,
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # 7. Reconstruction Head: Projects token to 16x16 pixel patch
        self.pred_head = nn.Linear(decoder_embed_dim, patch_size * patch_size, bias=True)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Truncated normal for embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_channel_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Xavier/Kaiming for projections
        nn.init.xavier_uniform_(self.pred_head.weight)
        if self.pred_head.bias is not None:
            nn.init.constant_(self.pred_head.bias, 0)
        nn.init.xavier_uniform_(self.decoder_proj.weight)
        if self.decoder_proj.bias is not None:
            nn.init.constant_(self.decoder_proj.bias, 0)

    def forward_encoder(
        self,
        opt: torch.Tensor,
        sar: Optional[torch.Tensor],
        mask_opt: torch.Tensor,
        mask_sar: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
        """
        Zero-Leakage Encoder:
        Takes only visible bands. Masked bands are physically NEVER indexed or passed to encoder.
        
        opt: (B, 7, H, W)
        sar: (B, 2, H, W) or None
        mask_opt: (B, 7) bool (True = MASKED)
        mask_sar: (B, 2) bool or None
        
        Returns:
            encoded_tokens: (B, L_max, D)
            padding_mask: (B, L_max) bool (True = padding / invalid)
            vis_channel_lists: List of lists containing visible channel indices per sample
        """
        B = opt.shape[0]
        device = opt.device
        dtype = opt.dtype
        
        if sar is None:
            mask_sar = torch.ones(B, NUM_SAR, dtype=torch.bool, device=device)
        elif mask_sar is None:
            mask_sar = torch.zeros(B, NUM_SAR, dtype=torch.bool, device=device)
            
        vis_channel_lists: List[List[int]] = []
        sample_tokens_list: List[torch.Tensor] = []
        max_vis_bands = 0
        
        for b in range(B):
            # Identify which optical channels are VISIBLE (mask == False)
            vis_opt = [c for c in range(NUM_OPTICAL) if not mask_opt[b, c]]
            
            # Identify which SAR channels are VISIBLE (mask == False)
            vis_sar = []
            if sar is not None:
                vis_sar = [NUM_OPTICAL + c for c in range(NUM_SAR) if not mask_sar[b, c]]
                
            vis_channels = vis_opt + vis_sar
            vis_channel_lists.append(vis_channels)
            max_vis_bands = max(max_vis_bands, len(vis_channels))
            
            # Zero-leakage extraction: Extract and tokenize ONLY visible bands
            band_tokens_for_sample = []
            for c in vis_channels:
                if c < NUM_OPTICAL:
                    # Optical band c: (1, 1, H, W)
                    band_slice = opt[b:b+1, c:c+1, :, :]
                else:
                    # SAR band (c - 7): (1, 1, H, W)
                    sar_idx = c - NUM_OPTICAL
                    band_slice = sar[b:b+1, sar_idx:sar_idx+1, :, :]
                    
                # (1, 64, D)
                tok = self.patch_embed(band_slice)
                # Add Spatial Positional Embedding + Dedicated Channel Embedding
                tok = tok + self.pos_embed + self.channel_embed[c].unsqueeze(0).unsqueeze(0)
                band_tokens_for_sample.append(tok)
                
            if band_tokens_for_sample:
                # Concatenate all visible tokens for sample b: (1, N_vis * 64, D)
                sample_tokens = torch.cat(band_tokens_for_sample, dim=1)
            else:
                # Fallback for safety (though CHK-G4a.5 guarantees no empty input)
                sample_tokens = torch.zeros(1, 0, self.embed_dim, device=device, dtype=dtype)
                
            sample_tokens_list.append(sample_tokens)
            
        # Padded batch construction (L_max = max_vis_bands * 64)
        L_max = max_vis_bands * self.num_patches
        if L_max == 0:
            L_max = self.num_patches  # minimum guard
            
        padded_tokens = torch.zeros(B, L_max, self.embed_dim, device=device, dtype=dtype)
        key_padding_mask = torch.ones(B, L_max, dtype=torch.bool, device=device)
        
        for b in range(B):
            tok = sample_tokens_list[b]
            n_tok = tok.shape[1]
            if n_tok > 0:
                padded_tokens[b, :n_tok, :] = tok[0]
                key_padding_mask[b, :n_tok] = False  # False means valid token in PyTorch
                
        # Transformer Encoder processing with padding mask
        # Tokens from sample b attend only to valid positions
        enc_out = self.encoder(padded_tokens, src_key_padding_mask=key_padding_mask)
        enc_out = self.encoder_norm(enc_out)
        
        return enc_out, key_padding_mask, vis_channel_lists

    def forward_decoder(
        self,
        enc_tokens: torch.Tensor,
        vis_channel_lists: List[List[int]],
        mask_opt: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-Band Decoder & Reconstruction:
        Decodes latent tokens and applies Reconstruction Head to produce full (B, 7, H, W) optical imagery.
        
        enc_tokens: (B, L_max, D)
        vis_channel_lists: List of visible channel indices
        mask_opt: (B, 7) bool
        
        Returns:
            pred_opt: (B, 7, 128, 128)
        """
        B = enc_tokens.shape[0]
        device = enc_tokens.device
        dtype = enc_tokens.dtype
        
        # Project encoder tokens to decoder dimension
        dec_vis = self.decoder_proj(enc_tokens)  # (B, L_max, D_dec)
        
        # Total fixed optical tokens to reconstruct per sample: 7 * 64 = 448 tokens
        total_opt_tokens = NUM_OPTICAL * self.num_patches
        
        # For each sample, assemble full sequence of optical tokens:
        # - If band c is VISIBLE: take its encoded token + decoder_channel_embed[c]
        # - If band c is MASKED: place mask_token + decoder_pos_embed + decoder_channel_embed[c]
        full_sample_tokens = []
        
        for b in range(B):
            vis_channels = vis_channel_lists[b]
            # Mapping from visible channel to its token offset in dec_vis[b]
            vis_offset_map = {}
            for i, c in enumerate(vis_channels):
                vis_offset_map[c] = i * self.num_patches
                
            sample_dec_tokens = []
            for c in range(NUM_OPTICAL):
                if not mask_opt[b, c] and c in vis_offset_map:
                    # Visible optical band: use encoded representation
                    off = vis_offset_map[c]
                    vis_tok = dec_vis[b:b+1, off:off + self.num_patches, :]
                    tok = vis_tok + self.decoder_channel_embed[c].unsqueeze(0).unsqueeze(0)
                else:
                    # Masked optical band: query mask token conditioned on channel + position
                    tok = (
                        self.mask_token
                        + self.decoder_pos_embed
                        + self.decoder_channel_embed[c].unsqueeze(0).unsqueeze(0)
                    )
                sample_dec_tokens.append(tok)
                
            full_sample_tokens.append(torch.cat(sample_dec_tokens, dim=1))
            
        full_batch_tokens = torch.cat(full_sample_tokens, dim=0)  # (B, 448, D_dec)
        
        # Decoder Transformer processing
        dec_out = self.decoder(full_batch_tokens)
        dec_out = self.decoder_norm(dec_out)  # (B, 448, D_dec)
        
        # Reconstruction Head: (B, 448, 16*16)
        patch_preds = self.pred_head(dec_out)
        
        # Unpatch into spatial 2D image (B, 7, 128, 128)
        # Reshape to (B, 7, 64, 16, 16)
        P = self.patch_size
        G = self.grid_size
        patch_preds = patch_preds.view(B, NUM_OPTICAL, G, G, P, P)
        # Permute to (B, 7, G, P, G, P) -> (B, 7, G*P, G*P) = (B, 7, 128, 128)
        pred_opt = patch_preds.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, NUM_OPTICAL, self.img_size, self.img_size)
        
        return pred_opt

    def forward(
        self,
        opt: torch.Tensor,
        sar: Optional[torch.Tensor] = None,
        mask_opt: Optional[torch.Tensor] = None,
        mask_sar: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Cross-Band Recovery.
        
        opt: (B, 7, 128, 128) Optical reflectance [0, 1]
        sar: (B, 2, 128, 128) S1 SAR dual-pol [VV, VH] or None
        mask_opt: (B, 7) bool (True = MASKED)
        mask_sar: (B, 2) bool or None
        
        Returns:
            dict with:
                'pred_opt': (B, 7, 128, 128) reconstructed optical reflectance
                'mask_opt': (B, 7) optical mask
                'mask_sar': (B, 2) SAR mask
                'latent': (B, L_max, D) encoded representations
        """
        B, C, H, W = opt.shape
        device = opt.device
        
        if mask_opt is None:
            mask_opt = torch.zeros(B, NUM_OPTICAL, dtype=torch.bool, device=device)
        if mask_sar is None and sar is not None:
            mask_sar = torch.zeros(B, NUM_SAR, dtype=torch.bool, device=device)
            
        enc_tokens, padding_mask, vis_channel_lists = self.forward_encoder(
            opt=opt,
            sar=sar,
            mask_opt=mask_opt,
            mask_sar=mask_sar
        )
        
        pred_opt = self.forward_decoder(
            enc_tokens=enc_tokens,
            vis_channel_lists=vis_channel_lists,
            mask_opt=mask_opt
        )
        
        return {
            "pred_opt": pred_opt,
            "mask_opt": mask_opt,
            "mask_sar": mask_sar,
            "latent": enc_tokens,
            "padding_mask": padding_mask
        }
