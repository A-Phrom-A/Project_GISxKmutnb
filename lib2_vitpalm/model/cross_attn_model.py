import torch
import torch.nn as nn
from .vit_encoder import ViTEncoder
from .seg_decoder import SegDecoder

class ViTCrossAttnModel(nn.Module):
    """
    Cross-Attention Fusion Architecture for GISxKMUTNB
    S2 is the primary modality. S1 and GSat are secondary.
    S2 queries information from S1 and GSat via Cross-Attention before decoding.
    """
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        out_channels=1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 3 Independent Encoders
        self.encoder_s1 = ViTEncoder(in_channels=2, img_size=img_size, patch_size=patch_size, 
                                     embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.encoder_s2 = ViTEncoder(in_channels=7, img_size=img_size, patch_size=patch_size, 
                                     embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.encoder_gsat = ViTEncoder(in_channels=3, img_size=img_size, patch_size=patch_size, 
                                       embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        
        # Cross Attention Layer
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder
        self.decoder = SegDecoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        """
        x: (B, 12, H, W)
        """
        # 1. Split modalities
        x_s1   = x[:, 0:2, :, :]
        x_s2   = x[:, 2:9, :, :]
        x_gsat = x[:, 9:12, :, :]
        
        # 2. Extract features
        feat_s1   = self.encoder_s1(x_s1)     # (B, L, C)
        feat_s2   = self.encoder_s2(x_s2)     # (B, L, C)
        feat_gsat = self.encoder_gsat(x_gsat) # (B, L, C)
        
        # 3. Cross Attention
        # Remove CLS token
        feat_s1 = feat_s1[:, 1:, :]
        feat_s2 = feat_s2[:, 1:, :]
        feat_gsat = feat_gsat[:, 1:, :]
        
        # Q = S2, K = V = [S1, GSat]
        kv = torch.cat([feat_s1, feat_gsat], dim=1) # (B, 2L, C)
        
        attn_out, _ = self.cross_attn(
            query=feat_s2,
            key=kv,
            value=kv
        )
        
        # Add & Norm
        fused_seq = self.norm(feat_s2 + attn_out) # (B, L, C)
        
        # 4. Decode
        out = self.decoder(fused_seq) # (B, img_size, img_size)
        return out
