import torch
import torch.nn as nn
from .vit_encoder import ViTEncoder
from .seg_decoder import SegDecoder

class ViTLateFusionModel(nn.Module):
    """
    Late Fusion Architecture for GISxKMUTNB
    Extracts features from 3 modalities independently using 3 ViT Encoders,
    concatenates the features, fuses them via 1x1 Convolution, and decodes.
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
        
        # Fusion Layer (1x1 Conv) to reduce 3 * embed_dim back to embed_dim
        self.fusion_conv = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1)
        self.act = nn.GELU()
        
        # Decoder
        self.decoder = SegDecoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size
        )

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x=None, s2=None, gsat=None, s1=None):
        """
        Supports either:
          - Single 12-band tensor: forward(x) where x is (B, 12, H, W)
          - Three tensors: forward(s1, s2, gsat) or forward(s1=s1, s2=s2, gsat=gsat)
        """
        # 1. Resolve modalities
        if s2 is not None and gsat is not None:
            x_s1, x_s2, x_gsat = x, s2, gsat
        elif s1 is not None and s2 is not None and gsat is not None:
            x_s1, x_s2, x_gsat = s1, s2, gsat
        elif x is not None and x.shape[1] == 12:
            x_s1   = x[:, 0:2, :, :]
            x_s2   = x[:, 2:9, :, :]
            x_gsat = x[:, 9:12, :, :]
        else:
            raise ValueError("Expected either a 12-band tensor (B, 12, H, W) or 3 separate tensors (s1, s2, gsat).")
        
        # 2. Extract features
        feat_s1   = self.encoder_s1(x_s1)     # (B, L, C)
        feat_s2   = self.encoder_s2(x_s2)     # (B, L, C)
        feat_gsat = self.encoder_gsat(x_gsat) # (B, L, C)
        
        B, L, C = feat_s1.shape
        H = W = self.img_size // self.patch_size
        
        # Remove CLS token
        feat_s1 = feat_s1[:, 1:, :]
        feat_s2 = feat_s2[:, 1:, :]
        feat_gsat = feat_gsat[:, 1:, :]
        
        # Reshape to (B, C, H, W)
        feat_s1 = feat_s1.transpose(1, 2).view(B, C, H, W)
        feat_s2 = feat_s2.transpose(1, 2).view(B, C, H, W)
        feat_gsat = feat_gsat.transpose(1, 2).view(B, C, H, W)
        
        # 3. Concatenate and Fuse
        feat_concat = torch.cat([feat_s1, feat_s2, feat_gsat], dim=1) # (B, 3C, H, W)
        fused = self.act(self.fusion_conv(feat_concat))               # (B, C, H, W)
        
        # Reshape back to sequence for decoder (B, L, C)
        fused_seq = fused.view(B, C, -1).transpose(1, 2)
        
        # 4. Decode
        out = self.decoder(fused_seq) # (B, img_size, img_size)
        return out
