import torch
from open_clip.transformer import Transformer
from torch import nn
from torch.nn import LayerNorm, TransformerDecoderLayer
from torch.nn.init import trunc_normal_
import torch.nn.functional as F


# text decoder
class PromptEncoderWithoutPositionembGC(nn.Module):
    def __init__(self, prompt_num=17,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=1,
                 embed_dim=512,
                 ca_layers=1,
                 ca_heads=8,
                 dropout=0.,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.embed_dim = embed_dim

        self.apply(self._init_weights)


        # MLP(LN(q)) + SA(LN(q; k; v))
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            # attn_mask=None
        )
        self.ln_final = LayerNorm(transformer_width)


    def init_weights(self, pretrained=None):
        return None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.eye_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)


    def forward(self, prompt_emb):
        B, K, C = prompt_emb.shape # (batch_size, num_token, embed_dim)

        x = prompt_emb
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x.reshape(B, K, self.embed_dim)

        return x


# visual-text decoder
class ContextDecoderGC(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 visual_dim=512,# 1024
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.apply(self._init_weights)

        # CA -> cross attention + MLP(LN(q))
        # torch.nn.TransformerDecoderLayer
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(transformer_width, transformer_heads, dropout=dropout, batch_first=True) for _ in range(transformer_layers)
        ])

        # LN
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text, visual):

        visual = self.memory_proj(visual)
        x = self.text_proj(text)
        proj = nn.Linear(1849, 32)
        visual_32 = proj(visual.transpose(1, 2)).transpose(1, 2)
        result = visual_32 + x
        return self.out_proj(result)
