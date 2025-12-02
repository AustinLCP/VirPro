import torch
from torch import nn
from torch.nn import LayerNorm
from open_clip.transformer import Transformer


class CLIPTextContextEncoderGC(nn.Module):
    def __init__(self, context_length=22,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=512, # 1024
                 out_dim=256,
                 vitae=False,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained
        self.vitae = vitae

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            # attn_mask=self.build_attention_mask()
        )

        self.register_buffer("attn_mask", self.build_attention_mask(), persistent=False)

        self.embed_dim = embed_dim

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            # print(self.vitae)
            if self.vitae:
                checkpoint = torch.load(pretrained, map_location='cpu')['state_dict']
            else:
                checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('module.'):  # For ViTAE loading
                    new_k = k.replace('module.', '')
                    if new_k.startswith('transformer.'):
                        state_dict[new_k] = checkpoint[k]

                    if new_k == 'positional_embedding' or new_k == 'text_projection' or new_k.startswith(
                            'token_embedding') or new_k.startswith('ln_final'):
                        if new_k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                            checkpoint[k] = checkpoint[k][:self.context_length]
                            print('positional_embedding is tuncated from 77 to', self.context_length)
                        state_dict[new_k] = checkpoint[k]
                else:
                    if k.startswith('transformer.'):
                        state_dict[k] = checkpoint[k]

                    if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                        if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                            checkpoint[k] = checkpoint[k][:self.context_length]
                            print('positional_embedding is tuncated from 77 to', self.context_length)
                        state_dict[k] = checkpoint[k]

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, x, context):
        x = x + self.positional_embedding
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x)
        x = self.transformer(x, attn_mask=self.attn_mask)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), context.argmax(dim=-1)] @ self.text_projection
        # x = x.reshape(B, K, self.embed_dim) k=prompt_bsz
        return x