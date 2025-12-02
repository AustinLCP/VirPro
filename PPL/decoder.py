import torch
from open_clip.transformer import Transformer
from torch import nn
from torch.nn import LayerNorm, TransformerDecoderLayer
from torch.nn.init import trunc_normal_
import torch.nn.functional as F

# from timm.models.layers import drop, drop_path, trunc_normal_
# from collections import OrderedDict



################
# text decoder #
################
# class QuickGELU(nn.Module):
#     def forward(self, x: torch.Tensor):
#         return x * torch.sigmoid(1.702 * x)
#
# class DropPath(nn.Module):
#
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)
#
# class ResidualAttentionBlock(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
#         super().__init__()
#
#         self.attn = nn.MultiheadAttention(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2 = LayerNorm(d_model)
#         self.attn_mask = attn_mask
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#     def attention(self, x: torch.Tensor):
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
#
#     def forward(self, x: torch.Tensor):
#         x = x + self.drop_path(self.attention(self.ln_1(x)))
#         x = x + self.drop_path(self.mlp(self.ln_2(x)))
#         return x
#
# class Transformer(nn.Module):
#     def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
#         super().__init__()
#         self.width = width
#         self.layers = layers
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
#         self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])
#
#     def forward(self, x: torch.Tensor):
#         return self.resblocks(x)

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

        # self.prompt_num = prompt_num

        self.embed_dim = embed_dim

        # self.positional_embedding = nn.Parameter(torch.empty(self.prompt_num, transformer_width))
        # self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        # self.out_proj = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, embed_dim)
        # )

        self.apply(self._init_weights)


        # SA -> self-attention
        # MLP -> transformer 的 FFN
        # 包含了 MLP(LN(q)) + SA(LN(q; k; v))
        self.transformer = Transformer( # 直接复用了 OpenAI-CLIP 的文本 Transformer
            width=transformer_width,
            layers=transformer_layers, # num of ResidualAttentionBlock (Transformer Encoder Layer)
            heads=transformer_heads, # num of Multi-Head Self-Attention
            # attn_mask=None
        )
    #      +───────────────────+
    # x ──LN─►    Multi - Head  │
        #  │   Self - Attention │
        #  +──────────┬────────+
        #             ▼
        #          (残差加)
        #             │
#            +─────────────────+
#      LN ──►    FeedForward   │  # 两层线性 + QuickGELU
#            +────────┬────────+
        #             ▼
        #          (残差加)

        # LN
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

    # μ: 所有 keypoint 的每个 prompt 对应的加权平均值
    # eg. 在这张图里, ‘car’ 这个词最典型的大致长什么样？
    def forward(self, prompt_emb):
        B, K, C = prompt_emb.shape # (batch_size, num_token, embed_dim)

        x = prompt_emb
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x) # 让同一批 prompt tokens 彼此“互相看”。每个 token 同时充当 Query/Key/Value，通过注意力权重把其它 token 的信息加权汇聚到自己身上,这一步输出的向量作为均值 𝜇

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x) # 与 σ 分布对齐 self.ln_final = LayerNorm(transformer_width)
        x = x.reshape(B, K, self.embed_dim)

        return x


#######################
# visual-text decoder #
#######################
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
#
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, q, k, v):
#         B, N, C = q.shape
#         assert k.shape == v.shape
#         B, M, C = k.shape
#         q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
#         k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
#         v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)
#
#         attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale
#
#         attn = attn.softmax(dim=-1)
#
#         x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
# class TransformerDecoderLayer(nn.Module):
#     def __init__(
#             self,
#             d_model,
#             nhead,
#             dropout=0.1,
#     ):
#         super().__init__()
#         self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
#         self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#
#         self.mlp = nn.Sequential(
#             nn.Linear(d_model, d_model * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * 4, d_model)
#         )
#
#     # x: text
#     # mem: visual
#     def forward(self, x, mem):
#         q = k = v = self.norm1(x)
#         x = x + self.self_attn(q, k, v)
#         q = self.norm2(x)
#         x = x + self.cross_attn(q, mem, mem)
#         x = x + self.dropout(self.mlp(self.norm3(x)))
#         return x

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

        # 自定义的 TransformerDecoderLayer
        # self.decoder = nn.ModuleList([
        #     TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
        # ])

        # LN
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        # self.modality_embed = nn.Parameter(torch.randn(2, transformer_width))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # σ: 所有 keypoint 的每个 prompt 对应的方差
    # 模型觉得这个类别的外观在图中可能的变化大小（视角、遮挡、光照）
    def forward(self, text, visual):

        # visual: [16,1849,256]
        # x: [16,32,256]

        visual = self.memory_proj(visual) # Linear 投影把 visual embeddings 映射到统一维度, 充当 Key/Value。
        x = self.text_proj(text) # 把 text embeddings 投影到同一维度, 充当 Query

        # modality embed
        # visual = self.memory_proj(visual) + self.modality_embed[1]
        # x = self.text_proj(text) + self.modality_embed[0]

        ####################
        # torch CA decoder #
        ####################
        # Q: txt, K,V: img
        # CA(LN(q;k;v)) + MLP(LN(q))
        # for layer in self.decoder:
        #     x = layer(x, visual)
        #
        # # LN, 与 μ 分布对齐
        # return self.out_proj(x)

        ##########
        # concat #
        ##########
        # cat = torch.cat([x, visual], dim=1)  # [16, 1881, 256]
        # result = F.adaptive_avg_pool1d(cat.transpose(1, 2), 32).transpose(1, 2)  # [16, 32, 256]
        # return self.out_proj(result)

        #######
        # add #
        #######
        proj = nn.Linear(1849, 32)  # 把 token 数压到 32
        visual_32 = proj(visual.transpose(1, 2)).transpose(1, 2)  # visual [16,1849,256] -> [16,32,256]
        result = visual_32 + x
        return self.out_proj(result)

        #################
        # dual CA Q_txt #
        #################
        # for layer in self.decoder:
        #     # CA_1
        #     enhanced_txt = layer(x, visual) # [16,24,256]  Q:txt, K,V: visual
        #
        #     # CA_2
        #     learnable_visual_embed = nn.Parameter(torch.randn(visual.shape[0], visual.shape[1], visual.shape[2]) * 0.02) # 符合正态分布的初始化
        #     enhanced_visual = layer(learnable_visual_embed, visual) # [16,1849,256]
        #
        #     # maxpooling + add
        #     # pool = nn.AdaptiveAvgPool1d(output_size=24)
        #     # enhanced_visual = pool(enhanced_visual.permute(0, 2, 1))  # → [B, C, 24]
        #     # enhanced_visual = enhanced_visual.permute(0, 2, 1)  # → [B, 24, C]
        #     # result = enhanced_visual + enhanced_txt
        #
        #     # CA_3
        #     result = layer(enhanced_txt, enhanced_visual)

        # return self.out_proj(result)


        #################
        # dual CA Q_vis #
        #################
        # for layer in self.decoder:
        #     # CA_1
        #     enhanced_visual = layer(visual, x) # [16,1849,256] Q:visual, K,V: txt
        #
        #     # CA_2
        #     learnable_txt_embed = nn.Parameter(torch.randn(x.shape[0], x.shape[1], x.shape[2]) * 0.02)  # 符合正态分布的初始化
        #     enhanced_txt = layer(learnable_txt_embed, x) # [16,1849,256]
        #
        #     pool = nn.AdaptiveAvgPool1d(output_size=24)
        #     enhanced_visual = pool(enhanced_visual.permute(0, 2, 1))  # → [B, C, 24]
        #     enhanced_visual = enhanced_visual.permute(0, 2, 1)  # → [B, 24, C]
        #
        #     result = enhanced_visual + enhanced_txt
        #
        # return self.out_proj(result)



class ContextDecoderGCFiLM(nn.Module):
    def __init__(
        self,
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=6,
        visual_dim=512,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()

        # == 1. 对齐维度并加上 modality/type embedding ===========
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        # Learnable modality embeddings (1×D)  🔧新增
        self.modality_embed = nn.Parameter(torch.randn(2, transformer_width))

        # == 2. 自注意力 + 双向跨注意力的 DecoderLayer =============
        self.decoder = nn.ModuleList(
            [
                FusionDecoderLayer(
                    d_model=transformer_width,
                    nhead=transformer_heads,
                    dropout=dropout,
                )
                for _ in range(transformer_layers)
            ]
        )

        # == 3. 输出层 ================================
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim),
        )

        self.apply(self._init_weights)

    # ------------------------------------------------
    def forward(self, text, visual):
        """
        Args
        ----
        text   : (B, Lt, C_text)  —— token‑level文本特征
        visual : (B, Lv, C_vis)   —— patch/ROI‑level视觉特征
        """
        B, Lt, _ = text.shape
        _, Lv, _ = visual.shape

        # 1) 维度对齐 + 注入模态位置信息
        txt = self.text_proj(text) + self.modality_embed[0]
        vis = self.visual_proj(visual) + self.modality_embed[1]

        # 2) 逐层融合
        for layer in self.decoder:
            txt = layer(txt, vis)

        # 3) 只回传文本 query 的最终表示，也可返回两者
        return self.out_proj(txt)  # (B, Lt, C_vis)

    # ------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# =========================================================
class FusionDecoderLayer(nn.Module):
    """
    • step‑1: Text/Visual 各自 Self‑Attn（保留 intra‑modality 结构信息）
    • step‑2: 双向 Cross‑Attn（Text→Visual, Visual→Text）
    • step‑3: FiLM‑style 门控融合（显式学习“看图说话”或“以文解图”的强弱）
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        # Self‑Attention
        self.self_attn_txt = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn_vis = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Cross‑Attention
        self.cross_t2v = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_v2t = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # FiLM‑style gating 🔧新增
        self.gate_t = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.Sigmoid())
        self.gate_v = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.Sigmoid())

        # FFNs
        self.ffn_txt = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_vis = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    # --------------------------------------------------------
    def forward(self, txt, vis):
        # ----- (a) Intra‑modality Self‑Attention -----
        txt = txt + self.dropout(self.self_attn_txt(txt, txt, txt, need_weights=False)[0])
        vis = vis + self.dropout(self.self_attn_vis(vis, vis, vis, need_weights=False)[0])

        # ----- (b) Cross‑Attention ----------
        t2v = self.cross_t2v(query=txt, key=vis, value=vis, need_weights=False)[0]
        # v2t = self.cross_v2t(query=vis, key=txt, value=txt, need_weights=False)[0]

        # FiLM‑gate：学习“我需要多少另一模态的信息”
        txt = txt + self.gate_t(txt) * self.dropout(t2v)
        # vis = vis + self.gate_v(vis) * self.dropout(v2t)

        # ----- (c) FFN -----
        txt = txt + self.dropout(self.ffn_txt(txt))
        # vis = vis + self.dropout(self.ffn_vis(vis))

        return txt




