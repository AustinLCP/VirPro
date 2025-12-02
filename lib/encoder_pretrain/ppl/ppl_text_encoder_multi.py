import torch
from torch import nn

from PPL.text_encoder import CLIPTextContextEncoderGC
from PPL.prompt_learner_multi import PromptLearnerMulti
from PPL.decoder import PromptEncoderWithoutPositionembGC, ContextDecoderGC, ContextDecoderGCFiLM
import torch.nn.functional as F



class PPLTextEncoderMulti(nn.Module):

    def __init__(self, class_names, n_RoI=3):
        super().__init__()

        self.with_prompt_encoder = True
        self.class_names = class_names
        # self.num_classes = len(self.class_names)
        self.emb_dim = 512

        with torch.no_grad():
            self.text_encoder = CLIPTextContextEncoderGC().to('cuda')
            self.text_encoder.init_weights("clip_ckp/ViT-B-32.pt")

        self.n_RoI = n_RoI
        self.prompt_learner_list = []
        for i in range(self.n_RoI):
            prompt_learner = PromptLearnerMulti(self.class_names, self.emb_dim).to('cuda') # 一个 prompt_learner 代表一个 RoI 的所有 prompts
            prompt_learner.text_encoder = self.text_encoder
            prompt_learner.prompt_learner_init()
            self.prompt_learner_list.append(prompt_learner)

        self.text_decoder = PromptEncoderWithoutPositionembGC().to('cuda') # prompt encoder
        self.visual_text_decoder = ContextDecoderGC().to('cuda') # context decoder
        # self.visual_text_decoder = ContextDecoderGCFiLM().to('cuda')  # context decoder


    def get_cls_token(self, visual_embed):
        cls_token = visual_embed.mean(dim=[2, 3])
        return cls_token


    # spatial adaption
    # visual_embeddings: (B, C, H, W)
    # cls_token: (B, C)
    def forward(self, cls_token, visual_embeddings):

        B, C, H, W = visual_embeddings.shape

        text_embeddings_list = []
        nc_text_embeddings_list = []

        n_prompt = 0
        for prompt_learner in self.prompt_learner_list:
            text_prompt, tokenized_prompts, nc_prompts, nc_tokenized_prompts = prompt_learner() # text_prompt: (prompt_bsz * n_cls, token_len, embed_dim) n_cls =1

            n_prompt = text_prompt.shape[0]  # 就是 prompt_bsz , 一个 batch 中每个 roi 对应的 prompt 数量 (attribute 数量)

            text_embeds = self.text_encoder(text_prompt, tokenized_prompts).expand(B, -1, -1)  # 形状 (B, prompt_bsz * n_cls, C)
            text_embeddings_list.append(text_embeds)

            nc_text_embeds = self.text_encoder(nc_prompts, nc_tokenized_prompts).view(len(self.class_names), n_prompt, -1).expand(B, -1, -1, -1)  # 形状 (B, prompt_bsz * n_cls, C)
            nc_text_embeddings_list.append(nc_text_embeds)

        text_embeddings = torch.stack(text_embeddings_list, dim=1)  # (B, n_roi, prompt_bsz * n_cls, C) (8,3,4,512)
        nc_text_embeddings = torch.stack(nc_text_embeddings_list, dim=1)  # (B, n_roi, n_cls, prompt_bsz, C)

        # 一个 batch 里面所有 RoI 的所有 text embeddings (8,12,512) 3个roi x 每个roi 4个text embed = 12
        text_embeddings = text_embeddings.reshape(text_embeddings.size(0), -1, text_embeddings.size(3)) # (B, n_roi*prompt_bsz*n_cls, C) n_cls = 1, 相当于让 n_roi 代替 n_cls
        nc_text_embeddings = nc_text_embeddings.mean(dim=[2]) # (B, n_roi, prompt_bsz, C) # 相当于让 n_roi 代替 n_cls

        # add cls token to visual embedding
        visual_tokens = torch.cat([cls_token.reshape(B, C, 1), visual_embeddings.reshape(B, C, H * W)],
                                  dim=2).permute(0, 2, 1)  # (B, H * W + 1, C)

        # model the relation of prompts
        if self.with_prompt_encoder:
            # μ=aug_text_emb
            # 所有 keypoint 的每个 prompt 对应的加权平均值
            # eg. 在这张图里，‘car’ 这个词最典型的大致长什么样？
            aug_text_emb = self.text_decoder(text_embeddings) # prompt_encoder 就是 text_decoder
            aug_text_emb = (text_embeddings * 0.7 + aug_text_emb * 0.3).view(B, self.n_RoI, n_prompt, -1) # (B, n_cls*n_roi, n_prompt, C) n_cls = 1
        else:
            aug_text_emb = text_embeddings.view(B, self.n_RoI, n_prompt, -1)

        # σ=text_st_dev,
        # context_decoder = visual-text decoder
        # 所有 keypoint 的每个 prompt 对应的方差
        # 这条 prompt 在给定图像里还能有多大弹性，即这条prompt所描述的外观在图中可能的变化大小（视角、遮挡、光照）
        text_st_dev = self.visual_text_decoder(text_embeddings, visual_tokens).view(B, self.n_RoI, n_prompt, C) # (B, n_cls*n_roi, n_prompt, C) n_cls = 1

        # E_prompts = prompt_embeddings
        # μ=aug_text_emb, σ=text_st_dev,
        prompt_embeddings = self.reparameterize(aug_text_emb, text_st_dev) # (B, n_cls*n_roi, C) n_cls = 1, n_component 被平均池化了


        # score map
        # visual_embeddings_norm = F.normalize(visual_embeddings, dim=1, p=2)
        # prompt_embeddings_norm = F.normalize(prompt_embeddings, dim=-1, p=2)
        # score_map = torch.einsum('bchw,bnkc->bnkhw', visual_embeddings_norm, prompt_embeddings_norm).reshape(B, -1, H, W)
        # x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)

        # loss prompt calculation
        loss_prompt = self.loss_prompt(aug_text_emb, text_st_dev, nc_text_embeddings)


        return prompt_embeddings, loss_prompt  # (B, n_cls*n_roi, C)


    def reparameterize_single_keypoint(self, mu, logvar):

        batch_size, keypoint_num, n_prompt, emb_dim = mu.size() # (B, 1, n_prompts, emb_dim)

        std = torch.exp(0.5 * logvar)  # (B, 1, n_prompts, emb_dim)
        eps = torch.randn_like(std)
        sample = eps * std + mu  # (B, 1, n_prompts, emb_dim)
        sample = sample.reshape(batch_size, keypoint_num*n_prompt, emb_dim)

        return sample


    def reparameterize(self, mu, logvar, n_components=8):
        # n_components: 从高斯分布中抽取的 prompt 数量, 也是最终保留的 attribute 的数量
        probs = []
        batch_size, keypoint_num, attr_num, emb_dim = mu.size()
        # logvar = torch.mean(mu ** 2 + logvar ** 2, dim=2, keepdim=True) - mu ** 2
        for i in range(n_components):
            ''' the other implementations '''
            # z=ϵσ+μ
            std = torch.exp(0.5 * logvar)  # std = log σ² (B, n_cls, prompt_bsz, C)
            eps = torch.randn_like(std)  # (B, n_cls, prompt_bsz, C) 相当于一个“骰子”, 每次采样都不一样，但平均值是 0、方差是 1
            # probs.append(eps.mul(std).add_(mu))
            sample = eps * std + mu  # (B, n_roi, prompt_bsz, C)
            sampled_idx = torch.randint(0, attr_num, (batch_size, 1, 1, 1)).to(
                mu.device)  # 每个 batch 随机选一个属性 id (B,1,1,1)
            sampled_attr = torch.gather(sample, 2, sampled_idx.expand(-1, keypoint_num, -1, emb_dim)).squeeze(
                2)  # (B, n_roi, C) 对于 batch 中的每个 keypoint，随机选 1 个属性 的采样向量
            probs.append(sampled_attr)
            # probs.append(mu + logvar * eps)

        probs = torch.stack(probs, dim=1)  # (B, n_components, n_roi, C)

        # max pooling (+mlp) fusion
        probs = probs.mean(dim=1) # (B, n_roi, C) 用平均池化把每个 keypoint 对应的 n_components 个 prompt 合成一条 prompt
        # mlp = SimpleMLP(512,256,512)
        # probs = mlp(probs)

        # simpl_mlp = SimpleMLP(probs.shape[-1],probs.shape[-1],probs.shape[-1])
        # probs = simpl_mlp(probs)

        # add fusion
        # probs = probs.sum(dim=1)

        # concat + mlp
        # prompt_fusion = PromptFusionConcat(n_components=n_components, C=512)
        # probs = prompt_fusion(probs)

        # conv1d + mlp fusion
        # prompt_fusion = PromptFusionConv(n_components=n_components)
        # probs = prompt_fusion(probs)

        return probs


    # mu / sigma: (B, n_keypoint, n_prompt, embed_dim)
    def loss_prompt(self, mu, sigma, nc_text_embeddings):
        num_k_a = mu.size(2)
        mus, logvars = mu.split(1, 2), sigma.split(1, 2)
        loss = 0
        # kl_div
        for idx in range(num_k_a):
            mean = mus[idx].squeeze(2)
            logvar = logvars[idx].squeeze(2)
            prior_mean = torch.zeros_like(mean)
            prior_std = torch.ones_like(logvar)
            prior = torch.distributions.Normal(loc=prior_mean, scale=prior_std)
            post = torch.distributions.Normal(loc=mean, scale=torch.exp(0.5 * logvar))
            # loss += (-0.5 * torch.sum(1. + logvar - mean ** 2 - logvar.exp(), dim=1)).mean()
            loss += torch.distributions.kl_divergence(post, prior).mean()
        kl_div = loss / num_k_a

        # l_div
        n_k = mu.size(1)
        B = mu.size(0)
        dissimilar_loss = 0
        nc_text_embeddings = F.normalize(nc_text_embeddings, p=2, dim=-1)
        for idx in range(n_k):
            nc_text_embedding = nc_text_embeddings[:, idx]
            dis = torch.matmul(nc_text_embedding, nc_text_embedding.transpose(1, 2).contiguous())
            batch_eye = torch.eye(nc_text_embedding.shape[1], dtype=dis.dtype, device=dis.device).unsqueeze(
                0).repeat(B, 1, 1)
            dissimilar_loss += ((dis - batch_eye) ** 2).mean()  # 0.2762
        l_div = dissimilar_loss / n_k #

        return kl_div*2 + l_div



class SimpleMLP(nn.Module):
    """
    输入 :  (B, n_roi, C)
    输出 :  (B, n_roi, out_dim)
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),          # 可选：归一化有助于训练稳定
            nn.Linear(in_dim, hidden_dim), # 线性层
            nn.ReLU(inplace=True),         # 激活
            nn.Linear(hidden_dim, out_dim) # 线性层
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape                      # (B, n_roi, C)
        x = x.view(B * N, C)                   # 合并 batch 和 roi，得到 (B*n_roi, C)
        x = self.mlp(x)                        # MLP 逐行处理
        x = x.view(B, N, -1)                   # 还原 (B, n_roi, out_dim)
        return x




class PromptFusionConv(nn.Module):
    def __init__(self, n_components: int):
        super().__init__()
        # 以 n_components 作为 in_channels，在长度维上卷积
        self.conv = nn.Conv1d(
            in_channels=n_components,
            out_channels=1,
            kernel_size=1,   # 线性加权融合
            bias=False       # 如需偏置可设 True
        )

    def forward(self, x):  # x: (B, n_components, n_roi, C)
        B, n_components, n_roi, C = x.shape
        # assert n_components == self.conv.in_channels, \
        #     f"输入的 n_components={n_components} 与 conv.in_channels 不一致"

        # 把 n_components 调到通道维，C 当作“序列长度”
        x = x.permute(0, 2, 1, 3).contiguous()   # (B, n_roi, n_components, C)
        x = x.view(B * n_roi, n_components, C)   # (B*n_roi, n_components, C)

        # Conv1d 融合：对每个 C 位置做 n_components→1 的线性组合
        x = self.conv(x)                         # (B*n_roi, 1, C)
        x = x.squeeze(1)                         # (B*n_roi, C)

        # 还原回 (B, n_roi, C)
        x = x.view(B, n_roi, C)
        return x



class PromptFusionConcat(nn.Module):
    def __init__(self, n_components, C):
        super().__init__()
        self.n_components = n_components
        self.mlp = nn.Sequential(
            nn.Linear(n_components * C, C),
            nn.ReLU(),
            nn.Linear(C, C)
        )

    def forward(self, x):  # x: (B, n_components, n_roi, C)
        B, n_components, n_roi, C = x.shape
        x = x.permute(0, 2, 1, 3)  # (B, n_roi, n_components, C)

        # 用 concat 而不是 reshape：
        x = torch.cat([x[:, :, i, :] for i in range(n_components)], dim=-1)  # (B, n_roi, n_components * C)

        x = self.mlp(x)  # -> (B, n_roi, C)
        return x
