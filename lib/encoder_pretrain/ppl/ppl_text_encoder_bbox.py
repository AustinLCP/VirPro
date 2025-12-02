import open_clip
import torch
from torch import nn

from PPL.text_encoder import CLIPTextContextEncoderGC
from PPL.prompt_learner_bbox import PromptLearnerBbox
from PPL.decoder import PromptEncoderWithoutPositionembGC, ContextDecoderGC
import torch.nn.functional as F



class PPLTextEncoderBbox(nn.Module):

    def __init__(self):
        super().__init__()

        self.with_prompt_encoder = True
        # self.class_names = class_names
        self.num_classes = 3
        self.emb_dim = 512

        with torch.no_grad():
            self.text_encoder = CLIPTextContextEncoderGC()
            self.text_encoder.init_weights("clip_ckp/ViT-B-32.pt")

        # self.prompt_learner = PromptLearner(self.class_names, self.emb_dim)
        self.text_decoder = PromptEncoderWithoutPositionembGC() # prompt encoder
        self.visual_text_decoder = ContextDecoderGC() # context decoder


    def get_cls_token(self, visual_embed):
        cls_token = visual_embed.mean(dim=[2, 3])
        return cls_token


    def prompt_learner_init(self, class_names):
        prompt_learner = PromptLearnerBbox(class_names, 512)
        prompt_learner.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        prompt_learner.text_encoder = self.text_encoder
        prompt_learner.token_prefix_suffix_tokenizer_prompts_init()
        return prompt_learner

    def get_text_embedding(self, roi_text_prompt_list):
        text_embeddings = []
        nc_prompts_list = []
        nc_tokenized_prompts_list = []
        n_prompt = 0
        for class_names in roi_text_prompt_list:
            prompt_learner = self.prompt_learner_init(class_names)
            text_prompt, tokenized_prompts, nc_prompts, nc_tokenized_prompts = prompt_learner() # text_prompt: (prompt_bsz * n_cls, token_len, embed_dim), tokenized_prompts (prompt_bsz * n_cls, token_len)
            n_prompt = text_prompt.shape[0] // self.num_classes

            nc_prompts_list.append(nc_prompts)
            nc_tokenized_prompts_list.append(nc_tokenized_prompts)

            text_embedding = self.text_encoder(text_prompt, tokenized_prompts) # (prompt_bsz * n_cls, C)
            text_embeddings.append(text_embedding)


        text_emb = torch.stack(text_embeddings, dim=0)  # (B, prompt_bsz * n_cls, C)
        # nc_prompts = torch.stack(nc_prompts_list, dim=0).mean(dim=0)  # (prompt_bsz * n_cls, C)
        # nc_tokenized_prompts = nc_tokenized_prompts_list[0]
        return text_emb, nc_prompts_list, nc_tokenized_prompts_list, n_prompt

    # spatial adaption
    # visual_embeddings: (B, C, H, W)
    # cls_token: (B, C)
    def forward(self, cls_token, visual_embeddings, roi_text_prompt_list):
        B, C, H, W = visual_embeddings.shape

        # text_prompt, tokenized_prompts, nc_prompts, nc_tokenized_prompts = self.prompt_learner() # text_prompt: (prompt_bsz * n_cls, token_len, embed_dim)
        # n_prompt = text_prompt.shape[0] // self.num_classes # 就是 prompt_bsz , 一个 batch 中每个 keypoint 对应的 prompt 数量 (attribute 数量)
        # text encoder 处理过 prompts 之后的 text embeddings
        # text_embeddings = self.text_encoder(text_prompt, tokenized_prompts).expand(B, -1, -1) # 形状 (B, prompt_bsz * n_cls, C) C 就是 emb_dim, prompt_bsz * n_cls 就是 num_token


        text_embeddings, nc_prompts_list,  nc_tokenized_prompts_list, n_prompt = self.get_text_embedding(roi_text_prompt_list) # text_embeddings: (B, prompt_bsz * n_cls, C)

        # cross-attn to enhance text emb
        visual_tokens = torch.cat([cls_token.reshape(B, C, 1), visual_embeddings.reshape(B, C, H * W)],
                                  dim=2).permute(0, 2, 1)  # (B, H * W + 1, C)

        # model the relation of prompts
        if self.with_prompt_encoder:
            # μ=aug_text_emb
            # 所有 keypoint 的每个 prompt 对应的加权平均值
            # eg. 在这张图里，‘car’ 这个词最典型的大致长什么样？
            aug_text_emb = self.text_decoder(text_embeddings) # prompt_encoder 就是 text_decoder
            aug_text_emb = (text_embeddings * 0.7 + aug_text_emb * 0.3).view(B, self.num_classes, n_prompt, -1) # (B, n_cls, n_prompt, C)
        else:
            aug_text_emb = text_embeddings.view(B, self.num_classes, n_prompt, -1)

        # σ=text_st_dev,
        # context_decoder = visual-text decoder
        # 所有 keypoint 的每个 prompt 对应的方差
        # 这条 prompt 在给定图像里还能有多大弹性，即这条prompt所描述的外观在图中可能的变化大小（视角、遮挡、光照）
        text_st_dev = self.visual_text_decoder(text_embeddings, visual_tokens).view(B, self.num_classes, n_prompt, C) # (B, n_cls, n_prompt, C)

        # E_prompts = prompt_embeddings
        # μ=aug_text_emb, σ=text_st_dev,
        if self.num_classes > 1:
            prompt_embeddings = self.reparameterize(aug_text_emb, text_st_dev) # (B, n_cls, n_components, C)
        else:
            prompt_embeddings = self.reparameterize_single_keypoint(aug_text_emb, text_st_dev)

        # score map
        # visual_embeddings_norm = F.normalize(visual_embeddings, dim=1, p=2)
        # prompt_embeddings_norm = F.normalize(prompt_embeddings, dim=-1, p=2)
        # score_map = torch.einsum('bchw,bnkc->bnkhw', visual_embeddings_norm, prompt_embeddings_norm).reshape(B, -1, H, W)
        # x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)

        # loss prompt calculation
        loss = 0
        for nc_prompts, nc_tokenized_prompts in zip(nc_prompts_list, nc_tokenized_prompts_list):
            nc_text_embeddings = self.text_encoder(nc_prompts, nc_tokenized_prompts).view(self.num_classes, n_prompt, -1).expand(B, -1, -1, -1)
            loss_prompt = self.loss_prompt(aug_text_emb, text_st_dev, nc_text_embeddings)
            loss += loss_prompt

        loss = loss / len(nc_prompts_list)
        return prompt_embeddings, loss # aug_text_emb, text_st_dev, x_orig[self.score_concat_index], score_map


    def reparameterize_single_keypoint(self, mu, logvar):

        batch_size, keypoint_num, n_prompt, emb_dim = mu.size() # (B, 1, n_prompts, emb_dim)

        std = torch.exp(0.5 * logvar)  # (B, 1, n_prompts, emb_dim)
        eps = torch.randn_like(std)
        sample = eps * std + mu  # (B, 1, n_prompts, emb_dim)
        sample = sample.reshape(batch_size, keypoint_num*n_prompt, emb_dim)

        return sample


    def reparameterize(self, mu, logvar, n_components=10):
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
            sample = eps * std + mu  # (B, n_cls, prompt_bsz, C)
            sampled_idx = torch.randint(0, attr_num, (batch_size, 1, 1, 1)).to(
                mu.device)  # 每个 batch 随机选一个属性 id (B,1,1,1)
            sampled_attr = torch.gather(sample, 2, sampled_idx.expand(-1, keypoint_num, -1, emb_dim)).squeeze(
                2)  # (B, n_cls, C) 对于 batch 中的每个 keypoint，随机选 1 个属性 的采样向量
            probs.append(sampled_attr)
            # probs.append(mu + logvar * eps)

        probs = torch.stack(probs, dim=1)  # (B, n_components, n_cls, C)

        probs = probs.mean(dim=1) # (B, n_cls, C) 用平均池化把每个 keypoint 对应的 n_components 个 prompt 合成一条 prompt

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