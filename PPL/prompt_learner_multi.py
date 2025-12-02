import torch
from torch import nn
from PPL.text_encoder import CLIPTextContextEncoderGC
import open_clip

class PromptLearnerMulti(nn.Module):

    def __init__(self, class_names, embed_dim, n_ctx=8, n_prompt=32, prompt_bsz=8, **kwargs):
        """
        Args:
            class_names: keypoints name, list
            n_ctx: learnable tokens, int
            n_prompt: number of attribute, int
            prompt_bsz: number of attribute in a batch, int
            **kwargs:
        """
        super().__init__()
        self.class_names = class_names
        self.name_lens = [len(name.split()) for name in class_names] # 每一个 keypoint string 的长度
        self.n_cls = len(class_names) # keypoint 的数量
        self.n_prompt = n_prompt      # 同一个 keypoint 对应的 attribute 数量 (列, Np)
        self.n_ctx = n_ctx            # 同一个 keypoint 的一个 attribute 对应的token数量 (行,L)prompt_size
        self.prompt_bsz = prompt_bsz     # 一个 batch 中加载的 attribute 避免数量 (一张图片中 Roi 数量，到时候 PPL 中会 expand 到 Batch size)
        assert n_prompt % self.prompt_bsz == 0
        self.n_iter = int(n_prompt / self.prompt_bsz)
        self.embed_dim = embed_dim

        # ctx: 一个 keypoint 对应的所有 attribute 的所有token 的整合向量
        self.ctx = nn.Parameter(torch.randn(n_prompt, n_ctx, embed_dim)) # (n_prompt, n_ctx, embed_dim)
        nn.init.trunc_normal_(self.ctx) # 使 ctx 的值符合 (截断的) 正态分布
        self.ctx = nn.Parameter(self.ctx.to('cuda'))

        # pos == 0: keypoint 放在前面 pos == 1: keypoint 插在中间 pos == 2: keypoint 放在后面
        self.pos = [2 for _ in range(n_prompt)]  # (n_prompt,)
        self.pos = torch.tensor(self.pos, device='cuda')

        self.iter_idx = 0

        self.text_encoder = None

    def prompt_learner_init(self):
        # text_encoder init
        with torch.no_grad():
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            # self.text_encoder = CLIPTextContextEncoderGC()
            # self.text_encoder.init_weights("clip_ckp/ViT-B-32.pt")

        # token_prefix, token_suffix, tokenized_prompts init
        prompt_prefix = ' '.join(['X'] * self.n_ctx)  # 'X X X X X X ...' (n_ctx 个 X)
        prompts = [prompt_prefix + ' ' + name for name in
                   self.class_names]  # ["X X ... keyword_1", "X X ... keyword_2"]
        tokenized_prompts = torch.cat([self.tokenizer(p, context_length=self.text_encoder.context_length) for p in prompts])
        tokenized_prompts = tokenized_prompts.to('cuda')
        self.tokenized_prompts = tokenized_prompts
        with torch.no_grad():
            embedding = self.text_encoder.token_embedding(tokenized_prompts)
        self.register_buffer('token_prefix', embedding[:, :1, :])
        self.register_buffer('token_suffix', embedding[:, 1 + self.n_ctx:, :])

        # nc_token_prefix, nc_token_suffix, nc_tokenized_prompts init
        nc_prompts = [prompt_prefix]
        nc_prompts = [nc_prompts for _ in range(self.n_cls)]
        nc_tokenized_prompts = torch.cat([self.tokenizer(p, context_length=self.text_encoder.context_length) for p in nc_prompts])
        nc_tokenized_prompts = nc_tokenized_prompts.to('cuda')
        self.nc_tokenized_prompts = nc_tokenized_prompts
        with torch.no_grad():
            embedding = self.text_encoder.token_embedding(nc_tokenized_prompts)
        self.register_buffer('nc_token_prefix', embedding[:, :1, :])
        self.register_buffer('nc_token_suffix', embedding[:, 1 + self.n_ctx:, :])



    def forward(self, test=False):
        ###########################################
        # 从 ctx 里随机抽取 prompt_bsz 个 attribute #
        ###########################################
        if self.n_iter > 1 and (not test):
            if self.iter_idx == 0:
                self.select_idx = torch.randperm(self.n_prompt)
            batch_idx = self.select_idx[self.iter_idx * self.prompt_bsz:(self.iter_idx + 1) * self.prompt_bsz] # (prompt_bsz,) 如果prompt_bsz=4, [[0,1,2,3] [4,5,6,7] ....] batch size
            ctx = self.ctx[batch_idx] # (prompt_bsz, n_ctx, embed_dim)
            pos = self.pos[batch_idx] # (prompt_bsz,)

            self.iter_idx += 1
            if self.iter_idx == self.n_iter:
                self.iter_idx = 0
        else:
            ctx = self.ctx
            pos = self.pos

        ############################################
        # Generalized Keypoint Placement (GKP) 插入 # keypoint name是放在 suffix 前半部分的
        ############################################
        prompt_size = ctx.shape[0] # 就是 prompt_bsz
        # 所有 keypoint 与当前 batch 中的 attribute 两两配对 (给每个 keypoint 配 prompt_bsz 个 attribute)
        # 最初 self.tokenized_prompts 的 shape 为 (n_cls, token_len)
        tokenized_prompts = self.tokenized_prompts.unsqueeze(1).repeat(1, prompt_size, 1).view(self.n_cls * prompt_size, -1) # (n_cls * prompt_bsz, token_len)
        n_cls = self.n_cls

        # keypoint name 放最后
        # [prefix tokens] + [attribute tokens (ctx)] + [keypoint name tokens + suffix tokens]
        ctx_end = ctx[pos == 2] # (num_pos=2, n_ctx, embed_dim)
        n_end = ctx_end.shape[0] # num of pos==2 的 attribute
        # token_prefix: (n_cls, 1, emd_dim), 每个 keypoint 都要和 n_end(num_pos=2的数量) 个 attributes 组合 → 所以 prefix 也要重复。
        prefix = self.token_prefix.unsqueeze(1).repeat(1, n_end, 1, 1) # (n_cls, n_end, 1, embed_dim)
        # token_suffix: (n_cls, suffix_len, emd_dim)
        suffix = self.token_suffix.unsqueeze(1).repeat(1, n_end, 1, 1) # (n_cls, n_end, suffix_len, emd_dim)
        ctx_end = ctx_end.unsqueeze(0).repeat(n_cls, 1, 1, 1)          # (n_cls, n_end, n_ctx, embed_dim)
        prompts_end = torch.cat([prefix, ctx_end, suffix], dim=2) # (n_cls, 1 + n_ctx + suffix_len, embed_dim) # 顺序固定

        # keypoint name 插中间
        # [prefix] + [ctx 前半] + [keypoint name] + [ctx 后半] + [suffix]
        ctx_middle = ctx[pos == 1]
        n_middle = ctx_middle.shape[0]
        prompts_middle = []
        half_n_ctx = self.n_ctx // 2
        for i in range(n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i + 1, :, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            class_i = self.token_suffix[i:i + 1, :name_len, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            suffix_i = self.token_suffix[i:i + 1, name_len:, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            ctx_i_half1 = ctx_middle[:, :half_n_ctx, :].unsqueeze(0)
            ctx_i_half2 = ctx_middle[:, half_n_ctx:, :].unsqueeze(0)
            prompt = torch.cat([
                prefix_i,  # (1, n_middle, 1, dim)
                ctx_i_half1,  # (1, n_middle, n_ctx//2, dim)
                class_i,  # (1, n_middle, name_len, dim)
                ctx_i_half2,  # (1, n_middle, n_ctx//2, dim)
                suffix_i  # (1, n_middle, *, dim)
            ], dim=2)
            prompts_middle.append(prompt)
        prompts_middle = torch.cat(prompts_middle, dim=0)

        # keypoint name 放前面
        # [prefix] + [keypoint name] + [attribute tokens (ctx)] + [suffix]
        ctx_front = ctx[pos == 0]
        n_front = ctx_front.shape[0]
        prompts_front = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i + 1, :, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            class_i = self.token_suffix[i:i + 1, :name_len, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            suffix_i = self.token_suffix[i:i + 1, name_len:, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            ctx_i = ctx_front.unsqueeze(0)
            prompt = torch.cat([
                prefix_i,  # (1, n_front, 1, dim)
                class_i,  # (1, n_front, name_len, dim)
                ctx_i,  # (1, n_front, n_ctx, dim)
                suffix_i  # (1, n_front, *, dim)
            ], dim=2)
            prompts_front.append(prompt)
        prompts_front = torch.cat(prompts_front, dim=0)
        # 一个 batch 中所有 keypoints 对应的所有 prompt (含 keypoint)
        # (prompt_size * n_cls, prompts_end + prompts_mid + prompts_front, embed_dim)
        prompts = torch.cat([prompts_end, prompts_middle, prompts_front], dim=1).view(prompt_size * n_cls, -1, self.embed_dim) # (prompt_size * n_cls, token_len, embed_dim)
        if test:
            return prompts, tokenized_prompts
        else:
            nc_prompts, nc_tokenized_prompts = self.only_prefix()
            return prompts, tokenized_prompts, nc_prompts, nc_tokenized_prompts

    # 构造 “不带 keypoint name” 的 prompt, 用于 diversity loss calculation
    def only_prefix(self):
        ctx = self.ctx
        prompt_size = ctx.shape[0]
        nc_tokenized_prompts = self.nc_tokenized_prompts.unsqueeze(1).repeat(1, prompt_size, 1).view(self.n_cls * prompt_size, -1)
        prefix = self.nc_token_prefix.unsqueeze(1).repeat(1, prompt_size, 1, 1)
        suffix = self.nc_token_suffix.unsqueeze(1).repeat(1, prompt_size, 1, 1)
        ctx_end = ctx.unsqueeze(0).repeat(self.n_cls, 1, 1, 1)
        # 一个 batch 中所有 keypoints 对应的所有 prompt (不含 keypoint)
        nc_prompts = torch.cat([prefix, ctx_end, suffix], dim=2).view(prompt_size * self.n_cls, -1, self.embed_dim)
        return nc_prompts, nc_tokenized_prompts      # (prompt_size * n_cls, token_len, embed_dim)
