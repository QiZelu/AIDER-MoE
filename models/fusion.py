"""两层时序融合：自注意力 + Top-2 路由专家 + 常开共享专家。"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .moe import TaskMoE


class TaskTransformerBlock(nn.Module):
    def __init__(self, seed, layer_index, width=1536, heads=12):
        super().__init__()
        # 保持历史每层的初始化顺序：LN、Attention、路由专家、Gate。
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 810000 + layer_index)
            self.norm = nn.LayerNorm(width)
            self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
            nn.init.xavier_uniform_(self.attention.in_proj_weight)
            nn.init.zeros_(self.attention.in_proj_bias)
            nn.init.xavier_uniform_(self.attention.out_proj.weight)
            nn.init.zeros_(self.attention.out_proj.bias)
            self.moe = TaskMoE(width, width, num_experts=7, top_k=2)

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 830000 + layer_index)
            self.shared_expert = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            nn.init.zeros_(self.shared_expert[-1].weight)
            nn.init.zeros_(self.shared_expert[-1].bias)

    def forward(self, features, condition, task_ids=None):
        normalized = self.norm(features.float())
        attention, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        features = features + attention
        routed, route = self.moe(features, condition, task_ids)
        shared = self.shared_expert(features.float())
        route['shared_output_norm'] = shared.float().norm(dim=(1, 2))
        route['routed_output_norm'] = routed.float().norm(dim=(1, 2))
        return features + routed + shared.to(routed.dtype), route


class TransformerFusion(nn.Module):
    def __init__(self, seed, position_encoding, depth=2, width=1536, heads=12):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TaskTransformerBlock(seed, layer_index, width, heads) for layer_index in range(depth)]
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 300000)
            head = nn.Sequential(
                nn.Linear(width, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.uniform_(layer.weight, -0.1, 0.1)
                    nn.init.zeros_(layer.bias)
        # 名称保留为 heads.shared，以兼容已经训练的检查点。
        self.heads = nn.ModuleDict({'shared': head})
        self.final_norm = nn.LayerNorm(width)
        self.use_checkpoint = True
        self.register_buffer('position_encoding', position_encoding.clone())

    def forward(self, features, condition, task_ids=None):
        features = features + self.position_encoding
        routes = {}
        for layer_index, block in enumerate(self.blocks):
            if self.training and torch.is_grad_enabled() and self.use_checkpoint:
                features, route = checkpoint(block, features, condition, task_ids, use_reentrant=False)
            else:
                features, route = block(features, condition, task_ids)
            routes[f'transformer.layer_{layer_index:02d}.shared'] = route
        return self.final_norm(features.float()), routes
