"""Task-conditioned sparse FFNs used by the shared Transformer."""

import torch
from torch import nn
from .registry import TASKS
from .task_private import task_linear


def validate_task_ids(ids, batch_size, num_tasks=len(TASKS)):
    if (
        ids.dtype != torch.int64
        or ids.shape != (batch_size,)
        or not batch_size
        or not bool(((ids >= 0) & (ids < num_tasks)).all())
    ):
        raise ValueError("Expected valid int64 task IDs [B]")


class TaskMoE(nn.Module):
    """Gate reads centered 768D task embeddings; selected FFNs process all frames."""

    def __init__(self, input_dim, output_dim, num_experts=15, top_k=3):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("Require 1 <= K <= N")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.num_experts, self.top_k = num_experts, top_k
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
            )
            for _ in range(num_experts)
        )
        self.gate = nn.Linear(768, num_experts)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def route(self, condition, task_ids=None):
        if condition.ndim != 2 or condition.shape[-1] != 768:
            raise ValueError("Expected CLIP task embeddings [B,768]")
        with torch.autocast(device_type=condition.device.type, enabled=False):
            logits = task_linear(self.gate, condition.float(), task_ids)
            selected, indices = torch.topk(logits, self.top_k, dim=-1)
            return dict(
                probabilities=logits.softmax(-1),
                topk_indices=indices,
                topk_weights=selected.softmax(-1),
            )

    def forward(self, x, condition, task_ids=None):
        if (
            x.ndim != 3
            or x.shape[-1] != self.input_dim
            or not len(x)
            or not x.shape[1]
            or condition.shape != (len(x), 768)
        ):
            raise ValueError("Invalid MoE feature/task shapes")
        route = self.route(condition, task_ids)
        indices, weights = route["topk_indices"], route["topk_weights"]
        mixed = torch.zeros(
            (*x.shape[:2], self.output_dim), device=x.device, dtype=torch.float32
        )
        for expert_id, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == expert_id)
            if not rows.numel():
                continue  # Unselected FFNs keep grad=None and no AdamW update.
            transformed = expert(x.index_select(0, rows).float())
            mixed = mixed.index_add(
                0, rows, transformed.float() * weights[rows, slots, None, None]
            )
        return mixed.to(transformed.dtype), route
