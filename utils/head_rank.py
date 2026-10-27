"""第9轮私有任务模型的冻结 Head 排序微调协议。"""

from models.registry import TASKS
from utils.private_task_modules import POLICIES

CONTRACT = 'aider_epoch9_frozen_private_heads_smoothl1_rank1_homogeneous_v1'
T2VQA_CONTRACT = 'aider_epoch9_frozen_private_heads_smoothl1_t2vqa_rank1_homogeneous_v1'
LOGISTIC_KIND = 'sign_only_pairwise_logistic_all_nontied_unordered_pairs'
T2VQA_KIND = 't2vqa_relu_all_ordered_pairs_differentiable_max_scale'
RANK_CONTRACTS = {LOGISTIC_KIND: CONTRACT, T2VQA_KIND: T2VQA_CONTRACT}
SAMPLING = 'homogeneous_task_round_robin_cpu_seed_plus_epoch'


def validate_config(config):
    p, t, a, m = (config[k] for k in ('head_rank', 'training', 'ablation', 'model'))
    expected_contract = RANK_CONTRACTS.get(p.get('rank_kind'))
    if (expected_contract is None or p['contract'] != expected_contract
            or a['contract'] != expected_contract
            or config['experiment']['contract'] != expected_contract):
        raise ValueError('Wrong Head ranking contract')
    if (m.get('head_policy') != 'private' or m.get('film_video_condition') != 'video_only'
            or any(m.get(k) != v for k, v in POLICIES.items())
            or 'private_task_modules' not in config):
        raise ValueError('Expected task-private FiLM/alpha/Gates and four inherited Heads')
    expected = dict(lr=1e-4, weight_decay=.05, warmup_steps=116, epochs=10,
                    gradient_clip=None, gradient_accumulation=1, module_lrs={},
                    rank_loss_weight=1., smooth_l1_beta=.1, loss_kind='smooth_l1_rank',
                    scheduler='cosine', resume=None, seed=42)
    if any(t.get(k) != v for k, v in expected.items()):
        raise ValueError('Head ranking training settings differ from the fixed protocol')
    if (a['depth'] != 2 or a['routed_experts'] != 7 or a['routed_top_k'] != 2
            or a['layerwise_always_on_experts'] != 1 or a['steps_per_epoch'] != 232
            or a['total_steps'] != 2320 or a['tasks'] != list(TASKS)
            or a['task_ids'] != list(range(4)) or a['split'] != 'original_461_115'
            or config['task']['order'] != list(TASKS)
            or config['loader']['video_batch_size'] != 8
            or config['loader']['sampler_seed'] != 9042
            or config['mixed_task_experiment']['sampling'] != SAMPLING):
        raise ValueError('Unexpected Head ranking model, task map or sampler')
    if (p['source_epoch'] != 9 or len(p['source_sha256']) != 64
            or p['source_checkpoint'].rsplit('/', 1)[-1] != 'epoch9.pt'):
        raise ValueError('Expected pinned epoch9.pt source')
