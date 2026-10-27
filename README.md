# AIDER-main：任务私有 FiLM / α / Gate / Head，无梯度裁剪

本目录整理自 `AIDER/outputs/multitask/runs/film_video_only_private_task_modules_no_clip_seed42`。
保留原始 best（第7轮）、last（第10轮）、epoch9（第9轮）检查点，原实验配置、完整10轮训练记录、评估表及诊断文件均原样复制。
模型推理前向沿用 AIDER 实现。本副本在路径迁移与严格限定的历史检查点加载支持之外，
增加可选的任务—专家互信息训练损失，原 AIDER 目录未作修改。

## 文件位置

- `models/`：完整模型及本地 VITAL / SlowFast / BLIP 实现。
- `engine/`、`dataset/`、`utils/`：训练、评估、输入处理和检查点校验。
- `configs/model.yaml`：默认的本版本完整配置；历史绝对路径在加载时自动迁移到当前目录。
- `configs/film_video_only_private_task_modules_no_clip.yaml`：原始入口配置。
- `outputs/multitask/runs/film_video_only_private_task_modules_no_clip_seed42/`：三份检查点和全部实验记录。
- `outputs/multitask/runs/film_video_only_private_heads_seed42/`：初始化及指标对照所需的只读配置、审计、评估记录，不包含对照模型权重。
- `pretrained_models/`：训练来源记录中登记的 VITAL、SlowFast、BLIP、BERT 和 CLIP 模型／词表文件，全部为实体副本。
- `artifacts/`：固定四任务 CLIP 表、位置编码、seed42 未训练初始化及校验常量。
- `data/AIGVQA-DB/`：原始461／115视频训练／测试清单，标签未变。
- `package_manifest.json`：每个打包文件的 SHA256、来源及三份原始 checkpoint 的固定哈希。
- `verification/`：依赖版本、复制校验、模型加载及单测报告。

## 实际模型

四任务依次为 static、temporal、dynamic、text_video。输入8帧，三分支宽度分别为512、256、768，融合宽度1536。
任务 FiLM 共12个私有调制器、4个私有 α 生成器、每层4个私有 Gate（共8个）、4个私有评分 Head。
视频残差调制和三个跨分支视频摘要、注意力、路由专家池及常开 FFN 仍在任务之间共享。
两层 MoE Transformer，每层12个注意力头、7个 Top-2 路由专家及1个常开共享 FFN。
配置中的 `readout_spec.depth=12` 是历史兼容字段；实际深度由 `ablation.depth=2` 决定。

原实验训练为混合 batch8，10轮，每轮231步，共2310次更新；模型seed42，采样seed9042～9051；
仅 FP32 Smooth L1（β=0.1），BF16前向、FP32可训练参数及Adam状态；主LR=1e-4、BLIP文本LR=1e-5，
AdamW weight decay=0.05，warmup116后cosine，**不裁剪梯度**。完整微调后的BLIP文本权重已包含在每份检查点中。

## 可选任务—专家互信息损失

`training.moe_mi_weight` 默认为 `0.0`；大于零时，训练与预检均使用
`total_loss = SmoothL1 - moe_mi_weight * mean_layer I(task; expert)`。
可先以 `0.001` 为实验起点，并用验证集选择权重。

当前 Gate 仅依赖固定任务向量，因此每步为全部注册任务各查询一次 Gate，
由 Top-K 前七维 Softmax 概率计算互信息。任务先验均匀，两层取平均，常开共享 FFN 不参与。
这不执行额外的专家 FFN，也不改动视频采样、回归损失的 sample-mean 或模型前向。
Batch 中允许任务频数不均、任务缺席及尾批；MI 与这些频数无关。
MI 会更新缺席任务的私有 Gate；其 FiLM、α、评分 Head 仍只在任务出现在 Batch 时更新。
该计算方式专用于当前的任务条件 Gate；以后改为视频内容条件路由时，需要按任务聚合真实视频路由概率。

在已有训练依赖、视频和预训练资源的环境中，使用新输出目录运行：

```bash
python -B train.py --gpu 0 \
  --set training.moe_mi_weight=0.001 \
  --set training.output_dir=outputs/multitask/runs/private_task_mi_seed42
```

每步日志分别记录 `regression_loss`、`mutual_information`（正值）、`mi_loss`（负互信息）、
`weighted_mi_loss`、`total_loss` 和 `mi_by_layer`。
每轮详情写入 `losses/epoch_XX.json`，汇总同时写入 `history.json` 的 `losses`；
为兼容原统计，`train_loss` 继续表示回归损失的样本均值，总目标见 `losses.total_loss`。
总目标可能为负，这是减去互信息后的正常结果。
检查点保存并核对 MI 权重与计算约定，加载时应使用对应训练配置。
仅训练冻结评分 Head 的两类流程不支持非零 MI 权重，启用会明确报错。
原始 `package_manifest.json` 未重写：原始包文件校验及历史检查点的严格源码校验会检测到此次变更，
应在原始副本中验证历史包。新训练检查点记录当前源码哈希。

相关 CPU 测试无需视频或预训练权重：

```bash
python -B -m unittest tests.test_moe_mi tests.test_training -v
```

## 运行

本机已验证的环境为 `/home/u202511081000084/miniconda3/envs/uve-copy/bin/python`。
`scripts/*.sh` 使用该环境并自动切换至本目录。迁移机器时使用 `requirements.txt` 和 `requirements-tested.txt` 对照安装依赖，
后者记录已测试版本，不是可在所有CUDA平台直接安装的完整环境锁；安装适合本机驱动的 PyTorch / torchvision。

```bash
# 检查全部文件哈希，CPU构造完整模型，加载best/last/epoch9并核对全部参数及四任务前向。
/home/u202511081000084/miniconda3/envs/uve-copy/bin/python -B scripts/verify_package.py

# 默认加载本版本best；--task支持四个任务。
bash scripts/infer.sh --gpu 0 --video /path/to/video.mp4 --prompt "original prompt" --task text_video

# 在原115视频测试集评估best，新输出目录必须不存在。
bash scripts/evaluate.sh --gpu 0 --output outputs/evaluation/private_task_best

# 评估last。
bash scripts/evaluate.sh --gpu 0 --checkpoint outputs/multitask/runs/film_video_only_private_task_modules_no_clip_seed42/last.pt --output outputs/evaluation/private_task_last

# 仅在需要重新训练时显式执行；不继承已训练权重，输出必须使用新的目录。
bash scripts/train.sh --gpu 0 --set training.output_dir=outputs/multitask/runs/new_private_task_no_clip

# 相关CPU单测。
/home/u202511081000084/miniconda3/envs/uve-copy/bin/python -B -m unittest tests.test_private_task_modules tests.test_training -v
```

视频数据与可重建的Prior缓存未复制。本机仍读取原视频目录；换机器时用环境变量 `AIDER_VIDEO_ROOT=/path/to/MOS_Videos` 指定视频根目录。
新生成的Prior缓存写入本目录的 `outputs/cache/`，不会写入原AIDER缓存。
单视频推理只需 `--video` 指向的视频，不需要整套训练数据。

## 校验与兼容范围

三份原始检查点不修改、不删优化器状态。只有 `package_manifest.json` 中SHA256精确匹配的三份历史权重获得路径迁移许可；
同时核验原数据／预训练资源内容哈希和打包代码哈希。其他不一致检查点继续拒绝。
本目录后续新训练产生的checkpoint使用当前包的正常来源校验。
历史报告中的绝对路径作为来源证据保留；运行时使用的是本目录中的模型资源。

这是单seed实验，测试集参与best选择；任务私有化与移除梯度裁剪同时改变，不能单独归因。
本副本不修改原Baseline，也不影响原目录正在运行的实验。
