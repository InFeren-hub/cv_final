# 基于 LeRobot ACT 的 CALVIN 跨环境泛化实验

本仓库为计算机视觉作业 Task 2 的代码与报告材料，内容包括：在 CALVIN 数据集上使用 LeRobot 框架中的 ACT（Action Chunking Transformer）算法训练视觉-动作策略，并在未见过的环境 D 上进行 zero-shot 动作误差评估。

## 实验结果

两组模型使用相同 ACT 网络结构和超参数，仅训练数据不同。

| 模型 | 训练数据 | 最终训练 Loss | 最终训练 Action L1 | 最终验证 Action L1 |
|---|---|---:|---:|---:|
| B-only | CALVIN 环境 B | 0.202813 | 0.199942 | 0.251965 |
| A+B+C joint | CALVIN 环境 A、B、C | 0.235493 | 0.239299 | 0.255756 |

在未参与训练的环境 D 上进行 zero-shot 动作误差评估：

| 模型 | Chunk Action L1 | First-action L1 |
|---|---:|---:|
| B-only | 0.215355 | 0.154377 |
| A+B+C joint | 0.172802 | 0.126927 |

A+B+C joint 模型相比 B-only 模型在环境 D 上将 chunk-level Action L1 降低了 **19.76%**。

## WandB 链接

- 项目主页：<https://wandb.ai/escan0r-fudan/calvin-act-val-curves>
- B-only run：<https://wandb.ai/escan0r-fudan/calvin-act-val-curves/runs/87h3dw7h>
- A+B+C joint run：<https://wandb.ai/escan0r-fudan/calvin-act-val-curves/runs/kzwrndpi>

## 仓库结构

```text
.
├── README.md
└── task2/
    ├── environment.yml
    ├── scripts/
    │   ├── prepare_calvin_data.sh
    │   ├── train_act_b_only.sh
    │   ├── train_act_joint_abc.sh
    │   ├── evaluate_zero_shot_d.sh
    │   └── export_result_plots.sh
    ├── src/
    │   ├── train_act_calvin.py
    │   ├── evaluate_zero_shot_d.py
    │   └── plot_results.py
    ├── report/
    │   ├── task2_report.tex
    │   └── experiment_summary.json
    └── figures/
```

## 环境配置

使用 conda 创建环境：

```bash
cd task2
conda env create -f environment.yml
conda activate calvin-act
```

## 数据准备

下载 LeRobot 格式的 CALVIN 数据集：

```bash
cd task2
bash scripts/prepare_calvin_data.sh /root/autodl-tmp/cv_hw3_task2/data/calvin-lerobot
```

期望的数据目录结构如下：

```text
/root/autodl-tmp/cv_hw3_task2/data/calvin-lerobot/
├── splitA/
├── splitB/
├── splitC/
└── splitD/
```

本实验只需要一份数据集。训练阶段使用 splitB 训练基础模型，使用 splitA、splitB、splitC 训练联合模型，测试阶段使用 splitD 做 zero-shot 评估。

## 模型训练

训练 B-only 基础策略：

```bash
cd task2
bash scripts/train_act_b_only.sh
```

训练 A+B+C 多环境联合策略：

```bash
cd task2
bash scripts/train_act_joint_abc.sh
```

两组训练使用相同主要超参数：

- Batch size：64
- Training steps：12,000
- Learning rate：1e-5
- Optimizer：AdamW
- Action chunk size：64
- Loss：Action L1 + KL regularization
- Validation interval：每 1,000 steps 验证一次

## 环境 D Zero-shot 测试

在未见过的环境 D 上评估两个 checkpoint：

```bash
cd task2
bash scripts/evaluate_zero_shot_d.sh
```

该脚本计算离线动作预测误差：

- Chunk-level Action L1
- First-action L1

本实验采用环境 D 专家轨迹上的离线动作误差作为 zero-shot 泛化指标。

## 图表导出

导出训练、验证和环境 D zero-shot 对比图：

```bash
cd task2
bash scripts/export_result_plots.sh
```

主要输出文件：

```text
outputs/validation_curve_comparison/
├── fig_train_total_loss.png
├── fig_train_action_l1_loss.png
├── fig_validation_total_loss.png
├── fig_validation_action_l1_loss.png
└── validation_curve_summary.json

outputs/zero_shot_d/
├── fig_zero_shot_d_action_error.png
└── zero_shot_d_summary.json
```

## 报告图表

报告中使用的图表已放在 `task2/figures/` 目录：

| 图表 | 文件 |
|---|---|
| 训练 total loss | `task2/figures/fig_train_total_loss.png` |
| 训练 Action L1 | `task2/figures/fig_train_action_l1_loss.png` |
| 验证 total loss | `task2/figures/fig_validation_total_loss.png` |
| 验证 Action L1 | `task2/figures/fig_validation_action_l1_loss.png` |
| 环境 D zero-shot 动作误差 | `task2/figures/fig_zero_shot_d_action_error.png` |

## 模型权重

报告中使用的最终 checkpoint：

```text
outputs/act_b_only_bs64_valcurve_20260616_174028/checkpoints/012000/pretrained_model
outputs/act_joint_abc_bs64_valcurve_20260616_180359/checkpoints/012000/pretrained_model
```

模型权重下载链接：<https://drive.google.com/file/d/14H8HYwJ4SKynrkCx36ngJ8cSX7DytoZK/view?usp=sharing>
