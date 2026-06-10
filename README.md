# TMM-SCC 实验框架

## 环境

```bash
pip install -r requirements.txt
```

## 入口

```bash
python main.py
```

GPU 编号：`utils/config.py` → `GPU`。

## 实验逻辑（学界规范对齐）

| 阶段 | 内容 |
|------|------|
| **白盒攻击** | 在 ALBEF surrogate 上生成 adv 样本（VLR/VE × TMM/SCC/Co-Attack/SGA） |
| **黑盒迁移** | 将 manifest 中的 adv 样本迁移评测受害模型 TCL / BLIP / CLIP |
| **ASR** | `(clean_accuracy - adv_accuracy) / clean_accuracy × 100`，clean 来自 `checkpoints/{task}/clean_metrics.json`（按需计算、缓存） |
| **VE 微调** | VLR 权重 warm-start → SNLI-VE，产出 VE 黑盒 victim 权重 |
| **消融** | 仅 VE 上 TMM vs SCC 的 `num_iters` 曲线（`ablation_200` 子集） |

默认攻击迭代 **`num_iters = 10`**（`AttackConfig`；测试 profile 仍用 2 以加快 smoke）。

## 控制台菜单

| 选项 | 功能 |
|------|------|
| **1–5** | 分步：子集 / 白盒 / 黑盒 / 消融 / 微调 |
| **6** | 本地测试全流程 `mini_100`（含微调+消融，`iters=2`） |
| **7** | 本地正式全流程 `main_1k`（无消融，`iters=10`） |
| **8** | 强制测试（同 6，强制重跑 VE 微调） |
| **9** | **云端·测试全流程** `mini_100` |
| **10** | **云端·消融** `ablation_200`（含 prepare）→ 据此修改正式 `num_iters` |
| **11** | **云端·正式全流程** `main_1k`（清 outputs 除 run.json） |
| **12** | **云端·正式(无微调)** 复用已有 `checkpoints/ve` |
| **q** | 退出 |

### 推荐云端顺序

```
9 测试全流程 → 10 消融 → （改 config 中 attack_num_iters）→ 11 正式全流程
```

- **11 / 12** 启动时 `clean_for_cloud_main()`：删除 `mini_100` / `main_1k` 攻击产物与 `finetune/` 等，**保留 `run.json` 与 `ablation_200` 消融目录**。

## 24G 显存预设（`HARDWARE_PRESETS.server_24g`）

| 参数 | 值 |
|------|-----|
| `batch_size_vlr` | 10 |
| `batch_size_ve` | 16 |
| `sga_batch_size_vlr` | 10 |

云端 profile 已内置上述值；本地 8G 用 `local_8g`。

## 目录

```
main.py          # 入口
utils/config.py  # 全部默认参数与 RunProfile
tmm_scc/         # 攻击与黑盒评测
checkpoints/     # 权重 + clean_metrics.json
outputs/         # 攻击产物 + run.json
data/            # 子集与全量标注
```

CLIP 权重缓存：`checkpoints/vlr/clip/`（离线 HuggingFace）。
