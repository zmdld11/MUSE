# 方案 A 设计：后处理二值化阈值 + 调性过滤（提音高 recall）

> 日期：2026-08-01 | 状态：✅ A/B 完成，结论=维持当前默认，进入方案 B 决策
> 前置：`project_log.md` 2026-08-01 诊断复核（实测确认两大瓶颈）

---

## 一、目标与成功标准

| 项 | 值 |
|---|---|
| 目标指标 | 全量 40 首 note_f1 从 0.217 显著提升 |
| 预估区间 | 0.45-0.6（以全量实测为准） |
| 验收条件 | 最优配置 note_f1 > 0.45 且 note_precision ≥ 0.5 |
| 附加要求 | 逐首对比，退化曲目记录在案；不改模型、不动训练 |
| 评测条件 | GiantMIDI 40 首, seed=42（与 07-31 基线完全同条件） |

## 二、改动清单

### 2.1 二值化阈值策略（`src/frame_post.py`）

- `process_frames` 新增参数 `binarize_threshold`：`None` = 现有自适应 percentile 模式；浮点数 = 固定阈值模式（直接作用于 HMM 平滑后验）
- `_adaptive_threshold_per_register` 已有 `fixed_threshold` 参数，复用，不改其实现
- A/B 变体：adaptive / fixed 0.3 / fixed 0.5

### 2.2 调性过滤开关（`src/note_post.py`）

- `refine_notes` 新增参数 `key_filter: bool = True`（默认保持现状，避免行为突变）
- `False` = 跳过 top-7 音级过滤（本轮 A/B 验证其影响）
- 若后续需要在乐谱级保留可读性：改用 `key_estimate.py` 的 K-S 调号检测 + 允许临时记号，另行设计

### 2.3 评测入口（`eval/eval.py` + `eval/metrics.py`）

- `eval.py` 新增 CLI：`--threshold {adaptive,0.3,0.5}`、`--key-filter {on,off}`，透传到后处理与帧级指标
- `metrics.py` 帧级二值化与后处理保持一致（`_binarize_frame_probs` 接受阈值参数）
- `eval.py` 启动时设置 `NUMBA_CACHE_DIR` / `TMP` 到项目内可写目录，修复 numba 缓存导致的挂起

### 2.4 附带小修

- `eval/diagnose.py` 第 3 层：est 音符零时长导致 mir_eval 崩溃（补 `offset-onset ≥ 1e-4` 过滤）

## 三、A/B 实验设计

| 变体 | 阈值 | 调性过滤 |
|---|---|---|
| V0（基线对照） | adaptive | on |
| V1 | adaptive | off |
| V2 | fixed 0.5 | on |
| V3 | fixed 0.5 | off |
| V4（复测 git 争议） | fixed 0.3 | off |

- 全量 40 首，seed=42，逐首输出 + 聚合 note_f1/P/R
- 结果写入 `eval/reports/`，并与 `report_20260731-194519.json` 基线对比

## 四、验收与回退

- 所有改动保持参数可切换，旧路径保留
- 若全量最优 < 0.35：保留实测最佳组合，写明原因，再决定是否进入方案 B
- 定案后：把最优配置固化为默认值，更新 `project_log.md` / `version_plan.md`
- 临时诊断脚本（`eval/reports/_*_verify.py`）实施完成后清理或归档

## 五、A/B 结果（2026-08-01 晚，全量 40 首）

8 个变体全部跑完（详见 `project_log.md` 2026-08-01 方案 A 节）：

- **当前默认（adaptive + 调性过滤）= 0.294，为全部变体最优**；固定阈值 / 关调性过滤 / capped 自适应均不敌
- 日志旧基线 0.217 系 fixed0.3 代码状态的误标数字；真实基线 0.294
- 验收未达（目标 >0.45），按回退条款：保留当前默认，参数开关保留供后续实验
- 副产品：fixed 0.5-0.7 使 offset_f1 翻倍（0.112→0.196-0.200），记入时值里程碑
- **下一步：方案 B**（每音高 Otsu/双峰阈值、连通域分音高、K-S 调号+临时记号的调性过滤）
