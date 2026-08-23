# 模型大横评报告（Grand Eval）

> 日期：2026-08-23 ｜ 状态：**进行中**（数据随实验完成逐格填入，`⬜` 为待填）
> 目的：对手上全部转写前端 × 分离配置做统一口径横评，为论文实验矩阵提供主表。

## 1. 参评模型

| 类别 | 模型 | 说明 |
|---|---|---|
| 转写·全乐器 | **ia-amt default** | anime-song instrument-agnostic-amt，Transkun Semi-CRF，全乐器单模型 |
| 转写·吉他专用 | **ia-amt guitar_v1_5** | 同上，吉他专调检查点（2026-07-22 版） |
| 转写·全乐器(token) | **YourMT3+** | T5 token 解码，cross-dataset 配方（HF Space 默认检查点） |
| 转写·钢琴 | **ByteDance (Kong 2021)** | 帧回归 onset/offset/velocity，钢琴专训 |
| 转写·吉他(Riley) | **riley_fl / riley_gaps** | QMUL，ByteDance HR 域适应（fl=失真补、gaps=GAPS 古典） |
| 转写·轻量基线 | **basic-pitch** | Spotify，通用轻量模型，阈值可调 |
| 分离 | **demucs htdemucs_6s** | 6 件套分离（drums/bass/other/vocals/piano/guitar） |
| 分离 | **VER-SEP 1.1** | 本项目微调 mel-band-roformer 吉他/其他二分（SDR 5.75） |

## 2. 指标定义与计算公式

所有事件级指标基于**贪心一对一 onset 匹配**（仓库 `eval/guitar_zeroshot_gpif.match_events`）：
参考音与估计音按音高分桶（严格音高指标）或不分桶（any-pitch 指标），
同桶内按 onset 时间贪心配对，配对条件 `|t_est − t_ref| ≤ tol`，每个估计音至多配一次。
全部指标为 **micro 汇总**（跨曲累计 TP/n_ref/n_est 再计算）。

| 指标 | 公式 | 说明 |
|---|---|---|
| 精度 P | P = TP / n_est | 估计音中正确的比例（错音率） |
| 召回 R | R = TP / n_ref | 参考音中被抓住的比例（漏音率） |
| note F1@τ | F1 = 2PR/(P+R)，τ∈{25,50}ms，严格音高 | 学界标准口径 |
| onset F1@τ | 同上，any-pitch（忽略音高） | 音头检测能力 |
| **AUPRC** | 对 (R_k, P_k) 工作点序列按 recall 排序做阶梯积分：AUC = Σ (R_{k+1}−R_k)·max_{j≤k} P_j | **主指标**。扫模型阈值旋钮（ia-amt: note-bias 0→8；basic-pitch: onset-threshold 0.3→0.9）取整条 PR 曲线；单点 P/R 依赖各模型默认阈值，不可比 |
| **P@R=0.35** | 曲线上 recall≥0.35 的点中精度最大值 | 记谱场景召回下限锚点 |
| **R@P=0.80** | 曲线上 precision≥0.80 的点中召回最大值 | 记谱场景精度下限锚点 |
| offset F1 | 在 onset@50ms 匹配成功的音对中，`|off_est − off_ref| ≤ max(50ms, 0.2·dur_ref)` 记 TP | 时值/收尾质量（记谱核心） |
| 数量校准比 | n_est / n_ref | 输出量是否健康（过碎→>1.2，过省→<0.8） |
| 时值中位比 | median(dur_est) / median(dur_ref) | 谱面可读性代理（碎/拖） |
| 过分割率 | 每个被命中的参考音对应的估计音段数均值 | ⬜ 汇总阶段后验计算 |

> 待补指标（轴 3）：per-instrument recall（GT 按 MIDI program 分组）。

## 3. 轴 1：吉他

### 3.1 GuitarSet test 60 首（干净独奏上界）

| 模型 | note F1@50 | P / R | onset F1@50 | offset F1 | 数量比 | 时值中位 |
|---|---|---|---|---|---|---|
| riley_fl | **0.9354** | .925/.946 | 0.9456 | 0.666 | 1.02 | 0.241s |
| **YourMT3+** | 0.9221 | .951/.895 | 0.9340 | **0.7266** | 0.94 | 0.230s |
| guitar_v1_5 | 0.9228 | .937/.909 | 0.9324 | 0.670 | 0.97 | 0.212s |
| ia-amt default | 0.9212 | .942/.902 | 0.9262 | 0.522 | 0.96 | 0.234s |
| riley_gaps | 0.9104 | .930/.892 | 0.9291 | 0.309 | 0.96 | 0.329s |
| basic-pitch | 0.7982 | .759/.842 | 0.8193 | 0.436 | 1.11 | 0.245s |

（GT 时值中位 0.227s，n_ref=8715）

### 3.2 F2 测试集 112 对（受控混音）——分离 × 前端矩阵（note F1@50）

| 前端 \ 输入 | raw 混音 | VER-SEP stem | demucs stem |
|---|---|---|---|
| **riley_fl** | **0.449** (P.45/R.45) | **0.587** (P.81/R.46) | 0.540 (P.76/R.42) |
| guitar_v1_5 | 0.392 (P.42/R.37) | 0.509 (P.87/R.36) | 0.454 (P.84/R.31) |
| basic-pitch | 0.378 (P.37/R.38) | 0.507 (P.64/R.42) | 0.466 (P.63/R.37) |
| ia-amt default | ⬜ | ⬜ | ⬜ |
| YourMT3+ | ⬜ | ⬜ | ⬜ |

### 3.3 PR 曲线（30 首子集，主指标）

| 模型 × 列 | AUPRC | P@R=0.35 | R@P=0.80 |
|---|---|---|---|
| guitar_v1_5 raw | 0.255 | 0.453 | 0 |
| guitar_v1_5 VER-SEP | 0.482 | 0.879 | 0.410 |
| ia-amt default raw | 0.250 | 0.422 | 0 |
| **ia-amt default VER-SEP** | **0.525** | **0.905** | **0.459** |
| basic-pitch raw | 0.313 | 0.482 | 0.073 |
| basic-pitch VER-SEP | 0.438 | 0.823 | 0.396 |

## 4. 轴 2：钢琴（F2-piano 40 对：MAESTRO test ×90s × MoisesDB 伴奏）

### note F1@50（ymt3 列补跑中，其余已出）

| 前端 \ 输入 | clean 独奏 | raw 混音 | demucs-piano |
|---|---|---|---|
| ByteDance | **0.969** (.98/.96) | 0.701 (.66/.75) | 0.702 (.63/.79) |
| ia-amt default | 0.964 (.99/.94) | **0.729** (.65/.83) | **0.785** (.79/.78) |
| YourMT3+ | 0.948 (.98/.91) | 0.521 (.39/.80) | 0.603 (.50/.76) |
| basic-pitch | 0.690 | 0.398 | 0.512 |

offset F1（clean）：ia_default 0.373 > bytedance 0.349 > basic 0.113；数量比：bytedance demucs 列 1.25（过火），ia_default demucs 列 0.98（分离把它混音列的 1.29 过火修正回健康）；**YourMT3+ 混音列数量比 2.07（狂开火，P 仅 .39），demucs 修正到 1.52 仍偏火**——token 系在密集钢琴混音上输出失控，与吉他轴上它的 offset 优势形成鲜明对照。

## 5. 轴 3：全乐器（URMP 44 首真实 + BabySlakh 20 首合成）

### note F1@50（⬜ 实验排队中；列为 raw / oracle 分离上界 / demucs stemwise）

| 前端 \ 列 | URMP raw | URMP oracle | URMP demucs | BS raw | BS oracle | BS demucs |
|---|---|---|---|---|---|---|
| ia-amt default | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| YourMT3+ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| basic-pitch | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| ByteDance（钢琴域外） | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| riley_gaps（吉他域外） | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

### per-instrument recall（⬜）

## 6. 结论（随数据滚动更新）

1. **分离的价值必须用曲线表述**：VER-SEP 使 guitar_v1_5 的 AUPRC 0.255→0.482（+88%）、ia-amt default 0.250→0.525（+110%）。单点 F1 只能看到 +11~12pt，严重低估。
2. **专用≠更强（在分离域）**：吉他专用 guitar_v1_5 在干净域小胜全乐器 default（0.923 vs 0.921），但在 VER-SEP stem 上整条 PR 曲线被 default 反超（AUPRC 0.482 vs 0.525）——专用化的收益不可迁移到分离域，域(分离伪影)比特化(乐器)影响更大。
3. **轻量模型在混音上被低估**：basic-pitch 默认阈值单点垫底，但阈值扫描后 raw 列 AUPRC 最高（0.313）；它输在 stem 列（0.438）——分离伪影对轻量模型伤害更大。
4. **offset 是 token 系的护城河**：YourMT3+ offset F1 0.727，大幅领先全部帧回归/Semi-CRF 系（0.31-0.67）；riley_gaps 的 0.309 与其「碎」的听感互证。时值结构指标比 note F1 更能区分模型代际。
5. **分离 stem 域的真王者是 riley_fl（单点）**：riley_fl+VER-SEP note F1 0.587，大幅领先 guitar_v1_5（0.509）/basic（0.507），且其 P.81/R.46 工作点的召回超过 guitar_v1_5 整条 PR 曲线在同等精度下的水平——即此优势并非阈值红利。注意 riley_fl 无阈值旋钮（PR 曲线不可扫，单点标注）；其 offset F1 仅 0.32，时值短板与 F1 长板并存。
6. **钢琴域：专域模型的优势只剩 0.5pt，且分离对它零收益**：干净域 bytedance 0.969 vs 全能 ia_default 0.964；混音与 demucs-piano 列 default 反超（0.729/0.785 vs 0.701/0.702）。**demucs 分离帮 default +5.6pt（0.729→0.785，精度 0.65→0.79、数量比 1.29→0.98 归位），对 bytedance 却 +0.03pt 持平**——钢琴域转写器对分离伪影的脆弱性与吉他轴的 demucs 表现（伤害召回）同构：**分离收益是「前端 × 伪影谱」的交互项，不是通用红利**。bytedance 的 offset F1 在自己主场（0.349）反被 default（0.373）超过。
7. ⬜（轴 3 结论待数据）

## 附：环境与复现

- 数据：GuitarSet test 60 / F2 test 112（GuitarSet×MoisesDB，−8..+6dB）/ F2-piano 40（MAESTRO×MoisesDB）/ URMP 44（YourMT3 Zenodo 镜像 8021437）/ BabySlakh 20（Zenodo 4603870）
- 脚本：`eval/grand_eval.py`（聚合）、`eval/pr_sweep.py`（PR 扫描）、`eval/run_axis2.sh`、`eval/run_axis3.sh`、`train/gen_f2_piano.py`
- 全部结果 JSON：`output/grand_eval/*.json`
