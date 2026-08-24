# 模型大横评报告（Grand Eval）

> 日期：2026-08-23（轴 3 收官 08-24）｜ 状态：**全部完成**（轴 1/2/3 + F2 矩阵全格，无待填项）
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

### 测试数据集

| 数据集 | 规模 | 内容与来源 | GT 形式 | 用于 |
|---|---|---|---|---|
| **GuitarSet test** | 60 首 | 真实吉他独奏录音（player-split test，10 位乐手） | 对齐 MIDI | 轴 1 干净独奏上界 |
| **F2 test** | 112 对 | GuitarSet 吉他 × MoisesDB Other 伴奏受控混音（−8..+6dB，本地合成，`train/gen_f2_mixes.py`） | 混音前吉他 MIDI（精确对齐） | 轴 1 受控混音矩阵（raw / VER-SEP / demucs 三列） |
| **F2-piano** | 40 对 | MAESTRO test 钢琴（前 90s）× MoisesDB Other 伴奏受控混音（`train/gen_f2_piano.py`） | MAESTRO MIDI | 轴 2（clean / raw / demucs-piano 三列） |
| **URMP** | 44 首 | 真实室内乐录音，2–5 件管弦乐器（YourMT3 Zenodo 镜像 8021437，16k） | Sco 谱面 MIDI（**已做逐曲仿射+平滑对齐修正**，见第 5 节） | 轴 3 真实录音压力测试（raw / oracle / demucs） |
| **BabySlakh** | 20 首 | 合成多轨流行编曲（Zenodo 4603870） | 渲染源 MIDI `all_src.mid`（严格对齐） | 轴 3 定量基准（raw / oracle / demucs） |
| 東の空 / 虚無の先 | 2 首 | 真实发行歌曲（失真/清音电吉他） | 人工谱面（GP8 / mscz） | 轴 1 真实曲固定对齐参考 |
| 地球最後の告白 | 1 首 | 真实发行歌曲 | 无可用谱面 | 仅听感评价，不入指标 |

> 输入形态术语：**原声独奏** = 无伴奏原始录音；**原声混音** = 含伴奏的原始混音；**分离音轨（stem）** = 混音经 VER-SEP / demucs 分离出的目标乐器轨；**oracle 分轨** = 数据集自带的干净分轨（URMP AuSep / BabySlakh 渲染 stems），代表分离上界。

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
| 过分割率 | 每个被命中的参考音，其时间窗 [onset−25ms, offset+25ms] 内同音高估计音的个数均值 | 碎裂度（>1 = 一音多段）。GuitarSet 实测：basic 1.249 > riley_fl 1.215 > ymt3 1.182 > riley_gaps 1.166 > guitar_v1_5 1.147 > ia_default 1.097 |

> 轴 3 per-instrument recall 已补（见第 5 节表）。

## 3. 轴 1：吉他

### 3.1 GuitarSet test 60 首（干净独奏上界）

> 数据集：**GuitarSet test 60**（真实吉他录音）｜输入形态：**原声独奏**（无伴奏，直接转写，无分离介入）

| 模型 | note F1@50 | P / R | onset F1@50 | offset F1 | 数量比 | 时值中位 |
|---|---|---|---|---|---|---|
| riley_fl | **0.9354** | .925/.946 | 0.9456 | 0.666 | 1.02 | 0.241s |
| **YourMT3+** | 0.9221 | .951/.895 | 0.9340 | **0.7266** | 0.94 | 0.230s |
| guitar_v1_5 | 0.9228 | .937/.909 | 0.9324 | 0.670 | 0.97 | 0.212s |
| ia-amt default | 0.9212 | .942/.902 | 0.9262 | 0.522 | 0.96 | 0.234s |
| riley_gaps | 0.9104 | .930/.892 | 0.9291 | 0.309 | 0.96 | 0.329s |
| basic-pitch | 0.7982 | .759/.842 | 0.8193 | 0.436 | 1.11 | 0.245s |

（GT 时值中位 0.227s，n_ref=8715；过分割率与估计音利用率：basic 1.249/0.81、riley_fl 1.215/0.93、ymt3 1.182/0.95、riley_gaps 1.166/0.94、guitar_v1_5 1.147/0.92、ia_default 1.097/0.94——riley_fl 的高过分割与其时值碎的听感互证；basic 有 ~19% 估计音完全匹配不上任何参考音）

### 3.2 F2 测试集 112 对（受控混音）——分离 × 前端矩阵（note F1@50）

> 数据集：**F2 test 112**（GuitarSet 吉他 × MoisesDB 伴奏受控混音）｜输入形态：raw 列 = **原声混音**直接转写；VER-SEP 列 = 混音经 **VER-SEP 1.1 分离出吉他 stem** 后转写；demucs 列 = 混音经 **demucs 分离出吉他 stem** 后转写（均为分离音轨独奏）

| 前端 \ 输入 | raw 混音 | VER-SEP stem | demucs stem |
|---|---|---|---|
| **riley_fl** | **0.449** (P.45/R.45) | **0.587** (P.81/R.46) | **0.540** (P.76/R.42) |
| guitar_v1_5 | 0.392 (P.42/R.37) | 0.509 (P.87/R.36) | 0.454 (P.84/R.31) |
| basic-pitch | 0.378 (P.37/R.38) | 0.507 (P.64/R.42) | 0.466 (P.63/R.37) |
| ia-amt default | 0.405 (P.40/R.41) | 0.531 (P.91/R.38) | 0.490 (P.88/R.34) |
| YourMT3+ | 0.318 (P.23/R.50) | 0.528 (P.73/R.41) | 0.452 (P.55/R.38) |

（08-24 补：ia default / ymt3 两行全 112 对跑齐，`output/grand_eval/f2_fill/metrics.json`。要点：①单点全量确认 PR 曲线的反转——default 在 VER-SEP 列 0.531 > guitar_v1_5 0.509；②default 的 VER-SEP 精度高达 .91 但数量比仅 0.41（保守漏报），与结论 2「AUPRC 0.525 反超」互补：它站在高精度工作点；③ymt3 raw 混音狂开火（数量比 2.13、P 仅 .23），分离把它治回健康（.73），+21pt 是全场最大分离增益——token 系对混音最敏感、对 stem 最受益；④riley_fl 三列全冠依旧。）

### 3.3 PR 曲线（30 首子集，主指标）

> 数据集：F2 test 的前 30 首子集｜输入形态：{**原声混音**（raw），**VER-SEP 分离音轨**（versep）}×{ia-amt（note-bias 0→8 扫描），basic-pitch（onset-threshold 0.3→0.9 扫描）}；Riley/ByteDance/YourMT3+ 无阈值旋钮，不参与扫描（单点见各表）

| 模型 × 列 | AUPRC | P@R=0.35 | R@P=0.80 |
|---|---|---|---|
| guitar_v1_5 raw | 0.255 | 0.453 | 0 |
| guitar_v1_5 VER-SEP | 0.482 | 0.879 | 0.410 |
| ia-amt default raw | 0.250 | 0.422 | 0 |
| **ia-amt default VER-SEP** | **0.525** | **0.905** | **0.459** |
| basic-pitch raw | 0.313 | 0.482 | 0.073 |
| basic-pitch VER-SEP | 0.438 | 0.823 | 0.396 |

## 4. 轴 2：钢琴（F2-piano 40 对：MAESTRO test ×90s × MoisesDB 伴奏）

> 数据集：**F2-piano 40**（MAESTRO test 钢琴前 90s × MoisesDB Other 伴奏受控混音）｜输入形态：clean 列 = **钢琴原声独奏**；raw 列 = **原声混音**直接转写；demucs-piano 列 = 混音经 **demucs 分离出钢琴 stem** 后转写（分离音轨独奏）。VER-SEP 为吉他/其他二分无钢琴轨，不参加本轴。

### note F1@50（全部完成）

| 前端 \ 输入 | clean 独奏 | raw 混音 | demucs-piano |
|---|---|---|---|
| ByteDance | **0.969** (.98/.96) | 0.701 (.66/.75) | 0.702 (.63/.79) |
| ia-amt default | 0.964 (.99/.94) | **0.729** (.65/.83) | **0.785** (.79/.78) |
| YourMT3+ | 0.948 (.98/.91) | 0.521 (.39/.80) | 0.603 (.50/.76) |
| basic-pitch | 0.690 | 0.398 | 0.512 |

offset F1（clean）：ia_default 0.373 > bytedance 0.349 > basic 0.113；数量比：bytedance demucs 列 1.25（过火），ia_default demucs 列 0.98（分离把它混音列的 1.29 过火修正回健康）；**YourMT3+ 混音列数量比 2.07（狂开火，P 仅 .39），demucs 修正到 1.52 仍偏火**——token 系在密集钢琴混音上输出失控，与吉他轴上它的 offset 优势形成鲜明对照。

## 5. 轴 3：全乐器（URMP 44 首真实 + BabySlakh 20 首合成）

> 数据集：**URMP 44**（真实室内乐录音）+ **BabySlakh 20**（合成多轨编曲）｜输入形态：raw 列 = **原声/合成混音**直接整体转写；oracle 列 = 数据集**自带干净分轨**逐轨转写后合并（分离上界）；demucs 列 = 混音经 **demucs 6 件套分离**逐轨转写后合并（stemwise 工作流）

### ⚠️ URMP GT 对齐修正（读表前必读）

URMP 的 Sco_*.mid 是**谱面原速 MIDI，与 AuMix 演录音 tempo 不一致且含逐曲 rubato**（乐手各自独立录音、非严格合拍）。直接当 GT 用时 onset 匹配全线崩溃（所有前端 onset_any@50 仅 0.16–0.29，连纯吉他模型 riley 都与全能模型同分——评测失效的明确信号）。诊断依据：GT 与预测的音高序列逐音一致、时间轴呈仿射失配（如 01_Jupiter 演奏慢 ~1.47×＋3s 前导）。

修正方案（`eval/urmp_align.py`，对所有前端统一使用同一映射，保证公平）：
1. 逐曲粗网格＋细化估计仿射 (scale, shift)——scale 实测散布 0.54–1.52，证明确属逐曲 rubato 而非系统变速；
2. 迭代平滑残差：以 ia_raw 预测为音频时间轴代理，**严格音高锚点**（同音高 onset 配对）＋高斯窗（σ=3s）平滑残差修正映射，5 轮迭代；
3. **越代理交叉验证**：用 ymt3_raw（未参与对齐的模型）检验映射迁移性——ia 匹配中位 0.499、ymt3 中位 0.461，同一映射跨模型成立，非对 ia 过拟合；
4. 残余错位天花板：对齐后 onset 匹配中位 ~0.50 后不再上升（σ 扫描 6→3 仅 +0.5pt），系乐手演奏本身与谱面有音级出入＋代理模型召回封顶，属数据集固有属性。

**结论：URMP 列的绝对值被系统性低估，只用于横向相对比较（前端×列在同一映射下公平）；全乐器轴的定量结论以 BabySlakh（合成、GT 严格对齐）为准，URMP 作为真实录音压力测试。** 另：Sco 原始 MIDI program 大量为默认值（1=钢琴 占 39%、48=弦乐组冒充大提琴），per-instrument 表按曲名后缀（vn/vc/fl/tpt…）重映射 GM program。

### note F1@50（列为 raw / oracle 分离上界 / demucs stemwise；URMP 为对齐修正后数值）

| 前端 \ 列 | URMP raw | URMP oracle | URMP demucs | BS raw | BS oracle | BS demucs |
|---|---|---|---|---|---|---|
| ia-amt default | **0.343** | 0.327 | 0.313 | 0.485 | 0.628 | 0.464 |
| YourMT3+ | 0.316 | **0.330** | 0.299 | **0.793** | **0.825** | 0.474 |
| basic-pitch | 0.237 | 0.275 | 0.189 | 0.296 | 0.458 | 0.319 |
| ByteDance（钢琴域外） | 0.171 | 0.140 | 0.170 | 0.337 | 0.474 | **0.365** |
| riley_gaps（吉他域外） | 0.249 | 0.262 | 0.241 | 0.332 | 0.451 | 0.331 |

P/R 与数量比（note@50 工作点，格式 P/R[count]）：

| 前端 | URMP raw | BS raw | BS oracle | BS demucs |
|---|---|---|---|---|
| ia-amt | .36/.33[0.90] | **.81/.35[0.43]** | .81/.51[0.64] | .53/.42[0.79] |
| YourMT3+ | .33/.30[0.92] | .84/.75[0.89] | .86/.79[0.93] | **.39/.61[1.56]** |
| basic-pitch | .23/.24[1.03] | .42/.23[0.54] | .47/.45[0.95] | .32/.32[1.01] |
| ByteDance | .23/.14[0.61] | .54/.24[0.45] | .50/.45[0.90] | .37/.36[0.98] |
| riley_gaps | .26/.24[0.91] | .50/.25[0.50] | .48/.42[0.88] | .35/.32[0.92] |

offset F1（BS raw / oracle / demucs）：ymt3 .51/.58/.27 一骑绝尘；ia .27/.38/.23；bd .13/.28/.16；basic/riley ≤.16。token 系的 offset 护城河延伸到全乐器轴。

### per-instrument recall（URMP 对齐后，GT program 按曲名重映射；raw 列）

| 乐器 | n_ref | ia | ymt3 | basic | riley |
|---|---|---|---|---|---|
| violin | 7667 | **.329** | .284 | .204 | .225 |
| viola | 2964 | **.289** | .235 | .197 | .231 |
| cello | 2344 | .291 | .251 | **.231** | .222 |
| bass | 406 | **.217** | .167 | .108 | .126 |
| trumpet | 4030 | .338 | **.355** | .284 | .290 |
| trombone | 1529 | .295 | **.343** | .266 | .296 |
| tuba | 865 | **.558** | .533 | .471 | .511 |
| horn | 840 | **.218** | .186 | .150 | .163 |
| alto_sax | 1917 | **.483** | .476 | .418 | .433 |
| oboe | 857 | **.217** | .189 | .197 | .161 |
| bassoon | 245 | **.102** | .094 | .069 | .033 |
| clarinet | 1857 | **.438** | .414 | .344 | .358 |
| flute | 4089 | **.362** | .341 | .256 | .139 |

## 6. 结论（随数据滚动更新）

1. **分离的价值必须用曲线表述**：VER-SEP 使 guitar_v1_5 的 AUPRC 0.255→0.482（+88%）、ia-amt default 0.250→0.525（+110%）。单点 F1 只能看到 +11~12pt，严重低估。
2. **专用≠更强（在分离域）**：吉他专用 guitar_v1_5 在干净域小胜全乐器 default（0.923 vs 0.921），但在 VER-SEP stem 上整条 PR 曲线被 default 反超（AUPRC 0.482 vs 0.525）——专用化的收益不可迁移到分离域，域(分离伪影)比特化(乐器)影响更大。
3. **轻量模型在混音上被低估**：basic-pitch 默认阈值单点垫底，但阈值扫描后 raw 列 AUPRC 最高（0.313）；它输在 stem 列（0.438）——分离伪影对轻量模型伤害更大。
4. **offset 是 token 系的护城河**：YourMT3+ offset F1 0.727，大幅领先全部帧回归/Semi-CRF 系（0.31-0.67）；riley_gaps 的 0.309 与其「碎」的听感互证。时值结构指标比 note F1 更能区分模型代际。
5. **分离 stem 域的真王者是 riley_fl（单点）**：riley_fl+VER-SEP note F1 0.587，大幅领先 guitar_v1_5（0.509）/basic（0.507），且其 P.81/R.46 工作点的召回超过 guitar_v1_5 整条 PR 曲线在同等精度下的水平——即此优势并非阈值红利。注意 riley_fl 无阈值旋钮（PR 曲线不可扫，单点标注）；其 offset F1 仅 0.32，时值短板与 F1 长板并存。
6. **钢琴域：专域模型的优势只剩 0.5pt，且分离对它零收益**：干净域 bytedance 0.969 vs 全能 ia_default 0.964；混音与 demucs-piano 列 default 反超（0.729/0.785 vs 0.701/0.702）。**demucs 分离帮 default +5.6pt（0.729→0.785，精度 0.65→0.79、数量比 1.29→0.98 归位），对 bytedance 却 +0.03pt 持平**——钢琴域转写器对分离伪影的脆弱性与吉他轴的 demucs 表现（伤害召回）同构：**分离收益是「前端 × 伪影谱」的交互项，不是通用红利**。bytedance 的 offset F1 在自己主场（0.349）反被 default（0.373）超过。
7. **全乐器轴（轴 3）：数据形态决定分离价值，stemwise 合并对 token 系是陷阱**
   - **合成密集多轨（BabySlakh，定量基准）**：YourMT3+ 断档第一（raw 0.793，oracle 0.825）——Slakh 式合成多轨是其训练谱系内的主场。ia-amt raw 只有 0.485，且形态是「高精低召」（P .81 / R .35，数量比 0.43）：全乐器模型在多乐器密集混音上系统性漏报。
   - **oracle 分离在密集混音上人人有份**：ia +14.3pt（0.485→0.628，数量比 0.43→0.64 回补召回）、basic +16.2pt、bd +13.7pt、riley +11.9pt、ymt3 +3.2pt。干净的逐乐器 stem 是密集混音转写的正确前置。
   - **demucs stemwise 只救弱模型、毁强模型**：basic/bd +2~3pt，ia −2pt（精度 .81→.53），ymt3 **−32pt**（0.793→0.474，精度 .84→.39，数量比 1.56）——每个 stem 都被全乐器前端转写一遍，同类音符在多 stem 重复出现，合并后假阳性翻倍。**stemwise 工作流必须配 per-stem 乐器门控（按 stem 类型过滤输出音色），否则对全乐器/token 前端是负资产**。
   - **真实录音（URMP）上分离无增益**：ia raw 0.343 即全场最高，oracle/demucs 对所有前端都持平或下降——小编制真实原声乐器混音并不难转写，分离引入的伪影与合并重复抵消了收益。与吉他轴（分离 AUPRC 翻倍）、BS 轴（oracle +14pt）合并看：**分离的收益 = f(混音密度, 乐器重叠度, 分离质量)，不是普适红利**；真实歌曲（吉他/钢琴主唱伴奏型）介于两者之间且更接近 URMP 形态。
   - **域外基线行为符合设计**：bd（钢琴单域）在 URMP 真实管弦上 0.14–0.17 全场垫底；但在 BS 上反而 0.34–0.47 居中——合成 stem 每轨乐器单一，单域模型也能捡到本域轨道的部分分数。riley 的 flute recall .139 vs ia .362 是最干净的域外信号。
   - **per-instrument**：ia-amt 在弦乐组与 flute 全面领先，ymt3 在铜管略优；tuba（少音符、音色纯）最好恢复 .56，bassoon（软、簧片、低频密集）最难 .10——per-instrument 分解比总分更能暴露音色盲区。

## 附：环境与复现

- 数据：GuitarSet test 60 / F2 test 112（GuitarSet×MoisesDB，−8..+6dB）/ F2-piano 40（MAESTRO×MoisesDB）/ URMP 44（YourMT3 Zenodo 镜像 8021437）/ BabySlakh 20（Zenodo 4603870）
- 脚本：`eval/grand_eval.py`（聚合）、`eval/pr_sweep.py`（PR 扫描）、`eval/urmp_align.py`（URMP GT 仿射+平滑残差对齐）、`eval/run_axis2.sh`、`eval/run_axis3.sh`、`train/gen_f2_piano.py`
- 全部结果 JSON：`output/grand_eval/*.json`（URMP 含对齐前基线 `axis3_urmp_noalign.json` 与对齐后 `axis3_urmp.json`；对齐参数在各 `urmp_align_s*.log` 尾部 JSON）
- 已知坑记录：URMP Sco MIDI 与音频含逐曲 rubato 失配（详见轴 3 对齐说明）；run_axis3.sh 的 BS 聚合曾误指 bd_oracle_urmp 且漏 riley（已修复，08-24）；BabySlakh all_src.mid 在 track 根目录而非 MIDI/ 子目录
