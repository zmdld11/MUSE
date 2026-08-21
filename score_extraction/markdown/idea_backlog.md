# 想法待实现清单 (Idea Backlog)

随手记录偶发想法，有时间一个个试。规则：
- 新想法直接加行，标注 **来源/日期**（如: 诊断 2026-08-02 / 论文AMT-CMT / 用户提议 / 闲聊）
- 优先级: **P0** 当前最痛 / **P1** 有价值 / **P2** 备选
- 状态: ⬜ 待做 → 🟦 进行中 → ✅ 完成 / ❌ 已否决（附原因）
- 试完一个就更新状态，并把结论写进 project_log.md

---

## A. 音符级后处理

| # | 想法 | 来源 | 优先级 | 状态 |
|---|------|------|--------|------|
| A1 | **碎音合并**: 同音高相邻碎音 merge + gap-fill — **❌ 实验失败 (2026-08-02)**：真实录音同音高碎音仅 0.8 个/GT（不是 4.7——那是其他音高音符）；合并反而误合真实快速音符，窗口覆盖 54.4%→45.0%。代码保留在 src/merge_notes.py（工具级） | 诊断 2026-08-02 | P0 | ❌ 失败 |
| A2 | **onset 抢占算法**: 新音开始时截断音高冲突的旧音 — offset 偏长的解法（踏板还原标准做法），放 note_post 输出前 | 用户提议 | P0 | ⬜ |
| A3 | **offset 偏长**: frame 尾段衰减慢（踏板/混响）→ 更鲁棒的 offset 检测（能量衰减阈值动态化？） | 评测诊断 | P1 | ⬜ |
| A4 | **min_note_len 重估**: 当前 5 帧≈116ms，可能漏真实钢琴的快速短音（同音重复、trill） | 评测诊断 | P2 | ⬜ |
| A5 | **高音区覆盖差** (9.0% vs 低音 42%): 泛音干扰 → 高音区专用过滤/阈值 | 诊断 2026-08-02 | P1 | ⬜ |
| A6 | **amplitude 过滤杂音** — ❌ 实验失败：真/杂音 amplitude 区分度不足 (0.58 vs 0.41)，网格过滤 F1 16.1%→16.3% 封顶 | 实验 2026-08-02 | P0 | ❌ 失败 |
| A7 | **basic-pitch 官方模型替代 ours** — ⚠️ 真实录音拾音远强 (窗口覆盖 77% vs 54%, recall 38% vs 21%) 但杂音更多 (3101 音, P=10.4%)，50ms-F1 打平 16%；真实录音可考虑 basic，合成评测仍用 ours | 实验 2026-08-02 | P1 | 🟦 待决策 |
| A8 | **分离钢琴轨转录** — ⚠️ 几乎无效 (窗口 +2pp, F1 持平)；混音串扰不是主因。管线仍保留钢琴轨+wiener (VER2.4 用户实验 F1 0.46→0.63，评估口径不同) | 实验 2026-08-02 | P2 | ❌ 无收益 |

## B. 记谱级

| # | 想法 | 来源 | 优先级 | 状态 |
|---|------|------|--------|------|
| B1 | **beat-depth 节拍加权量化**: 音符按节拍层级加权（强拍=0, 次强拍=1...），量化时强拍优先对齐；Ablation 显示还减少弱拍短音符漏检。BPM 检测已验证可用 | 论文 AMT-CMT | P1 | ⬜ |
| B2 | **节奏量化重新评估**: BP onset 变准后，之前"误差>半格"禁用的 quantize_onsets 可以重测 | 评测诊断 | P1 | ⬜ |
| B3 | **小节完整性检查**: 每小节时值是否等于拍数×拍号，自动报告破损小节 | 四层级梳理 | P1 | ⬜ |
| B4 | **自研三指标**: 记谱级评测指标（当时设想的，未实现） | 设计阶段 | P1 | ⬜ |
| B5 | **重叠消解**: 音符重叠无处理（"音被挤"）→ 与 A2 抢占算法配合 | 听感反馈 | P1 | ⬜ |
| B6 | **等音拼写/连音线补全**: rule-based 符号一致性后处理（论文解码后做的: enharmonic spelling, tie/beam completion） | 论文 AMT-CMT | P2 | ⬜ |

## C. 评测

| # | 想法 | 来源 | 优先级 | 状态 |
|---|------|------|--------|------|
| C1 | **DTW score-audio 对齐**: 替代 compare_real 的固定 --shift，自动处理 rubato/局部速度波动 | 论文 AMT-CMT | P1 | ⬜ |
| C2 | **真实录音数据集扩展**: 目前只有 1 首（夜の向日葵）；URMP 有配对 MusicXML+音频，或从 GiantMIDI 挑同风格 | 论文 AMT-CMT | P1 | ⬜ |
| C3 | **评测基线对照更新**: 论文给了 MT3 等参考值（URMP: Frame 0.885/Onset 0.820/O+O 0.619），可与我们的合成评测对照 | 论文 AMT-CMT | P2 | ⬜ |
| C4 | **compare_real 自动化**: 自动拟合偏移（网格搜索）替代手动 --shift | 诊断 2026-08-02 | P2 | 🟦 部分（--shift 已加，网格搜索未固化） |

## D. 模型 / 训练

| # | 想法 | 来源 | 优先级 | 状态 |
|---|------|------|--------|------|
| D1 | **真实录音域差**: 模型在 FluidSynth 合成音频训练，真实钢琴表现差（碎音+音高不稳）→ 真实录音微调/数据增强（踏板模拟? 混响?） | 诊断 2026-08-02 | P0 | ⬜ |
| D2 | **训练完整性**: VER2.2 只训到 23K steps（参考 O&F 50K），训练不足是否影响真实域泛化 | 训练诊断 | P2 | ⬜ |

## E. 杂项

| # | 想法 | 来源 | 优先级 | 状态 |
|---|------|------|--------|------|
| E1 | 论文 PDF 提取工具已装 PyMuPDF（`_amt_cmt_extracted.txt` 是 AMT-CMT 全文） | 工具 | ✅ | 完成 |
| E2 | 剩余论文阅读: article 文件夹还有 Guitar 论文等，读完后提取可用点进此表 | 用户 | P2 | ⬜ |

## F. 糊混音 / 分离域问题（2026-08-18 新增）

| # | 想法 | 来源 | 优先级 | 状态 |
|---|------|------|--------|------|
| F1 | **糊混音是域问题而非模型问题**：录音室混音（bus压缩/串音/bleed/限幅）让吉他频谱天然重叠，很多源头上就是糊的。方向1：混音风格增广——对干净 stem 施加 EQ/压缩/饱和/MP3 等链路模拟，让分离器见过各种"糊法" | 用户 | P1 | ⬜ |
| F2 | 方向2：下游域适应——AMT 直接在"糊 stem"（RoFormer 输出含伪影）上训练，把糊的问题交给转录端消化，而不是追求完美分轨 | 用户+codex讨论 | P1 | ⬜ |
| F3 | 方向3：bleed 分层评测——MoisesDB 197 条 has_bleed 轨已在 pilot manifest 标记，可对比模型在 bleed/no-bleed 上的 SI-SDR 与下游 onset F1，量化"糊"的影响 | ZCode | P2 | ⬜ |
| F4 | 方向4：主观糊度协议——SDR 不反映"听感糊"，建立 rhythm 段清晰度/attack 完整度/伪影三项人工打分，配合下游指标选型 | 用户 | P2 | ⬜ |
| F5 | **失真吉他保真度**：纯吉他段输出仍糊（quantum jump 开头），证明是模型对失真谱的保真问题而非分离干扰。候选：v2 训练数据失真轨加权；提高 mel 分辨率（num_bands/dim_f）；失真吉他专用二分模型；或接受现状走 F2（AMT 域适应吃糊 stem） | 用户听感+量化互证 | P1 | ⬜ |
| F6 | **人声预处理不必要**：vocal/inst 成对对比无现象级差异，Guitar/Other 的 Other 已吸收人声，管线无需加人声消除 | 用户听感 | 已验证 | ✅ |

## G. 新方向调研（2026-08-20，若 v2 仍不可用则启用）
| # | 方向 | 要点 | 状态 |
|---|------|------|------|
| G1 | **MuScriptor（Kyutai 2026-07）前端替换** | decoder-only Transformer + MT3 token 解码（架构代差 vs ByteDance/Riley 帧级回归头）；真实录音 11k 小时 + 合成 145 万 MIDI 训练；失真电吉他占真实集 46.8%（正中痛点）；36 乐器组含吉他细分；内部真实测试集 Onset F1 60.4 vs YourMT3+ 21.9；开源权重 60M-1.3B（github.com/muscriptor/muscriptor）；5s 分段 16kHz 推理，60M 本地可跑。验证：東の空/虚無の先 stems+原始混音双测、GuitarSet 同口径 | ⬜ 待试 |
| G2 | YourMT3+（ISMIR 2024）备选 | 开源生态成熟，cross-stem augmentation 对吉他增强；2025 AMT Challenge 前三全为 YourMT3+ 系（冠军 MIROS=MusicFM 编码器+T5 解码，Slakh F1 0.83） | ⬜ |
| G3 | EGDB（Chen&Hsiao 2025） | 240 首失真电吉他 tab 数据集，string/fret 阶段可用 | ⬜ |
| G4 | 关键认知 | 帧级 onset/offset 回归头（ByteDance/Riley 系）在真实混音上已到天花板；社区主路线已转 token 解码 + 大规模真实域数据。我们可复用资产：VER-SEP 分离器（MuScriptor 可吃 stem 或全混音+乐器条件）、F2 微调方法论、记谱层 | 记录 |
| G5 | **instrument-agnostic-amt（用户发现，2026-08-21 已实测采纳）** | anime-song/instrument-agnostic-amt：Transkun Neural Semi-CRF 区间解码 + 双轴 Transformer + HCQT，乐器无关，MIT 协议。实测（同口径）：GuitarSet 0.9115（-2.4pt vs fl，高精度）；東の空 onset@50 0.3647（+14pt vs fl 最佳）；虚無の先输出数量校准（2482≈参考2792）、时值中位 0.351s（fl 0.163s）；**原始混音≈stem 输入（分离可省）**；附力度/鼓/乐器分类/beat-chord；训练框架全开（stem 混合增广原生支持，将来域适应可用它而非 ByteDance trainer）。Windows 推理补丁：eval/ia_amt_run.py | ✅ 采纳为吉他前端 |
| G6 | ia-amt 后续增强方向 | ① `--instrument` / `--allowed-instruments` 吉他子类（acoustic/clean/distorted/muted/harmonics）条件化推理待试；② 其训练框架 + F2 数据（现成 1792 对）做吉他域适应——比 ByteDance trainer 更对齐（区间级监督）；③infer_stem.py 分轨转写工作流可接 VER-SEP stem；④ 鼓组/力度输出对最终记谱（谱面力度记号）直接可用 | ⬜ |

## H. ia-amt 弱点清单与创新点空位（2026-08-21 代码取证 + 实测）
| # | 弱点（代码/README/实证依据） | 创新点机会 |
|---|---|---|
| H1 | **无 tab/弦品输出**：PredictedNote 只有 pitch/区间/velocity/slot，无 string/fret。GuitarSet GT 与 EGDB（240 失真 tab）现成 | **最明确的空位**：pitch+timing→弦品指法的记谱层（可做 playability 约束的 DP/搜索），直通 GP 吉他谱成品。现有 AMT 模型都没做这一层 |
| H2 | **Solo 训练，混音依赖 stem 工作流**（infer_stem.py 用外部 stem_splitter 包）；我们的「raw mix≈stem」结论只测了吉他主导曲，多乐器混音会全转（instrument-agnostic 有音高就转） | 管线级乐器路由：ia-amt 乐器条件（--allowed-instruments/36 类）+ VER-SEP 分离 + 逐音乐器分类做多乐器混音→分轨乐谱系统 |
| H3 | **失真电吉他自认弱** + 实测精度低（東の空 P 0.239，泛音多检 2141 vs 参考 661）；CQT 22050Hz/84bins（顶 ~5.9kHz）丢高次泛音 | 用其 trainer（全开、区间级监督、stem 混合增广原生）+ 现成 F2 1792 对做吉他域适应微调 |
| H4 | **无逐音置信度**（PredictedNote 无 score 字段） | 校对界面/主动学习都需要；可在 semi-CRF 区间分数上导出（内部有 logit） |
| H5 | **velocity 是独立后处理模型**（guitar ckpt velocity=100 默认） | 力度头微调或后处理校准 |
| H6 | **无标准 benchmark 数字**（README 无对比表）；项目 2026-05 才首发，评测不公开 | 我们的三参考评测体系（GuitarSet/GP8/mscz + 分离域口径）可发布为 benchmark |
| H7 | **Note over-segmentation 列 known limitation**（我们实测时值中位 0.351s vs 参考 ~0.5s，仍偏碎） | 记谱层 tie/合并（我们 Layer 5 已有雏形，其长时值输出让这层首次可行） |
| H8 | 鼓组 experimental、乐器分类 experimental、分轨间时基对齐问题（README 自述 stem workflow 各轨独立转写会有小偏移） | 鼓轨用专门模型（如 2025 challenge 系）；全局时基校准层 |

## I. 论文定位（2026-08-21 研判）
**时间线（已核实）**：ICASSP 2027 全文截稿 2026-09-16（仅剩26天）；ISMIR 2026 LBD（2页非存档，滚动审稿）截稿 2026-09-25，会期 11-08~12 阿布扎比；ISMIR 正刊按惯例 ~4月截稿（2027届待公布）；EUSIPCO 2027 惯例 ~2月截稿；IEEE SPL 随时投稿（5页快审）。
**重要竞情修正（H1 不是空位！）**：tab/弦品方向 2025 年已热——TART（arXiv 2510.02597，四阶段 audio→tab 含弦品+技巧检测，EGDB 评测）、Fretting-Transformer（2506.14223，T5 编码解码 MIDI→tab）、MIDI→tab ML 方法（2510.10619）、SynthTab/EGDB（2309.09085）、IEEE Access 2025 综述。**但均为干净/DI 音频路线（EGDB=clean DI）；「真实乐队失真混音→tab」端到端仍是空位，且我们的 Semi-CRF 前端+三参考评测是差异点。定 H1 前必须精读 TART/Fretting-Transformer。**
**定位决策**：主线=真实混音条件下的吉他 AMT（失真域适应 / 真实混音 tab 生成二选一，精读竞品后定）；大创/MUSE 管线=系统与 demo 载体；评测体系=实验部分。近期目标 ISMIR 2026 LBD（MUSE 系统 demo，低风险练兵）；ICASSP 2027 仅当 9 月初实验出数才冲；中期 EUSIPCO 2027 / IEEE SPL；主目标 ISMIR 2027 正刊。
- CCF 分级（2026-03 第七版前后口径）：**ICASSP=B（会）、TASLP=B（刊）**；ISMIR=C（会）、EUSIPCO=C（会）、IEEE SPL=C（刊）；IEEE Access 不在 CCF 目录。**注意：CCF 只认 full paper——ISMIR LBD（2页非存档）不算**。保研/学院视角：B 类（ICASSP/TASLP）>C 类；方法主投建议锁定 ICASSP 2028 周期（2027-09 截稿）或 TASLP，快速产出走 SPL/EUSIPCO（C），ISMIR 正刊做社区影响力（C）。

## J. 遵义会议（2026-08-21）：论文主攻方向决议
**失真吉他为什么不准（机理清单，论文分析节素材）**：①失真=非线性整形→互调产物+谐波密度爆炸→频谱掩蔽，多音谐波纠缠；②压缩抹平 attack→onset 瞬态线索消失（漏检而非不准，ia-amt 检出时 onset 误差仅 17.5ms）；③泛音重影（基频弱谐波强，kyomu 40% 同 onset 多音高）；④带 GT 的失真数据稀缺（EGDB 是 clean DI；真实过载录音无 GT）；⑤HCQT 84 bins 顶 ~4.2kHz + pitch-slot 门控（待定位的漏检高音候选原因）。
**用户关键观察（金子）**：demucs 分离底噪大幅伤 ia-amt 输出 → 只能 raw 输入；而社区默认"先分离再转写"。若"分离伪影伤害 ≥ 干扰本身"系统性成立 = 反共识可发表发现。
**主推论文**：《分离还要不要？真实乐队混音下伪影鲁棒的吉他 AMT》三贡献：①分离×AMT 交互系统研究+失真失效分类学（三参考量化）；②伪影感知增广微调 ia-amt（F2 方法论换基座，trainer 全开）±泛音一致性解码；③真实混音多参考 benchmark+记谱可读性指标。投稿 ICASSP 2028（B，2027-09）/扩 ISMIR 2027/TASLP。差异化逻辑：不拼通用转写（数据拼不过），只打"真实失真混音→可用谱"（敌自认弱+无记谱层）。
**备选**：A 中国民乐 AMT（36 类无二胡；连续音高 vs Semi-CRF 离散区间假设天然冲突；无 GT 数据集需合成路线=复刻 SynthTab 方法论，工程月级，当第二篇）；B 节拍约束 Semi-CRF 记谱解码（最新颖最险，并入主推当第二贡献）。
**本周便宜实验**：1)定位漏检高音（diff raw vs stem pitch 集合）；2){原始,demucs,VER-SEP}×ia-amt×3曲 mini 矩阵验证伪影效应；3)失真失效分桶表；4)记谱量化文献（Cemgil 谱系）；5)服务器起 ia-amt 微调环境+F2 数据搬迁。
**查证补充**：夏雨=DeepPiano 创始人（智曲科技，清华 CS，实时钢琴转录陪练，产品无论文）；CCMUSIC(TISMIR)/ChMusic/二胡技巧集(1500 clips 11 技法)均无转写 GT。
