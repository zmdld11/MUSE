# AGENTS.md — MUSE 识谱系统工作规则（每次会话自动加载）

## 定位

本仓库 = **聆谱 MUSE 识谱系统**（开源、公开）。姊妹仓（2026-09-14 重组定稿）：

- `D:\program_project\VV-SVT` — **论文模型仓**（私有）：VT 塔式人声转写线（音符+韵母+颤音头）
  的训练/推理/评测代码（平铺开源训练仓格式）+ 实验台账（issue 板）+ 实验记录；
  开工先读那边 HANDOFF.md
- `D:\program_project\VV-SVT-Paper` — 论文写作与投稿、调研与选题、文献库（纯文本）
- dyylab 服务器 `~/zmdld11/score_extraction` — **仍是 09-14 前旧布局**，迁服待办见 VV-SVT issue 板

## 仓库布局（2026-09-14 训练代码迁出后）

- **四大工作区只留运行必需**：`score_extraction/src/`（管线）+ `score_extraction/model/`
  （运行 ckpt：`vocal_crnn/m3st500`、`vt_demo/` 三件套）+ `score_extraction/external/`
  （第三方权重）+ `score_extraction/data/`（仅 2 个先验 JSON）+ `score_extraction/output/`
  （仅 web_jobs/one_click）+ `instrument_recognition/`、`source_separation/`、`frontend/`
- **`MUSE\train\`（git 隔离）** = 本仓自训模型训练区：`vocal_crnn/`（m 系生产线）、
  `guitar_crnn/`、`sep_tasep/`、`piano_legacy/`、`nam/`、`build/`（数据构建）、
  `eval/`（MUSE 线评测）、`data/ model/ output/ legacy/`（训练数据/研究 ckpt/实验产物归档）
- **`score_extraction/runtime/`（git 隔离）** = 推理代码部署副本：`vocal_crnn/`（m3 引擎三件）、
  `vt/`（塔式引擎件）、`piano/`（model/model_v4）、`notation/`（validate_musicxml）。
  **改模型代码后须手动同步副本**（塔线源头在 VV-SVT 仓，其余在 `MUSE\train\`）
- 人声引擎开关 `MUSE_VOCAL_ENGINE = some|m3|vt`（默认 some；vt = VV-SVT 塔式部署配置）

## 工作流

- 系统改动照旧：**当天改动当天 commit，commit 后直接 push origin**（09-08 授权；
  tag/Release 仍等用户指令）
- 研究类请求（模型训练/评测/论文写作）→ 按 VV-SVT/AGENTS.md 执行；
  塔线代码直接改 VV-SVT 仓（live 树），m 系/吉他/分离训练线改 `MUSE\train\`

## 铁律（违反=事故）

1. git：**当天改动当天 commit，commit 后直接 push origin**（push 授权 2026-09-08；
   跨天未提交修改曾整块丢失）；tag/Release 仍等用户指令
2. 本机 python = 项目根 `env/python.exe`，且 cwd 必须在项目根（相对路径锚点）
3. 服务器只动 `~/zmdld11`；大上传前 `df -h` 现查；删任何东西先报清单等用户点头
4. 公开仓库只放识谱系统：研究产物（新脚本/权重/记录）一律不 `git add`，
   `.gitignore` 已隔离 `/train/`、`score_extraction/runtime/`、权重与产物目录；
   研究数字/内容绝不进本仓（commit、issue 一律不进）
