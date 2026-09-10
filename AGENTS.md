# AGENTS.md — MUSE 识谱系统工作规则（每次会话自动加载）

## 定位

本仓库 = **聆谱 MUSE 识谱系统**（开源、公开）。研究线两个私有仓（2026-09-10 重组）：

- `D:\program_project\VV-SVT` — VV-SVT 研究仓：实验台账（issue 板）+ 实验记录 +
  研究代码快照（开工先读那边 HANDOFF.md）
- `D:\program_project\VV-SVT-Paper` — 论文写作与投稿、调研与选题、文献库

## 工作流

- 系统改动照旧：**当天改动当天 commit，commit 后直接 push origin**（09-08 授权；
  tag/Release 仍等用户指令）
- 研究类请求（模型训练/评测/论文写作）→ 按 VV-SVT/AGENTS.md 执行；
  **live 研究代码仍在 MUSE 本地目录**（`score_extraction/{train,eval}/`、`data/`、`output/`
  已脱离 git 跟踪，属私有研究资产，勿重新纳入）；VV-SVT 仓持有其代码快照，
  收工跑 `VV-SVT\sync_from_muse.ps1` 同步（09-17 截稿后研究代码物理迁出 MUSE）

## 铁律（违反=事故）

1. git：**当天改动当天 commit，commit 后直接 push origin**（push 授权 2026-09-08；
   跨天未提交修改曾整块丢失）；tag/Release 仍等用户指令
2. 本机 python = 项目根 `env/python.exe`，且 cwd 必须在项目根（相对路径锚点）
3. 服务器只动 `~/zmdld11`；大上传前 `df -h` 现查；删任何东西先报清单等用户点头
4. 公开仓库只放识谱系统：研究产物（新脚本/权重/记录）一律不 `git add`，
   `.gitignore` 已隔离 `score_extraction/{eval,train 研究部分,model}`；
   研究数字/内容绝不进本仓（commit、issue 一律不进）
