# AGENTS.md — MUSE 识谱系统工作规则（每次会话自动加载）

## 定位

本仓库 = **聆谱 MUSE 识谱系统**（开源、公开）。论文与研究线在私有工作区
`D:\program_project\VocalTest`（工作规则见其 AGENTS.md，开工先读那边看板）。

## 工作流

- 系统改动照旧：改完留工作区，**不 git commit/push——用户自理**
- 研究类请求（模型训练/评测/论文写作）→ 按 VocalTest/AGENTS.md 的双工作区模式执行；
  实验代码与数据仍在 MUSE 本地目录（`score_extraction/{train,eval}/`、`data/`、`output/`
  已脱离 git 跟踪，属私有研究资产，勿重新纳入）

## 铁律（违反=事故）

1. 不 git commit/push——用户自理（除非用户明确委托的 git 手术类任务）
2. 本机 python = 项目根 `env/python.exe`，且 cwd 必须在项目根（相对路径锚点）
3. 服务器只动 `~/zmdld11`；大上传前 `df -h` 现查；删任何东西先报清单等用户点头
4. 公开仓库只放识谱系统：研究产物（新脚本/权重/记录）一律不 `git add`，
   `.gitignore` 已隔离 `score_extraction/{eval,train 研究部分,model}`
