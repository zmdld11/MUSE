# 一体化引导 v2: 补齐 MAESTRO 30s 缓存 (与训练 MAX_DUR=30 一致)
#   -> 启动混合训练 -> 写 PID -> 启动监控循环
# 修复: $args 是 PowerShell 保留自动变量, 改名 $trainArgs
$ErrorActionPreference = "Continue"
$ws = "d:\program_project\MUSE\score_extraction"
$py = "C:/Users/ROG/.conda/envs/score_build/python.exe"
$log = "$ws\train\bootstrap.log"
$pidFile = "$ws\train\train.pid"

function W($m) {
    Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [BOOT] $m"
}

Add-Content $log "===== bootstrap 启动 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="

# 1. 补齐 MAESTRO 30s 缓存 (train 全量 + validation 50)
$env:PYTHONIOENCODING = "utf-8"
W "生成 MAESTRO 30s 缓存 (train + validation)..."
& $py -u -c @"
import sys, logging
sys.path.insert(0, r'$ws')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
from train.maestro_dataset import MaestroDataset
t = MaestroDataset(split='train', max_dur_sec=30)
print(f'CACHE30 train: {len(t)} 段', flush=True)
v = MaestroDataset(split='validation', max_files=50, max_dur_sec=30)
print(f'CACHE30 validation: {len(v)} 段', flush=True)
"@ 2>&1 | Out-File -Append $log
W "MAESTRO 30s 缓存就绪, 启动训练..."

# 2. 启动混合训练 (后台, 记 PID)
$trainArgs = @(
    "-u", "train/train_overnight.py",
    "--epochs", "100", "--lr", "1e-4",
    "--save", "VER3.0_MixedReal",
    "--resume", "model/VER2.2_BootstrapFull_latest.pt"
)
$proc = Start-Process -FilePath $py -ArgumentList $trainArgs -WorkingDirectory $ws `
    -WindowStyle Hidden -PassThru
if (-not $proc) { W "!! 训练启动失败"; exit 1 }
Set-Content $pidFile $proc.Id
W "训练已启动 PID=$($proc.Id)"

# 3. 监控循环 (20 分钟间隔)
$trainLog = "$ws\train\log_overnight.log"
$monitorLog = "$ws\train\monitor.log"
while ($true) {
    Start-Sleep -Seconds 1200
    $ts = Get-Date -Format "HH:mm:ss"
    $alive = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if (-not $alive) {
        $last = Get-Content $trainLog -Tail 3 -ErrorAction SilentlyContinue
        if ($last -match "Done in") { W "$ts 训练正常完成" }
        else { W "$ts !! 训练进程消失且无完成标志 — 崩溃" }
        break
    }
    $lastWrite = (Get-Item $trainLog).LastWriteTime
    $age = ((Get-Date) - $lastWrite).TotalMinutes
    if ($age -gt 30) {
        W "$ts !! 日志 $([math]::Round($age)) 分钟无更新 — 卡死, 停止"
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        break
    }
    $tail = Get-Content $trainLog -Tail 30 -ErrorAction SilentlyContinue
    if ($tail -match "nan|NaN|inf|-inf") {
        W "$ts !! 检测到 NaN/Inf — 停止训练"
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        break
    }
    $ep = ($tail | Select-String "Epoch \d+/\d+: train_loss=" | Select-Object -Last 1).Line
    if ($ep) { W "$ts $ep" }
}

Add-Content $log "===== bootstrap 结束 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
