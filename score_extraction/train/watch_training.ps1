# 训练 watchdog (2026-08-04): 检测 DataLoader worker 挂死并自动重启训练.
# 逻辑: 每 60s 检查 log_overnight.log 是否在 10 分钟内有过写入;
#       若超时或训练进程消失 (且未完成), 杀掉并 resume 重启 (最多损失 1 epoch).
$ErrorActionPreference = 'SilentlyContinue'

$wd = 'D:\program_project\MUSE\score_extraction'
$log = Join-Path $wd 'train\log_overnight.log'
$py = 'C:\Users\ROG\.conda\envs\score_build\python.exe'
$pidFile = Join-Path $wd 'train\train.pid'
$watchLog = Join-Path $wd 'train\watchdog.log'

function Write-WatchLog([string]$msg) {
    Add-Content -Path $watchLog -Value ("{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg) -Encoding UTF8
}

function Start-Training {
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $py
    $psi.Arguments = 'train\train_overnight.py --resume model\VER3.0_MixedReal_latest.pt --epochs 100 --lr 1e-4 --save VER3.0_MixedReal --workers 2 --onset-weight 5 --onset-dilate 0'
    $psi.WorkingDirectory = $wd
    $psi.UseShellExecute = $true
    $psi.CreateNoWindow = $true
    $psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $p = [System.Diagnostics.Process]::Start($psi)
    Set-Content -Path $pidFile -Value $p.Id
    Write-WatchLog ("started training PID={0}" -f $p.Id)
}

function Test-Done {
    if (Test-Path $log) {
        $tail = Get-Content $log -Tail 20 -Encoding UTF8
        return ($tail -match 'Done in')
    }
    return $false
}

Write-WatchLog "watchdog started"
Start-Training

while ($true) {
    Start-Sleep -Seconds 60

    if (Test-Done) {
        Write-WatchLog "training finished, watchdog exits"
        break
    }

    $trainPid = 0
    if (Test-Path $pidFile) { $trainPid = [int](Get-Content $pidFile) }
    $alive = $null -ne (Get-Process -Id $trainPid -ErrorAction SilentlyContinue)

    $stale = $false
    if (Test-Path $log) {
        $lastWrite = (Get-Item $log).LastWriteTime
        $stale = ((Get-Date) - $lastWrite).TotalMinutes -gt 10
    }

    if (-not $alive) {
        Write-WatchLog ("training PID {0} dead, restarting" -f $trainPid)
        Start-Training
    }
    elseif ($stale) {
        Write-WatchLog ("training PID {0} stale {1:N1}min, killing and restarting" -f $trainPid, ((Get-Date) - (Get-Item $log).LastWriteTime).TotalMinutes)
        taskkill /PID $trainPid /T /F | Out-Null
        Start-Sleep -Seconds 5
        Start-Training
    }
}
