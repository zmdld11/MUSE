# 训练监控循环 (独立于会话存活)
# 每 20 分钟检查训练日志: 卡死/崩溃/发散/NaN, 异常则停止训练进程
$log = "d:\program_project\MUSE\score_extraction\train\monitor.log"
$trainLog = "d:\program_project\MUSE\score_extraction\train\log_overnight.log"
$pidFile = "d:\program_project\MUSE\score_extraction\train\train.pid"
$trainPid = if (Test-Path $pidFile) { [int](Get-Content $pidFile) } else { $null }

function Write-Mon($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content $log "$ts [MONITOR] $msg"
}

Add-Content $log "===== 监控启动 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="

$lastEpochLine = ""
$stallCount = 0

while ($true) {
    Start-Sleep -Seconds 1200  # 20 分钟
    $ts = Get-Date -Format "HH:mm:ss"

    # 1. 训练进程是否存活
    $trainProc = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $_.Id -eq $trainPid
    }
    if (-not $trainProc) {
        # 进程不存在: 检查日志是否正常完成
        $lastLines = Get-Content $trainLog -Tail 5 -ErrorAction SilentlyContinue
        if ($lastLines -match "Done in") {
            Write-Mon "$ts 训练正常完成"
            break
        }
        Write-Mon "$ts !! 训练进程消失 (PID=$trainPid)，日志无完成标志 — 可能崩溃"
        break
    }

    # 2. 卡死检测: 日志最后修改时间 > 30 分钟
    $lastWrite = (Get-Item $trainLog).LastWriteTime
    $ageMin = ((Get-Date) - $lastWrite).TotalMinutes
    if ($ageMin -gt 30) {
        Write-Mon "$ts !! 日志 $([math]::Round($ageMin)) 分钟无更新 — 疑似卡死，停止训练"
        Stop-Process -Id $trainPid -Force -ErrorAction SilentlyContinue
        break
    }

    # 3. 发散/NaN 检测
    $tail = Get-Content $trainLog -Tail 30 -ErrorAction SilentlyContinue
    if ($tail -match "nan|NaN|inf|-inf") {
        Write-Mon "$ts !! 检测到 NaN/Inf loss — 停止训练"
        Stop-Process -Id $trainPid -Force -ErrorAction SilentlyContinue
        break
    }

    # 4. epoch 指标趋势
    $epochLine = ($tail | Select-String "Epoch \d+/\d+: train_loss=" | Select-Object -Last 1).Line
    if ($epochLine) {
        Write-Mon "$ts $epochLine"
        if ($lastEpochLine -and $epochLine -ne $lastEpochLine) {
            # 解析 val_loss 是否反弹 (连续2次上升 >30%)
            $curVal = [regex]::Match($epochLine, "val_loss=([\d.]+)").Groups[1].Value
            $lastVal = [regex]::Match($lastEpochLine, "val_loss=([\d.]+)").Groups[1].Value
            if ($curVal -and $lastVal -and [double]$lastVal -gt 0) {
                $ratio = [double]$curVal / [double]$lastVal
                if ($ratio -gt 1.5) {
                    Write-Mon "$ts !! val_loss 反弹 ($lastVal -> $curVal, x$([math]::Round($ratio,2)))"
                }
            }
        }
        $lastEpochLine = $epochLine
    }

}

Add-Content $log "===== 监控结束 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
