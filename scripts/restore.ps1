<#
  Roll the project back to a snapshot made by scripts\snapshot.ps1.

    powershell -File scripts\restore.ps1 -Snapshot C:\F\Backups\CarParking\<stamp>-<label> [-Code] [-Images] [-Data] [-Pg] [-Force] [-WhatIf]

  Default (no switch) = -Code -Images: the software goes back, the history recorded since is kept.
  -Data   also puts back ./data (SQLite files, config, slot maps)         } these throw away history written after the
  -Pg     also puts back the PostgreSQL dump (geometry versions, history) } snapshot and need -Force
  -WhatIf shows what would happen and changes nothing.
  A pre-restore snapshot of the current state is taken first, so the restore can itself be undone.
#>
param(
    [Parameter(Mandatory = $true)][string]$Snapshot,
    [switch]$Code, [switch]$Images, [switch]$Data, [switch]$Pg,
    [switch]$Force, [switch]$WhatIf, [switch]$NoPreSnapshot,
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = "Continue"
Set-Location $Root
function Step($m) { Write-Host "[restore] $m" }
function Die($m) { Write-Host "[restore] STOP: $m" -ForegroundColor Red; exit 1 }

if (-not (Test-Path "$Snapshot\manifest.json")) { Die "no manifest.json in $Snapshot" }
if (-not ($Code -or $Images -or $Data -or $Pg)) { $Code = $true; $Images = $true }
if (($Data -or $Pg) -and -not $Force -and -not $WhatIf) { Die "-Data / -Pg discard history written after the snapshot: add -Force" }
$m = Get-Content "$Snapshot\manifest.json" -Raw | ConvertFrom-Json
Step "snapshot $($m.stamp) '$($m.label)': code=$Code images=$Images data=$Data pg=$Pg whatif=$WhatIf"

# the snapshot must still be intact before anything is overwritten
Step "verifying the snapshot's hashes"
foreach ($f in $m.files) {
    $p = Join-Path $Snapshot $f.path
    if (-not (Test-Path $p) -or (Get-FileHash $p -Algorithm SHA256).Hash -ne $f.sha256) { Die "snapshot file damaged or missing: $($f.path)" }
}
$xd = @("$Root\data", "$Root\backups", "$Root\node_modules", "$Root\frontend\node_modules", "$Root\frontend\dist", "__pycache__")
$xf = @("*.pyc", "*.db-journal", "*.db-wal", "*.db-shm")

if ($WhatIf) {
    if ($Code) { Step "would mirror code\ over the project (files added since are removed):"; robocopy "$Snapshot\code" $Root /MIR /L /XD $xd /XF $xf /NJH /NJS /NP /NDL | Where-Object { $_ -match "(New File|Newer|Older|\*EXTRA)" } | Select-Object -First 40 }
    if ($Data) { Step "would mirror data\ over .\data" }
    if ($Pg) { Step "would pg_restore pg\carparking.dump into the carparking database" }
    if ($Images) { $m.images | ForEach-Object { Step "would re-tag $($_.snap_tag) as $($_.name)" } }
    Step "then docker compose up -d --no-build"
    exit 0
}

if (-not $NoPreSnapshot) {
    Step "pre-restore snapshot of the current state"
    & powershell -NoProfile -File "$PSScriptRoot\snapshot.ps1" -Label "pre-restore" -Root $Root -Dest (Split-Path -Parent $Snapshot)
    if ($LASTEXITCODE -ne 0) { Die "the pre-restore snapshot failed; nothing was changed" }
}

Step "stopping worker, sync, ui, api"
docker compose stop worker sync ui api | Out-Null

if ($Code) {
    Step "code: mirroring"
    robocopy "$Snapshot\code" $Root /MIR /XD $xd /XF $xf /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { Die "robocopy code exit $LASTEXITCODE" }
}
if ($Data) {
    Step "data: mirroring"
    robocopy "$Snapshot\data" "$Root\data" /MIR /XF *.db-journal *.db-wal *.db-shm /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { Die "robocopy data exit $LASTEXITCODE" }
}
if ($Pg) {
    Step "postgres: pg_restore"
    docker compose up -d db | Out-Null
    docker compose cp "$Snapshot\pg\carparking.dump" db:/tmp/carparking.dump | Out-Null
    docker compose exec -T db pg_restore -U carparking -d carparking --clean --if-exists --no-owner --single-transaction /tmp/carparking.dump
    if ($LASTEXITCODE -ne 0) { Die "pg_restore exit $LASTEXITCODE (the single transaction was rolled back)" }
    docker compose exec -T db rm -f /tmp/carparking.dump | Out-Null
}
if ($Images) {
    foreach ($i in $m.images) {
        docker image inspect $i.snap_tag *> $null
        if ($LASTEXITCODE -ne 0 -and $i.tar) { Step "docker load $($i.tar)"; docker load -i (Join-Path $Snapshot $i.tar) | Out-Null }
        docker image inspect $i.snap_tag *> $null
        if ($LASTEXITCODE -ne 0) { Die "image $($i.snap_tag) is gone and the snapshot has no tar (use -SaveImages next time, or rebuild from the restored code)" }
        docker tag $i.snap_tag $i.name
        Step "image $($i.name) -> $($i.snap_tag)"
    }
}

Step "starting the stack"
docker compose up -d --no-build | Out-Null
if ($m.setup_was_running) { docker compose --profile setup up -d --no-build setup | Out-Null }

Step "waiting for healthy services (up to 3 min)"
$deadline = (Get-Date).AddMinutes(3)
do {
    Start-Sleep 5
    $states = @(docker compose ps --format "{{.Service}}={{.Health}}") | Where-Object { $_ -match "=(starting|unhealthy)$" }
} while ($states.Count -and (Get-Date) -lt $deadline)
if ($states.Count) { Die "not healthy: $($states -join ', ')" }
$port = if ($env:CARPARK_API_PORT) { $env:CARPARK_API_PORT } else { "8080" }
try { $r = Invoke-WebRequest "http://localhost:$port/api/health" -UseBasicParsing -TimeoutSec 10; Step "api health $($r.StatusCode)" }
catch { Die "api health check failed: $($_.Exception.Message)" }
if ($Code) {
    $bad = 0
    foreach ($f in $m.files | Where-Object { $_.path -like "code\*" }) {
        $rel = $f.path.Substring(5)
        $p = Join-Path $Root $rel
        if (-not (Test-Path $p) -or (Get-FileHash $p -Algorithm SHA256).Hash -ne $f.sha256) { $bad++ }
    }
    if ($bad) { Write-Host "[restore] note: $bad code file(s) differ from the snapshot (the running stack may have rewritten them)" -ForegroundColor Yellow }
}
Write-Host "[restore] OK: restored from $Snapshot" -ForegroundColor Green
exit 0
