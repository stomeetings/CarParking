<#
  Full local snapshot of the project (no git): code + config + ./data + PostgreSQL + image tags, with a self-check.

    powershell -File scripts\snapshot.ps1 [-Label task2-pre] [-Cold] [-SaveImages] [-Dest C:\F\Backups\CarParking]

  Writes <Dest>\<yyyyMMdd-HHmmss>-<Label>\ with: code\ (everything but data, backups, node_modules, dist), data\ (SQLite files
  copied with SQLite's online backup, so a running worker is fine), pg\carparking.dump (pg_dump -Fc), manifest.json (hashes,
  image ids, compose state) and optionally images\*.tar. The images are also tagged :snap-<stamp>.
  -Cold   stop worker + sync while the databases are copied (about 10 s), then start them again.
  Exit code 0 only when every check passed. The snapshot holds .env (secrets): keep the folder private.
  Restore: scripts\restore.ps1.
#>
param(
    [string]$Label = "manual",
    [switch]$Cold,
    [switch]$SaveImages,
    [string]$Dest = "C:\F\Backups\CarParking",
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = "Continue"
Set-Location $Root
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$safe = ($Label -replace "[^A-Za-z0-9_.-]", "-")
$snap = Join-Path $Dest "$stamp-$safe"
$fail = New-Object System.Collections.Generic.List[string]
function Step($m) { Write-Host "[snapshot] $m" }
function Bad($m) { $script:fail.Add($m); Write-Host "[snapshot] FAILED: $m" -ForegroundColor Red }

if ($snap.StartsWith($Root, [StringComparison]::OrdinalIgnoreCase)) { throw "Dest must be outside the project folder" }
New-Item -ItemType Directory -Force "$snap\code", "$snap\data", "$snap\pg" | Out-Null

# 1. code + config (everything except live data, old backups and rebuildable dependency folders)
$xd = @("$Root\data", "$Root\backups", "$Root\node_modules", "$Root\frontend\node_modules", "$Root\frontend\dist", "__pycache__")
$xf = @("*.pyc", "*.db-journal", "*.db-wal", "*.db-shm")
Step "code -> $snap\code"
robocopy $Root "$snap\code" /E /XD $xd /XF $xf /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { Bad "robocopy code exit $LASTEXITCODE" }

# 2. data: plain files by robocopy, SQLite databases through the online backup API in a one-off container of the live image
Step "data files -> $snap\data"
robocopy "$Root\data" "$snap\data" /E /XF *.db *.db-journal *.db-wal *.db-shm /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { Bad "robocopy data exit $LASTEXITCODE" }
$stopped = $false
try {
    if ($Cold) { Step "cold: stopping worker + sync"; docker compose stop worker sync | Out-Null; $stopped = $true }
    Step "SQLite databases (online backup)"
    $out = docker compose run --rm --no-deps -v "${snap}\data:/out" -v "${Root}\scripts\db_backup.py:/tmp/db_backup.py:ro" worker python /tmp/db_backup.py /data /out
    if ($LASTEXITCODE -ne 0) { Bad "db_backup exit $LASTEXITCODE" }
    $dbs = @($out | Where-Object { $_ -like "{*" } | ForEach-Object { $_ | ConvertFrom-Json })
    foreach ($d in $dbs) { Write-Host ("[snapshot]   {0}: {1} ({2:N0} bytes)" -f $d.db, $d.check, $d.bytes) }
    if ($dbs.Count -eq 0) { Bad "no SQLite database was backed up" }
}
finally {
    if ($stopped) { Step "cold: starting worker + sync"; docker compose start worker sync | Out-Null }
}

# 3. PostgreSQL (custom-format dump: restorable with pg_restore, carries geometry versions, camera registry, history)
Step "PostgreSQL dump"
docker compose exec -T db pg_dump -U carparking -Fc -f /tmp/carparking.dump carparking
if ($LASTEXITCODE -ne 0) { Bad "pg_dump exit $LASTEXITCODE" }
docker compose cp db:/tmp/carparking.dump "$snap\pg\carparking.dump" | Out-Null
if ($LASTEXITCODE -ne 0) { Bad "docker compose cp exit $LASTEXITCODE" }
$toc = docker compose exec -T db pg_restore --list /tmp/carparking.dump
if ($LASTEXITCODE -ne 0 -or -not ($toc | Select-String "TABLE DATA")) { Bad "pg_restore --list found no table data" }
docker compose exec -T db rm -f /tmp/carparking.dump | Out-Null

# 4. images: local tags (instant), optional tar files (about 6 GB)
$images = @("stomeetings/carparking:local", "stomeetings/carparking-setup:local")
$imageInfo = @()
foreach ($img in $images) {
    $id = docker image inspect $img --format "{{.Id}}" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $id) { Write-Host "[snapshot] image $img not present, skipped"; continue }
    $tag = ($img -split ":")[0] + ":snap-$stamp"
    docker tag $img $tag
    if ($LASTEXITCODE -ne 0) { Bad "docker tag $tag" }
    $imageInfo += [ordered]@{ name = $img; id = $id; snap_tag = $tag; tar = $null }
}
if ($SaveImages -and $imageInfo.Count) {
    New-Item -ItemType Directory -Force "$snap\images" | Out-Null
    foreach ($i in $imageInfo) {
        $tar = "images\" + ($i.snap_tag -replace "[/:]", "_") + ".tar"
        Step "docker save $($i.snap_tag)"
        docker save -o "$snap\$tar" $i.snap_tag
        if ($LASTEXITCODE -ne 0) { Bad "docker save $($i.snap_tag)" } else { $i.tar = $tar }
    }
}

# 5. manifest (no secret is read into it: files are only hashed)
Step "hashing"
$ps = docker compose ps -a --format "{{.Service}}|{{.Image}}|{{.Status}}"
$running = @(docker compose ps --services --status running)
$files = Get-ChildItem $snap -Recurse -File | ForEach-Object {
    [ordered]@{ path = $_.FullName.Substring($snap.Length + 1); bytes = $_.Length; sha256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash }
}
$manifest = [ordered]@{
    stamp = $stamp; label = $Label; root = $Root; cold = [bool]$Cold; created = (Get-Date).ToString("o")
    images = $imageInfo; compose_ps = @($ps); running_services = $running
    setup_was_running = ($running -contains "setup"); files = @($files)
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content "$snap\manifest.json" -Encoding UTF8

# 6. self-check: files re-hashed from disk, code mirror identical to the source, SQLite copies integrity-checked above
Step "verifying"
foreach ($f in $manifest.files) {
    if ((Get-FileHash (Join-Path $snap $f.path) -Algorithm SHA256).Hash -ne $f.sha256) { Bad "hash mismatch $($f.path)" }
}
$diff = robocopy $Root "$snap\code" /E /L /XD $xd /XF $xf /NJH /NJS /NP /NDL /R:0 /W:0 | Where-Object { $_ -match "\S" -and $_ -notmatch "^\s*(Same|\*EXTRA)" }
$diff = @($diff | Where-Object { $_ -match "(New File|Newer|Older|Changed|Modified)" })
if ($diff.Count) { Write-Host "[snapshot] code differs from source (files changed while copying?):"; $diff | Select-Object -First 10 | ForEach-Object { Write-Host "   $_" } }
$count = @($manifest.files).Count
$mb = [math]::Round((($manifest.files | ForEach-Object { $_.bytes } | Measure-Object -Sum).Sum) / 1MB, 1)
if ($fail.Count) {
    Write-Host "[snapshot] FAILED ($($fail.Count) problem(s)): $snap" -ForegroundColor Red
    exit 1
}
Write-Host "[snapshot] OK: $snap  ($count files, $mb MB, $($imageInfo.Count) image tag(s))" -ForegroundColor Green
Write-Output $snap
exit 0
