# Start the demo on this PC and expose it with a free Cloudflare quick tunnel (no account needed).
#   powershell -ExecutionPolicy Bypass -File tools\start_public_demo.ps1
# Prints a public https://*.trycloudflare.com link. If a tunnel is already running it is kept, so the link stays the
# same; a new tunnel (after a reboot) gets a new link. Works only while this PC, the app and the tunnel keep running
# (no sleep). Stop both: tools\stop_public_demo.ps1. Free the GPU but keep the link: stop only the uvicorn process.
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$logDir = Join-Path $repo 'results\public_demo'
New-Item -ItemType Directory -Force $logDir | Out-Null
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $env:MODEL_DEVICE = 'cuda' }  # GPU: ~5x faster per message

Start-Process "$repo\.venv\Scripts\python.exe" -ArgumentList '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '7860' `
    -WorkingDirectory $repo -WindowStyle Minimized -RedirectStandardError "$logDir\app.log" | Out-Null
Write-Host 'Starting the app (about 30 s)...'
$ok = $false
for ($i = 0; $i -lt 60 -and -not $ok; $i++) {
    Start-Sleep 2
    try { Invoke-RestMethod http://127.0.0.1:7860/health -TimeoutSec 2 | Out-Null; $ok = $true } catch {}
}
if (-not $ok) { throw "The app did not start; see $logDir\app.log" }

$urlFile = Join-Path $logDir 'url.txt'
if ((Get-Process cloudflared -ErrorAction SilentlyContinue) -and (Test-Path $urlFile)) {
    # The tunnel survived an app restart: keep its link (and any QR code printed from it)
    Write-Host "Public link (tunnel kept): $(Get-Content $urlFile)"
    return
}
$cf = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cf) { $cf = 'C:\Program Files (x86)\cloudflared\cloudflared.exe' }
$tlog = Join-Path $logDir 'tunnel.log'
if (Test-Path $tlog) { Remove-Item $tlog }
Start-Process $cf -ArgumentList 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:7860', '--logfile', $tlog `
    -WindowStyle Minimized | Out-Null
$url = $null
for ($i = 0; $i -lt 30 -and -not $url; $i++) {
    Start-Sleep 2
    if (Test-Path $tlog) {
        $m = Select-String -Path $tlog -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' | Select-Object -First 1
        if ($m) { $url = $m.Matches[0].Value }
    }
}
if (-not $url) { throw "No tunnel link yet; see $tlog" }
$url | Set-Content $urlFile
Write-Host "Public link: $url"
