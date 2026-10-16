# Stop the demo app and the Cloudflare tunnel started by tools\start_public_demo.ps1.
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like '*uvicorn*app.main:app*' -or $_.Name -eq 'cloudflared.exe' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Write-Host 'Demo and tunnel stopped.'
