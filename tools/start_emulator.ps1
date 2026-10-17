# Start the Android emulator with the Mongolian Latin keyboard installed and selected (Windows, no Android Studio).
#
#   powershell -ExecutionPolicy Bypass -File tools\start_emulator.ps1            # window, for the projector
#   powershell -ExecutionPolicy Bypass -File tools\start_emulator.ps1 -Build     # rebuild the APK first
#   powershell -ExecutionPolicy Bypass -File tools\start_emulator.ps1 -NoWindow  # headless (adb screencap)
#
# Expects the portable tools laid out by the 2026-10-18 setup (see keyboard/android/README.md):
#   C:\Users\<you>\tools\jdk\jdk-17.*, tools\gradle-8.9, tools\android-sdk (cmdline-tools, platform-tools,
#   emulator, platforms;android-34, build-tools;34.0.0, system-images;android-34;google_apis;x86_64) and an AVD "kb34".
param([switch]$Build, [switch]$NoWindow, [string]$Tools = "$env:USERPROFILE\tools", [string]$Avd = "kb34")

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$jdk = (Get-ChildItem "$Tools\jdk" -Directory | Select-Object -First 1).FullName
$env:JAVA_HOME = $jdk
$env:ANDROID_HOME = "$Tools\android-sdk"
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
$env:Path = "$jdk\bin;$Tools\gradle-8.9\bin;$env:ANDROID_HOME\platform-tools;$env:ANDROID_HOME\emulator;" + $env:Path
$apk = "$repo\keyboard\android\app\build\outputs\apk\debug\app-debug.apk"
$ime = "mn.translit.keyboard/.MnLatinIme"

if ($Build -or -not (Test-Path $apk)) {
    Push-Location "$repo\keyboard\android"
    try { gradle assembleDebug --console=plain } finally { Pop-Location }
}

$running = (& adb devices) -match "emulator-"
if (-not $running) {
    $emuArgs = @("-avd", $Avd, "-no-audio", "-no-boot-anim", "-no-snapshot")
    if ($NoWindow) { $emuArgs += @("-no-window", "-gpu", "swiftshader_indirect") }
    Start-Process -FilePath "emulator" -ArgumentList $emuArgs -WindowStyle Normal
    Write-Host "Booting $Avd ..."
    $deadline = (Get-Date).AddMinutes(5)
    do { Start-Sleep -Seconds 5; $boot = (& adb -e shell getprop sys.boot_completed 2>$null) } while ($boot -ne "1" -and (Get-Date) -lt $deadline)
    if ($boot -ne "1") { throw "emulator did not boot in 5 minutes" }
}

& adb -e install -r $apk | Out-Null
& adb -e shell ime enable $ime | Out-Null
& adb -e shell ime set $ime | Out-Null
# a text field to type into: Chrome's address bar (verified 2026-10-18; dismiss its first-run dialogs once)
& adb -e shell am start -n com.android.chrome/com.google.android.apps.chrome.Main -d "about:blank" 2>$null | Out-Null
Write-Host "Keyboard installed and selected. Tap the address bar; the strip shows the Cyrillic candidates."
Write-Host "Switch keyboards with the keyboard icon in the navigation bar if Gboard comes up instead."
Write-Host "Screenshot: adb -e shell screencap -p /sdcard/s.png; adb -e pull /sdcard/s.png shot.png"
