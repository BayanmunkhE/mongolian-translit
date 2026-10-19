"""Type on the emulator IME via adb taps, grab a screenshot after every key, build an animated GIF."""
import subprocess, sys, time, io
from PIL import Image

ADB = r"C:\Users\zyota\tools\android-sdk\platform-tools\adb.exe"
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/keyboard/emulator_typing.gif"
TEXT = sys.argv[2] if len(sys.argv) > 2 else "margaash uulzy "

KEYS = {}
for i, c in enumerate("qwertyuiop"):
    KEYS[c] = (54 + 108 * i, 1834)
for i, c in enumerate("asdfghjkl"):
    KEYS[c] = (60 + 120 * i, 1960)
for i, c in enumerate("zxcvbnm"):
    KEYS[c] = (180 + 120 * i, 2086)
KEYS[" "] = (270, 2212)  # bottom row: space (5 units) + - , . ! ? (1 unit each) over 1080 px
KEYS["-"] = (594, 2212); KEYS[","] = (702, 2212); KEYS["."] = (810, 2212); KEYS["!"] = (918, 2212); KEYS["?"] = (1026, 2212)
SHIFT = (60, 2086)


def adb(*args):
    return subprocess.run([ADB, "-e", *args], capture_output=True, timeout=30)


def tap(x, y):
    adb("shell", "input", "tap", str(x), str(y))


def shot():
    adb("shell", "screencap", "-p", "/sdcard/f.png")
    raw = subprocess.run([ADB, "-e", "exec-out", "cat", "/sdcard/f.png"], capture_output=True, timeout=30).stdout
    return Image.open(io.BytesIO(raw)).convert("RGB")


def crop(im):
    # omnibox row + keyboard area, like results/keyboard/emulator_ime_crop.png
    top = im.crop((0, 120, 1080, 230))
    kb = im.crop((0, 1650, 1080, 2262))
    out = Image.new("RGB", (1080, top.height + kb.height + 16), (246, 244, 238))
    out.paste(top, (0, 0)); out.paste(kb, (0, top.height + 16))
    return out.resize((720, int(out.height * 720 / 1080)), Image.LANCZOS)


frames, durations = [], []
# close anything on top (Lens etc.), focus the omnibox and clear it with hardware DEL keys
adb("shell", "input", "keyevent", "4"); time.sleep(0.5)
adb("shell", "am", "start", "-n", "com.android.chrome/com.google.android.apps.chrome.Main", "-d", "about:blank"); time.sleep(1.5)
tap(480, 202); time.sleep(1.0)
adb("shell", "input", "keyevent", "KEYCODE_MOVE_END")
for _ in range(40):
    adb("shell", "input", "keyevent", "67")
time.sleep(0.8)
frames.append(crop(shot())); durations.append(900)
for ch in TEXT:
    if ch.isupper():
        tap(*SHIFT); time.sleep(0.25)
    x, y = KEYS[ch.lower()]
    tap(x, y)
    time.sleep(0.5)
    frames.append(crop(shot()))
    durations.append(1100 if ch in " ,.!?" else 550)
durations[-1] = 2500
frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
import os
print(f"{len(frames)} frames -> {OUT} ({os.path.getsize(OUT) / 2**20:.2f} MB)")
