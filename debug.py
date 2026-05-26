# debug.py — run this to find out what's going wrong
import os
from pathlib import Path
from config import AUDIO_DIR, AUDIO_FILES, DEEPGRAM_API_KEY, OPENAI_API_KEY

print("=== 1. API Keys ===")
print(f"Deepgram : {'SET ✓' if DEEPGRAM_API_KEY != 'YOUR_DEEPGRAM_KEY' else 'NOT SET ✗'}")
print(f"OpenAI   : {'SET ✓' if OPENAI_API_KEY   != 'YOUR_OPENAI_KEY'   else 'NOT SET ✗'}")

try:
    from config import SARVAM_API_KEY
    print(f"Sarvam   : {'SET ✓' if SARVAM_API_KEY != 'YOUR_SARVAM_KEY' else 'NOT SET ✗'}")
except:
    print("Sarvam   : NOT SET ✗")

print("\n=== 2. Audio Files ===")
missing, found = [], []
for locality, fname in AUDIO_FILES.items():
    full = Path(AUDIO_DIR) / fname
    if full.exists():
        size = full.stat().st_size
        found.append(f"  ✓ {locality:<20} → {fname}  ({size} bytes)")
    else:
        missing.append(f"  ✗ {locality:<20} → {fname}  ← FILE NOT FOUND")

for f in found:  print(f)
for m in missing: print(m)

print(f"\n{len(found)}/20 files found")

print("\n=== 3. Test one Deepgram call ===")
if DEEPGRAM_API_KEY != 'YOUR_DEEPGRAM_KEY' and found:
    import requests
    # pick first found file
    first_locality = next(l for l in AUDIO_FILES if (Path(AUDIO_DIR) / AUDIO_FILES[l]).exists())
    first_file = Path(AUDIO_DIR) / AUDIO_FILES[first_locality]
    print(f"Testing with: {first_file}")
    with open(first_file, "rb") as f:
        resp = requests.post(
            "https://api.deepgram.com/v1/listen",
            params={"model": "nova-2-general", "language": "hi"},
            headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/mp4"},
            data=f, timeout=30
        )
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text[:300]}")
else:
    print("Skipped — fix API key or files first")