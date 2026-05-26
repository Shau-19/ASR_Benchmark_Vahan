# paste in python shell
import requests, os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("SARVAM_API_KEY")
with open("Audio_vahan_dataset/peenya.m4a", "rb") as f:
    r = requests.post(
        "https://api.sarvam.ai/speech-to-text",
        headers={"api-subscription-key": key},
        files={"file": ("peenya.m4a", f, "audio/mp4")},
        data={"language_code": "hi-IN", "model": "saarika:v2"},
        timeout=60
    )
print(r.status_code, r.text[:400])