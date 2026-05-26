import time
import requests
from pathlib import Path

# Set your Sarvam API key as env var SARVAM_API_KEY or paste below
import os
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")


# ── Single-file transcription ─────────────────────────────────

def transcribe(audio_path: str) -> dict:
    """
    Send one audio file to Sarvam AI and return a result dict.

    Returns:
        {
            "transcript": str,
            "latency_s" : float,
            "model"     : str,
            "error"     : str | None
        }
    """
    path = Path(audio_path)
    if not path.exists():
        return _error_result(f"File not found: {audio_path}")

    url     = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": SARVAM_API_KEY}

    t0 = time.time()
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                url,
                headers=headers,
                files={"file": (path.name, f, _mime(path))},
                data={
                    "language_code"  : "hi-IN",   # Hindi; also try "kn-IN" for Kannada
                    "model"          : "saarika:v2.5",
                    "with_timestamps": "false",
                },
                timeout=60,
            )
        latency = round(time.time() - t0, 3)

        if resp.status_code != 200:
            return _error_result(f"HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        # Sarvam returns { "transcript": "...", ... }
        transcript = data.get("transcript", "")

        return {
            "transcript": transcript,
            "latency_s" : latency,
            "model"     : "sarvam-saarika-v2",
            "error"     : None,
        }

    except Exception as e:
        return _error_result(str(e))


# ── Batch helper ──────────────────────────────────────────────

def transcribe_batch(audio_files: dict) -> dict:
    """
    audio_files: { locality_name: filepath, ... }
    Returns:     { locality_name: result_dict, ... }
    """
    results = {}
    for locality, filepath in audio_files.items():
        print(f"  [Sarvam]   {locality} ...", end=" ", flush=True)
        results[locality] = transcribe(filepath)
        status = "✓" if not results[locality]["error"] else "✗"
        print(status)
    return results


# ── Internal ──────────────────────────────────────────────────

def _mime(path: Path) -> str:
    return {
        ".wav" : "audio/wav",
        ".mp3" : "audio/mpeg",
        ".m4a" : "audio/mp4",
        ".ogg" : "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }.get(path.suffix.lower(), "audio/wav")


def _error_result(msg: str) -> dict:
    return {
        "transcript": "",
        "latency_s" : 0.0,
        "model"     : "sarvam-saarika-v2",
        "error"     : msg,
    }