import time
import requests
from pathlib import Path
from config import DEEPGRAM_API_KEY


# ── Single-file transcription ─────────────────────────────────

def transcribe(audio_path: str) -> dict:
    """
    Send one audio file to Deepgram and return a result dict.

    Returns:
        {
            "transcript" : str,
            "confidence" : float,   # Deepgram's own confidence score
            "latency_s"  : float,   # wall-clock seconds for the API call
            "model"      : str,
            "error"      : str | None
        }
    """
    path = Path(audio_path)
    if not path.exists():
        return _error_result(f"File not found: {audio_path}")

    # Detect MIME type from extension
    mime_map = {
        ".wav" : "audio/wav",
        ".mp3" : "audio/mpeg",
        ".m4a" : "audio/mp4",
        ".ogg" : "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }
    mime = mime_map.get(path.suffix.lower(), "audio/wav")

    url = "https://api.deepgram.com/v1/listen"
    params = {
        "model"     : "nova-2-general",
        "language"  : "hi",          # Hindi; Deepgram auto-detects code-switch
        "punctuate" : "false",
        "diarize"   : "false",
    }
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type" : mime,
    }

    t0 = time.time()
    try:
        with open(path, "rb") as f:
            resp = requests.post(url, params=params, headers=headers, data=f, timeout=30)
        latency = round(time.time() - t0, 3)

        if resp.status_code != 200:
            return _error_result(f"HTTP {resp.status_code}: {resp.text[:200]}")

        data       = resp.json()
        channel    = data["results"]["channels"][0]["alternatives"][0]
        transcript = channel.get("transcript", "")
        confidence = channel.get("confidence", 0.0)

        return {
            "transcript": transcript,
            "confidence": round(confidence, 3),
            "latency_s" : latency,
            "model"     : "deepgram-nova-2",
            "error"     : None,
        }

    except Exception as e:
        return _error_result(str(e))


# ── Batch helper ─────────────────────────────────────────────

def transcribe_batch(audio_files: dict) -> dict:
    """
    audio_files: { locality_name: filepath, ... }
    Returns:     { locality_name: result_dict, ... }
    """
    results = {}
    for locality, filepath in audio_files.items():
        print(f"  [Deepgram] {locality} ...", end=" ", flush=True)
        results[locality] = transcribe(filepath)
        status = "✓" if not results[locality]["error"] else "✗"
        print(status)
    return results


# ── Internal ──────────────────────────────────────────────────

def _error_result(msg: str) -> dict:
    return {
        "transcript": "",
        "confidence": 0.0,
        "latency_s" : 0.0,
        "model"     : "deepgram-nova-2",
        "error"     : msg,
    }
