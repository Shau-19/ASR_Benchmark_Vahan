import time
import requests
from pathlib import Path
from config import OPENAI_API_KEY


# ── Single-file transcription ─────────────────────────────────

def transcribe(audio_path: str) -> dict:
    """
    Send one audio file to OpenAI Whisper API and return a result dict.

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

    url     = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    t0 = time.time()
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                url,
                headers=headers,
                files={"file": (path.name, f, _mime(path))},
                data={
                    "model"   : "whisper-1",
                    "language": "hi",       # hint: Hindi; handles code-switch well
                    "response_format": "text",
                },
                timeout=60,
            )
        latency = round(time.time() - t0, 3)

        if resp.status_code != 200:
            return _error_result(f"HTTP {resp.status_code}: {resp.text[:200]}")

        return {
            "transcript": resp.text.strip(),
            "latency_s" : latency,
            "model"     : "whisper-1",
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
        print(f"  [Whisper]  {locality} ...", end=" ", flush=True)
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
        "model"     : "whisper-1",
        "error"     : msg,
    }
