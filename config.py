# ============================================================
# config.py — Central config for the ASR benchmark pipeline
# ============================================================
from dotenv import load_dotenv
load_dotenv()
import os

# ── API Keys ──────────────────────────────────────────────────
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
SARVAM_API_KEY   = os.getenv("SARVAM_API_KEY")

# ── Paths ─────────────────────────────────────────────────────
AUDIO_DIR   = "Audio_vahan_dataset"
RESULTS_DIR = "results"

# ── Locality → file mapping ───────────────────────────────────
AUDIO_FILES = {
    # Original 20
    "Koramangala"    : "Koramangala .m4a",
    "Indiranagar"    : "Indiranagar .m4a",
    "Whitefield"     : "Whitefield .m4a",
    "Electronic City": "Electronic city .m4a",
    "Marathahalli"   : "Marathahalli .m4a",
    "Jayanagar"      : "Jayanagar .m4a",
    "Rajajinagar"    : "rajajinagar .m4a",
    "Hebbal"         : "Hebbel .m4a",
    "Yelahanka"      : "Yelahanka .m4a",
    "Banashankari"   : "Banashankari .m4a",
    "HSR Layout"     : "HSR layout .m4a",
    "BTM Layout"     : "BTM layout .m4a",
    "Majestic"       : "majestic .m4a",
    "Silk Board"     : "silk board.m4a",
    "Bellandur"      : "Bellandur .m4a",
    "Sarjapur"       : "Sarjapur .m4a",
    "Bommanahalli"   : "Bommanahalli .m4a",
    "KR Puram"       : "KR Puram.m4a",
    "Peenya"         : "peenya.m4a",
    "Yeshwanthpur"   : "Yeshwantpur .m4a",
    # 4 extra — different speakers / conditions
    "Indiranagar_2"  : "Indiranagar_2.m4a",
    "Bellandur_2"    : "Bellandur2.m4a",
    "Sarjapur_2"     : "Sarjapur_2.m4a",
    "Yeshwanthpur_2" : "yeshwantpur 2.m4a",
}

# ── Ground-truth transcripts ──────────────────────────────────
GROUND_TRUTH = {
    "Koramangala"    : "haan main koramangala mein rehta hoon",
    "Indiranagar"    : "mera ghar indiranagar ke paas hai",
    "Whitefield"     : "haan main whitefield se aata hoon",
    "Electronic City": "mera office electronic city mein hi hai",
    "Marathahalli"   : "marathahalli bridge ke paas drop karo",
    "Jayanagar"      : "jayanagar mein ek room dhundh raha hoon",
    "Rajajinagar"    : "rajajinagar se bus pakadta hoon main",
    "Hebbal"         : "hebbal flyover ke paas rehta hoon",
    "Yelahanka"      : "yelahanka new town mein shift hua hoon",
    "Banashankari"   : "banashankari temple ke paas ghar hai mera",
    "HSR Layout"     : "hsr layout sector 2 mein hoon main abhi",
    "BTM Layout"     : "btm layout first stage jaante ho",
    "Majestic"       : "majestic se bmtc leke jaata hoon",
    "Silk Board"     : "silk board par traffic bahut hai",
    "Bellandur"      : "bellandur road pe flat liya hai",
    "Sarjapur"       : "sarjapur road par kaam karta hoon",
    "Bommanahalli"   : "bommanahalli se koramangala 20 min hai",
    "KR Puram"       : "kr puram railway station ke paas khada hoon jaldi aaja",
    "Peenya"         : "peenya industrial area mein factory hai",
    "Yeshwanthpur"   : "yeshwanthpur se train pakadni hai mereko jaldi aa",
    # Extra samples
    "Indiranagar_2"  : "mera ghar indiranagar ke paas hi hai",
    "Bellandur_2"    : "bellandur lake ke paas flat liya hai",
    "Sarjapur_2"     : "sarjapur road pe khada hun jaldi aaja",
    "Yeshwanthpur_2" : "yeshwantpur se train leni hai jaldi aa",
}

# ── Audio condition tags ──────────────────────────────────────
AUDIO_CONDITIONS = {
    "Koramangala"    : "quiet",
    "Indiranagar"    : "quiet",
    "Whitefield"     : "quiet",
    "Electronic City": "quiet",
    "Marathahalli"   : "street_noise",
    "Jayanagar"      : "street_noise",
    "Rajajinagar"    : "street_noise",
    "Hebbal"         : "street_noise",
    "Yelahanka"      : "background_noise",
    "Banashankari"   : "background_noise",
    "HSR Layout"     : "rushed",
    "BTM Layout"     : "rushed",
    "Majestic"       : "whispered",
    "Silk Board"     : "whispered",
    "Bellandur"      : "phone_quality",
    "Sarjapur"       : "phone_quality",
    "Bommanahalli"   : "noisy_fast",
    "KR Puram"       : "noisy_fast",
    "Peenya"         : "quiet_mumbled",
    "Yeshwanthpur"   : "quiet",
    # Extra samples
    "Indiranagar_2"  : "street_noise",       # friend, moderate noise
    "Bellandur_2"    : "street_noise",       # friend, traffic noise
    "Sarjapur_2"     : "street_noise",       # you, traffic noise
    "Yeshwanthpur_2" : "street_noise",       # you, modulated voice, traffic
}

# ── Speaker tags (for multi-speaker analysis) ─────────────────
SPEAKER = {
    "Koramangala"    : "self",
    "Indiranagar"    : "self",
    "Whitefield"     : "self",
    "Electronic City": "self",
    "Marathahalli"   : "self",
    "Jayanagar"      : "self",
    "Rajajinagar"    : "self",
    "Hebbal"         : "self",
    "Yelahanka"      : "self",
    "Banashankari"   : "self",
    "HSR Layout"     : "self",
    "BTM Layout"     : "self",
    "Majestic"       : "self",
    "Silk Board"     : "self",
    "Bellandur"      : "self",
    "Sarjapur"       : "self",
    "Bommanahalli"   : "self",
    "KR Puram"       : "self",
    "Peenya"         : "self",
    "Yeshwanthpur"   : "self",
    "Indiranagar_2"  : "friend",
    "Bellandur_2"    : "friend",
    "Sarjapur_2"     : "self",
    "Yeshwanthpur_2" : "self_modulated",
}