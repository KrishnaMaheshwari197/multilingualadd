"""
Dukaan Awaaz: a voice interview with a shopkeeper -> a spoken WhatsApp ad in their language.

Pipeline (all Sarvam AI):
  Bulbul (text-to-speech)  asks each question aloud, then speaks the finished ad
  Saaras (speech-to-text)  transcribes the spoken answers (code-mixed speech is fine)
  Sarvam-105B (chat)       turns the answers into a short radio-style ad script + a WhatsApp caption

Run:
  pip install -r requirements.txt
  put SARVAM_API_KEY=... in .env   (or paste the key in the sidebar)
  streamlit run app.py
"""

import base64
import json
import os
import re
import urllib.parse
from datetime import datetime

import requests
import streamlit as st
import streamlit.components.v1 as components

API = "https://api.sarvam.ai"
STT_MODEL = "saaras:v3"
CHAT_MODEL = "sarvam-105b"

# Read SARVAM_API_KEY (and anything else) from .env next to this file; a real env var wins.
_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env):
    for _line in open(_env, encoding="utf-8"):
        _name, _sep, _value = _line.strip().partition("=")
        if _sep and not _name.startswith("#"):
            os.environ.setdefault(_name.strip(), _value.strip())

# label -> (language code, English name)
LANGS = {
    "हिन्दी (Hindi)": ("hi-IN", "Hindi"),
    "বাংলা (Bengali)": ("bn-IN", "Bengali"),
    "English": ("en-IN", "English"),
    "मराठी (Marathi)": ("mr-IN", "Marathi"),
    "தமிழ் (Tamil)": ("ta-IN", "Tamil"),
    "తెలుగు (Telugu)": ("te-IN", "Telugu"),
    "ಕನ್ನಡ (Kannada)": ("kn-IN", "Kannada"),
    "മലയാളം (Malayalam)": ("ml-IN", "Malayalam"),
    "ગુજરાતી (Gujarati)": ("gu-IN", "Gujarati"),
    "ਪੰਜਾਬੀ (Punjabi)": ("pa-IN", "Punjabi"),
    "ଓଡ଼ିଆ (Odia)": ("od-IN", "Odia"),
}

# language -> {gender: (model, speaker)}. bulbul:v4-flash has native voices (and "ads" voices
# for Hindi, English and Tamil); bulbul:v3 fills the gaps.
V4, V3 = "bulbul:v4-flash", "bulbul:v3"
VOICES = {
    "hi-IN": {"male": (V4, "shubh_hi_ads"), "female": (V4, "ritu_hi_ads_informal")},
    "en-IN": {"male": (V4, "varun_en_ads"), "female": (V4, "simran_en_ads")},
    "bn-IN": {"male": (V4, "arnab_bn_conversation"), "female": (V4, "roopa_bn_conversational")},
    "mr-IN": {"male": (V4, "nilesh_mr_conversation"), "female": (V4, "ishita_mr_conversational")},
    "ta-IN": {"male": (V4, "aravind_ta_ads"), "female": (V3, "kavitha")},
    "te-IN": {"male": (V4, "tarun_te_narration"), "female": (V4, "kavitha_te_conversation")},
    "kn-IN": {"male": (V4, "chetan_kn_conversation"), "female": (V4, "chaitra_kn_conversation")},
    "ml-IN": {"male": (V3, "shubh"), "female": (V3, "priya")},
    "gu-IN": {"male": (V4, "bhavik_gu_conversation"), "female": (V4, "pooja_gu_conversational")},
    "pa-IN": {"male": (V4, "anand_pa_conversation"), "female": (V3, "simran")},
    "od-IN": {"male": (V3, "shubh"), "female": (V3, "priya")},
}
FALLBACK_VOICE = {"male": (V3, "shubh"), "female": (V3, "priya")}

QUESTIONS = {
    "en-IN": [
        "Namaste! What is the name of your shop, and where is it?",
        "What do you sell? What is your shop best known for?",
        "What offer or news do you want to tell your customers, and until when is it valid?",
        "On which phone number should customers contact you?",
    ],
    "hi-IN": [
        "नमस्ते! आपकी दुकान का नाम क्या है, और यह कहाँ पर है?",
        "आप क्या-क्या बेचते हैं? आपकी दुकान किस चीज़ के लिए मशहूर है?",
        "आप ग्राहकों को कौन सा ऑफ़र या ख़बर बताना चाहते हैं, और यह कब तक चलेगा?",
        "ग्राहक आपसे किस फ़ोन नंबर पर संपर्क करें?",
    ],
    "bn-IN": [
        "নমস্কার! আপনার দোকানের নাম কী, আর দোকানটা কোথায়?",
        "আপনি কী কী বিক্রি করেন? আপনার দোকান কোন জিনিসের জন্য বিখ্যাত?",
        "গ্রাহকদের কোন অফার বা খবর জানাতে চান, আর সেটা কতদিন চলবে?",
        "গ্রাহকরা কোন ফোন নম্বরে যোগাযোগ করবেন?",
    ],
}

AD_KEYS = ["shop_name", "script", "caption", "phone", "voice"]


# ----------------------------------------------------------------------------
# Sarvam API calls
# ----------------------------------------------------------------------------
def _fail(name, r):
    raise RuntimeError(f"Sarvam {name} failed ({r.status_code}): {r.text[:300]}")


def stt(audio_bytes, lang_code, key):
    """Speech -> text. The REST endpoint takes clips up to 30 seconds."""
    r = requests.post(
        f"{API}/speech-to-text",
        headers={"api-subscription-key": key},
        files={"file": ("answer.wav", audio_bytes, "audio/wav")},
        data={"model": STT_MODEL, "mode": "codemix", "language_code": lang_code},
        timeout=60,
    )
    if not r.ok:
        _fail("speech-to-text", r)
    return (r.json().get("transcript") or "").strip()


@st.cache_data(show_spinner=False)
def tts(text, lang_code, gender, _key, pace=1.0):
    """Text -> MP3 bytes in the language's voice. Cached, so each text is synthesised once."""
    model, speaker = VOICES.get(lang_code, FALLBACK_VOICE)[gender]
    attempts = [(model, speaker)]
    if model != V3:  # if a newer voice is unavailable, fall back to the classic one
        attempts.append(FALLBACK_VOICE[gender])
    for model, speaker in attempts:
        r = requests.post(
            f"{API}/text-to-speech",
            headers={"api-subscription-key": _key},
            json={"text": text[:2500], "language_code": lang_code, "model": model,
                  "speaker": speaker, "pace": pace, "output_audio_codec": "mp3"},
            timeout=90,
        )
        if r.ok:
            return b"".join(base64.b64decode(a) for a in r.json()["audios"])
    _fail("text-to-speech", r)


def chat(messages, key):
    headers = {"api-subscription-key": key}
    # Thinking is on by default; even at "low" it can use up every token and leave
    # the reply empty, so switch it off (null) for these short JSON jobs.
    body = {"model": CHAT_MODEL, "messages": messages, "temperature": 0.6,
            "max_tokens": 2048, "reasoning_effort": None,
            "response_format": {"type": "json_object"}}
    r = requests.post(f"{API}/v1/chat/completions", headers=headers, json=body, timeout=120)
    if not r.ok:  # retry with the bare minimum in case an optional field is rejected
        r = requests.post(f"{API}/v1/chat/completions", headers=headers,
                          json={"model": CHAT_MODEL, "messages": messages,
                                "reasoning_effort": None}, timeout=120)
    if not r.ok:
        _fail("chat", r)
    return r.json()["choices"][0]["message"].get("content") or ""


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("The model did not return JSON")
    return json.loads(m.group(0))


# ----------------------------------------------------------------------------
# Interview questions and ad writing
# ----------------------------------------------------------------------------
def get_questions(lang_code, lang_name, key):
    if lang_code in QUESTIONS:
        return QUESTIONS[lang_code]
    try:
        reply = chat([{
            "role": "user",
            "content": (
                f"Translate these questions into natural, friendly spoken {lang_name}, "
                f"written in the native {lang_name} script. Return only JSON of the form "
                '{"questions": ["...", "...", "...", "..."]}.\n\n'
                + json.dumps(QUESTIONS["en-IN"], ensure_ascii=False)
            ),
        }], key)
        qs = parse_json(reply).get("questions")
        if isinstance(qs, list) and len(qs) == len(QUESTIONS["en-IN"]):
            return [str(q) for q in qs]
    except Exception:
        pass
    return QUESTIONS["en-IN"]


def make_ad(lang_name, qa, key, previous=None, change=None):
    system = f"""You write short spoken radio-style ads for small neighbourhood shops in India.
The ad is sent as a voice note on WhatsApp, so it must sound natural when read aloud.
From the shopkeeper's spoken answers, return ONE JSON object with exactly these keys:
- shop_name: the shop's name
- script: the ad to be spoken, 50 to 80 words (about 25 seconds). Warm, lively and local.
  Write it fresh as an announcer inviting customers (not in the shopkeeper's first person,
  and do not copy their sentences). Say the shop name near the start and again at the end.
  Include the offer, how long it runs
  and where the shop is, if given. End with a call to action. Wherever the phone number goes,
  write exactly {{PHONE}} and nothing else (it is filled in later). No emojis, no hashtags,
  no stage directions, no bullet points, only the words to be spoken.
- caption: a WhatsApp text message of 2 to 4 short lines with a few emojis, to send along with
  the voice note; put {{PHONE}} where the phone number goes
- phone: the phone number exactly as the shopkeeper said it, digits only; "" if not given
- voice: "male" or "female" if the shopkeeper asked for one, otherwise ""

Rules:
- Write script and caption in {lang_name}, in its native script. Keep brand names and everyday
  English words the way people say them.
- Use ONLY facts from the answers. Never invent prices, discounts, dates, addresses or phone numbers.
- If there is no phone number, do not use {{PHONE}}.
- Output only the JSON object."""

    user = "Interview with the shopkeeper:\n" + "\n".join(f"Q: {q}\nA: {a}" for q, a in qa)
    if previous and change:
        user += (
            "\n\nCurrent ad JSON:\n" + json.dumps(previous, ensure_ascii=False)
            + f"\n\nThe shopkeeper asked for this change (spoken): {change}\n"
            "Apply it and return the full updated JSON."
        )

    best = None
    for _ in range(3):
        try:
            data = parse_json(chat([{"role": "system", "content": system},
                                    {"role": "user", "content": user}], key))
        except Exception:
            continue
        if not (isinstance(data, dict) and data.get("script")):
            continue
        ad = {k: str(data.get(k) or "").strip() for k in AD_KEYS}
        ad["phone"] = re.sub(r"\D", "", ad["phone"])
        # Only a spoken change request may switch the voice.
        if not change or ad["voice"] not in ("male", "female"):
            ad["voice"] = (previous or {}).get("voice", "")
        best = ad
        if not ad["phone"] or "{PHONE}" in ad["script"]:
            return ad
    if best:  # the model kept leaving the number out of the script: say it at the end
        best["script"] += " {PHONE}"
        if "{PHONE}" not in best["caption"]:
            best["caption"] += "\n📞 {PHONE}"
        return best

    # Last resort so the demo never dead-ends: read the answers out as they are.
    if previous:
        return previous
    answers = [a for _, a in qa] + [""] * 4
    return {"shop_name": answers[0], "script": " ".join(a for a in answers[:3] if a) + " {PHONE}",
            "caption": "\n".join(answers[:3]) + "\n📞 {PHONE}",
            "phone": re.sub(r"\D", "", answers[3]), "voice": ""}


def spoken_phone(digits):
    """Space the digits out so the voice reads them one by one: 98110 23456 -> 9 8 1 1 0, 2 3 4 5 6."""
    if not digits:
        return ""
    groups = [digits[:5], digits[5:]] if len(digits) == 10 else [digits]
    return ", ".join(" ".join(g) for g in groups if g)


def written_phone(digits):
    return f"{digits[:5]} {digits[5:]}" if len(digits) == 10 else digits


def fill_phone(text, phone):
    text = text.replace("{PHONE}", phone)
    return re.sub(r"\s+([,.!?।])", r"\1", text).strip()  # tidy up when the number is empty


# ----------------------------------------------------------------------------
# Share button (phones can hand the MP3 + caption straight to WhatsApp)
# ----------------------------------------------------------------------------
SHARE_HTML = r"""
<style>
  body { margin: 0; font-family: system-ui, sans-serif; background: transparent; }
  button { font: 600 15px system-ui, sans-serif; padding: 10px 18px; border-radius: 999px; border: 0;
           cursor: pointer; background: #25D366; color: #fff; width: 100%; }
  p { font-size: 13px; color: #888; margin: 6px 0 0; text-align: center; }
</style>
<button id="share">📤 Share voice ad to WhatsApp</button>
<p id="note"></p>
<script>
const D = __DATA__;
const bytes = Uint8Array.from(atob(D.mp3), (c) => c.charCodeAt(0));
const file = new File([bytes], 'voice-ad.mp3', { type: 'audio/mpeg' });
const btn = document.getElementById('share');
if (navigator.canShare && navigator.canShare({ files: [file] })) {
  btn.onclick = () => navigator.share({ files: [file], text: D.caption }).catch(() => {});
} else {
  btn.style.display = 'none';
  document.getElementById('note').textContent = 'Open this page on your phone to share directly to WhatsApp.';
}
</script>
"""


def share_html(mp3, caption):
    payload = json.dumps({"mp3": base64.b64encode(mp3).decode(), "caption": caption})
    return SHARE_HTML.replace("__DATA__", payload.replace("</", "<\\/"))


# ----------------------------------------------------------------------------
# Saved ads: one JSON (interview + ad) and one MP3 per ad, in history/
# ----------------------------------------------------------------------------
HISTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history")


def save_entry(entry, mp3=None):
    os.makedirs(HISTORY, exist_ok=True)
    with open(os.path.join(HISTORY, entry["id"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=1)
    if mp3 is not None:
        with open(os.path.join(HISTORY, entry["id"] + ".mp3"), "wb") as f:
            f.write(mp3)


def load_entry(entry_id):
    try:
        with open(os.path.join(HISTORY, entry_id + ".json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def saved_mp3(entry_id):
    try:
        with open(os.path.join(HISTORY, entry_id + ".mp3"), "rb") as f:
            return f.read()
    except OSError:
        return None


def list_entries():
    if not os.path.isdir(HISTORY):
        return []
    entries = (load_entry(name[:-5]) for name in os.listdir(HISTORY) if name.endswith(".json"))
    return sorted((e for e in entries if e), key=lambda e: e["id"], reverse=True)


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Dukaan Awaaz", page_icon="📢", layout="centered")
ss = st.session_state


def reset():
    for k in list(ss.keys()):
        if k not in ("api_key", "lang_label", "gender"):
            del ss[k]


def open_entry(entry_id):
    """Button callback: runs before the next rerun, so it may set widget values."""
    e = load_entry(entry_id)
    if not e or e.get("lang_label") not in LANGS:
        return
    reset()
    ss.lang_label = ss.active_lang = e["lang_label"]
    ss.gender = e.get("gender", "male")
    ss.questions, ss.answers, ss.ad = e["questions"], e["answers"], e["ad"]
    ss.entry_id = entry_id
    if e.get("last_change"):
        ss.last_change = e["last_change"]


def show_saved_ads():
    with st.sidebar:
        st.divider()
        st.subheader("📁 Saved ads")
        entries = list_entries()
        if not entries:
            st.caption("Every ad you make is saved here.")
        for e in entries[:50]:
            when = datetime.strptime(e["id"][:15], "%Y%m%d-%H%M%S").strftime("%d %b, %I:%M %p")
            shop = (e["ad"].get("shop_name") or "Ad")[:26]
            st.button(f"{shop} · {when}", key=f"open_{e['id']}", on_click=open_entry,
                      args=(e["id"],), help=e["lang_label"], use_container_width=True,
                      type="primary" if e["id"] == ss.get("entry_id") else "secondary")


if "pending_gender" in ss:  # a widget's value can only be set before it is drawn
    ss.gender = ss.pop("pending_gender")

with st.sidebar:
    st.header("Setup")
    typed_key = st.text_input("Sarvam API key", type="password", key="api_key",
                              help="Or set SARVAM_API_KEY in .env.")
    KEY = typed_key or os.environ.get("SARVAM_API_KEY", "")
    label = st.selectbox("Shopkeeper's language", list(LANGS), key="lang_label")
    GENDER = st.radio("Ad voice", ["male", "female"], horizontal=True, key="gender",
                      format_func=lambda g: "👨 Male" if g == "male" else "👩 Female")
    if st.button("🔄 Start over", use_container_width=True):
        reset()
        st.rerun()

LANG_CODE, LANG_NAME = LANGS[label]
if ss.get("active_lang") != label:
    reset()
    ss.active_lang = label
ss.setdefault("answers", [])
ss.setdefault("ad", None)
ss.setdefault("tries", 0)
ss.setdefault("edits", 0)

st.title("📢 Dukaan Awaaz")
st.caption("Answer four questions by voice in your own language. Get a spoken WhatsApp ad.")

if not KEY:
    st.info("Add your Sarvam API key in the sidebar to begin.")
    st.stop()

if ss.get("error"):
    st.error(ss.pop("error"))

if "questions" not in ss:
    with st.spinner("Preparing the interview…"):
        ss.questions = get_questions(LANG_CODE, LANG_NAME, KEY)
questions = ss.questions
step = len(ss.answers)

# Conversation so far
for q, a in zip(questions, ss.answers):
    with st.chat_message("assistant"):
        st.write(q)
    with st.chat_message("user"):
        st.write(a)

# ---------------------------- Interview ----------------------------
if step < len(questions):
    rec_key = f"rec_{step}_{ss.tries}"
    rec = ss.get(rec_key)
    if rec is not None:  # an answer was just recorded: transcribe it, then move on
        with st.spinner("Listening…"):
            try:
                text = stt(rec.getvalue(), LANG_CODE, KEY)
                if text:
                    ss.answers.append(text)
                else:
                    ss.error = "I could not hear anything. Please answer again."
            except Exception as e:
                ss.error = str(e)
        ss.tries += 1
        st.rerun()

    q = questions[step]
    with st.chat_message("assistant"):
        st.write(q)
        try:
            st.audio(tts(q, LANG_CODE, GENDER, KEY), format="audio/mpeg", autoplay=True)
        except Exception as e:
            st.caption(f"(Could not play the question aloud: {e})")

    st.audio_input(f"🎙️ Answer {step + 1} of {len(questions)} (up to 30 seconds)", key=rec_key)
    with st.expander("Microphone not working? Type the answer"):
        with st.form(f"type_{step}", clear_on_submit=True):
            typed = st.text_input("Answer")
            if st.form_submit_button("Send") and typed.strip():
                ss.answers.append(typed.strip())
                st.rerun()

# ---------------------------- Write the ad ----------------------------
elif ss.ad is None:
    with st.spinner("Writing your ad…"):
        ss.ad = make_ad(LANG_NAME, list(zip(questions, ss.answers)), KEY)
    ss.entry_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    st.rerun()

# ---------------------------- Result ----------------------------
else:
    edit_key = f"edit_{ss.edits}"
    edit = ss.get(edit_key)
    if edit is not None:  # a spoken change request
        with st.spinner("Updating your ad…"):
            try:
                change = stt(edit.getvalue(), LANG_CODE, KEY)
                if change:
                    ss.ad = make_ad(LANG_NAME, list(zip(questions, ss.answers)), KEY,
                                    previous=ss.ad, change=change)
                    ss.last_change = change
                    if ss.ad.get("voice") in ("male", "female"):
                        ss.pending_gender = ss.ad["voice"]  # "use a female voice" moves the sidebar too
                else:
                    ss.error = "I could not hear the change. Please try again."
            except Exception as e:
                ss.error = str(e)
        ss.edits += 1
        st.rerun()

    ad = ss.ad
    script = fill_phone(ad["script"], spoken_phone(ad["phone"]))
    caption = fill_phone(ad["caption"], written_phone(ad["phone"]))

    # Save the ad; reuse the saved MP3 if the words and voice have not changed.
    entry_id = ss.setdefault("entry_id", datetime.now().strftime("%Y%m%d-%H%M%S"))
    entry = {"id": entry_id, "lang_label": label, "gender": GENDER, "questions": questions,
             "answers": ss.answers, "ad": ad, "script": script,
             "last_change": ss.get("last_change", "")}
    old = load_entry(entry_id)
    same_audio = old and old.get("script") == script and old.get("gender") == GENDER
    mp3 = saved_mp3(entry_id) if same_audio else None

    st.subheader("Your voice ad")
    if mp3 is None:
        try:
            with st.spinner("Recording the ad…"):
                mp3 = tts(script, LANG_CODE, GENDER, KEY, pace=1.1)
        except Exception as e:
            st.error(str(e))
            st.stop()
        save_entry(entry, mp3)
    elif entry != old:
        save_entry(entry)
    st.audio(mp3, format="audio/mpeg", autoplay=True)

    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download voice ad (MP3)", mp3, file_name="voice-ad.mp3",
                       mime="audio/mpeg", use_container_width=True)
    c2.link_button("💬 Open WhatsApp with caption",
                   "https://wa.me/?text=" + urllib.parse.quote(caption),
                   use_container_width=True)
    components.html(share_html(mp3, caption), height=70)

    with st.expander("What the ad says"):
        st.write(script)

    st.markdown("**Caption** (use the copy icon, then paste it with the voice note in WhatsApp)")
    st.code(caption, language=None)

    st.divider()
    if ss.get("last_change"):
        st.caption(f"Last change you asked for: “{ss.last_change}”")
    st.audio_input("🎙️ Want a change? Say it (for example: “make it for Diwali”, "
                   "“make it shorter”, “use a female voice”)", key=edit_key)
    st.caption("To send: download the voice ad, open WhatsApp → your broadcast list → "
               "attach it as audio → paste the caption.")

show_saved_ads()  # last, so an ad saved during this run is already listed
