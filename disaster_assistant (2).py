"""
ResQora  -  AI Disaster Shield  |  single-file web app
=======================================================
Open-weight models (run locally through Ollama):
  * SLM (small, fast)   -> instant triage: disaster type, urgency, language
  * LLM (larger, deeper) -> detailed guidance for non-critical situations
Live, key-free data: Open-Meteo (weather) + USGS (earthquakes)
Works even if the models are offline: built-in offline playbooks (EN/HI).

SETUP
  1) Install Ollama: https://ollama.com  then:
       ollama pull llama3.2:3b      # SLM
       ollama pull llama3.1:8b      # LLM   (or qwen2.5:7b, mistral, gemma2:9b ...)
  2) pip install fastapi uvicorn httpx
  3) python disaster_assistant.py     ->  open http://localhost:8000

ENV (optional): OLLAMA_URL, SLM_MODEL, LLM_MODEL, PORT

SAFETY: This is a support tool, NOT a replacement for official services.
Always call local emergency services first. Numbers below are India defaults
- change EMERGENCY_NUMBERS for your country.
"""
import os, json, re
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
SLM = os.getenv("SLM_MODEL", "llama3.2:3b")
LLM = os.getenv("LLM_MODEL", "llama3.1:8b")

EMERGENCY_NUMBERS = [  # verify for your region
    {"name": "National Emergency", "number": "112"},
    {"name": "Ambulance", "number": "108"},
    {"name": "Fire", "number": "101"},
    {"name": "Police", "number": "100"},
    {"name": "NDMA Disaster Helpline", "number": "1078"},
]

# --------------------------------------------------------------------------
# PROMPTS
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are ResQora, an AI Disaster Response Assistant for floods, earthquakes, fires, cyclones, landslides, extreme weather and other emergencies.
Goals: protect human life; give clear actionable guidance; identify risks; never invent information.
RULES:
- Safety instructions first. Simple language. Reply in the user's language (English, Hindi or other).
- NEVER invent alerts, evacuation orders, shelter locations, emergency numbers or rescue availability. If unknown, say so.
- Separate verified data (only the VERIFIED DATA block given to you) from prediction/uncertainty.
- Always tell the user to call local emergency services (112 in India) for life-threatening danger.
- Ask at most ONE short question, only if needed.
- Flood: warn about rising water, never walk/drive through floodwater, move to higher ground.
- Earthquake: Drop, Cover, Hold On during shaking; afterwards beware gas leaks, fires, aftershocks, damaged buildings.
- Fire: move away from smoke/fire, evacuate if ordered or in immediate danger, stay low in smoke.
- Cyclone: follow official orders, stay away from coast and flood-prone areas.
- Landslide: move away from unstable slopes, watch for cracks/falling rocks.
FORMAT (keep EXACT headings; if life-threatening keep it SHORT):
🚨 ALERT LEVEL: <Critical/High/Moderate/Low/Unknown>
📍 LOCATION: <known location or "Location unavailable">
⚠️ WHAT IS HAPPENING: <short>
🛡️ WHAT TO DO NOW:
- action
🏠 SAFEST OPTION: <Evacuate / Stay indoors / Move to higher ground / Seek shelter / Other>
📌 IMPORTANT RISKS:
- risk
📡 INFORMATION STATUS: <Verified / Partially verified / Prediction / Unknown>"""

TRIAGE_PROMPT = """Classify the emergency message. Reply ONLY with JSON:
{"disaster":"flood|earthquake|fire|cyclone|landslide|heatwave|other|none",
 "urgency":"CRITICAL|HIGH|MODERATE|LOW|UNKNOWN",
 "language":"en|hi|other"}
CRITICAL = person is in immediate life danger now (trapped, drowning, injured, shaking, fire nearby)."""

# --------------------------------------------------------------------------
# OFFLINE PLAYBOOKS (used as fallback and for the offline-guides tab)
# --------------------------------------------------------------------------
PLAYBOOKS = {
    "earthquake": {
        "en": {"happening": "Earthquake / ground shaking.",
               "do": ["During shaking: DROP to hands and knees, take COVER under a sturdy table, HOLD ON.",
                      "Stay away from windows, shelves and heavy objects. If outdoors, move to an open area.",
                      "After shaking: check for injuries, expect aftershocks, smell for gas - if so, leave and do not use flames or switches."],
               "safest": "Stay low and covered; then move outdoors to an open area if the building is damaged.",
               "risks": ["Aftershocks", "Damaged buildings / falling objects", "Gas leaks and fires"]},
        "hi": {"happening": "भूकंप / ज़मीन हिल रही है।",
               "do": ["झटकों के दौरान: झुकें, मज़बूत मेज़ के नीचे छिपें और उसे पकड़े रहें।",
                      "खिड़कियों, अलमारियों और भारी सामान से दूर रहें। बाहर हों तो खुली जगह जाएँ।",
                      "झटके रुकने पर: चोट जाँचें, आफ्टरशॉक के लिए तैयार रहें। गैस की गंध हो तो बाहर निकलें, माचिस/स्विच का उपयोग न करें।"],
               "safest": "नीचे झुके और ढके रहें; इमारत क्षतिग्रस्त हो तो खुले स्थान पर जाएँ।",
               "risks": ["आफ्टरशॉक", "क्षतिग्रस्त इमारतें / गिरता सामान", "गैस रिसाव और आग"]}},
    "flood": {
        "en": {"happening": "Flooding / rapidly rising water.",
                "do": ["Move to higher ground or an upper floor NOW. Do not wait for water to rise further.",
                       "Never walk, swim or drive through floodwater - 15 cm of moving water can knock you down.",
                       "Switch off electricity at the main if safe; keep phone, torch, water, medicines and documents with you."],
                "safest": "Move to higher ground.",
                "risks": ["Fast-rising water and strong currents", "Electric shock", "Contaminated water", "Blocked roads"]},
        "hi": {"happening": "बाढ़ / पानी तेज़ी से बढ़ रहा है।",
               "do": ["तुरंत ऊँची जगह या ऊपरी मंज़िल पर जाएँ। पानी और बढ़ने का इंतज़ार न करें।",
                      "बाढ़ के पानी में पैदल, तैरकर या गाड़ी से न जाएँ - बहता पानी आपको गिरा सकता है।",
                      "सुरक्षित हो तो मेन बिजली बंद करें; फ़ोन, टॉर्च, पानी, दवाइयाँ और दस्तावेज़ साथ रखें।"],
               "safest": "ऊँची जगह पर जाएँ।",
               "risks": ["तेज़ बहाव", "बिजली का करंट", "दूषित पानी", "बंद सड़कें"]}},
    "fire": {
        "en": {"happening": "Fire / smoke danger.",
               "do": ["Leave immediately. Do not stop for belongings. Use stairs, not lifts.",
                      "Stay low under smoke, cover nose and mouth with a wet cloth. Touch doors with the back of your hand before opening.",
                      "Once out, stay out. Call the fire service (101 / 112)."],
               "safest": "Evacuate.",
               "risks": ["Toxic smoke", "Rapid fire spread", "Blocked exits"]},
        "hi": {"happening": "आग / धुएँ का खतरा।",
               "do": ["तुरंत बाहर निकलें। सामान के लिए न रुकें। लिफ्ट नहीं, सीढ़ियों का उपयोग करें।",
                      "धुएँ में झुककर चलें, गीले कपड़े से नाक-मुँह ढकें। दरवाज़ा खोलने से पहले हाथ के पिछले हिस्से से छूकर देखें।",
                      "बाहर निकलने के बाद वापस न जाएँ। फ़ायर सर्विस (101 / 112) को कॉल करें।"],
               "safest": "तुरंत बाहर निकलें।",
               "risks": ["ज़हरीला धुआँ", "आग का तेज़ फैलाव", "बंद निकास"]}},
    "cyclone": {
        "en": {"happening": "Cyclone / severe storm.",
               "do": ["Follow official evacuation orders immediately. Move away from the coast and low-lying areas.",
                      "If sheltering: stay indoors, away from windows, in the strongest inner room. Keep torch, radio, water, food.",
                      "Do not go out during the calm 'eye' - winds return suddenly."],
               "safest": "Seek shelter (or evacuate if ordered).",
               "risks": ["Destructive winds", "Storm surge and flooding", "Flying debris", "Power cuts"]},
        "hi": {"happening": "चक्रवात / भीषण तूफ़ान।",
               "do": ["सरकारी निकासी आदेश का तुरंत पालन करें। तटीय और निचले इलाकों से दूर जाएँ।",
                      "घर में रहें तो खिड़कियों से दूर, सबसे मज़बूत भीतरी कमरे में रहें। टॉर्च, रेडियो, पानी, भोजन रखें।",
                      "तूफ़ान की 'आँख' के शांत समय में बाहर न निकलें - हवा अचानक लौट आती है।"],
               "safest": "आश्रय लें (आदेश हो तो निकासी करें)।",
               "risks": ["विनाशकारी हवाएँ", "समुद्री उफान और बाढ़", "उड़ता मलबा", "बिजली कटौती"]}},
    "landslide": {
        "en": {"happening": "Landslide / unstable slope danger.",
               "do": ["Move away from the slope immediately, sideways out of the path - not downhill along it.",
                      "Watch for cracks, tilting trees, muddy water or rumbling sounds. Avoid river channels.",
                      "Do not return until authorities say it is safe."],
               "safest": "Move to stable, higher ground away from slopes.",
               "risks": ["Further slides", "Blocked roads", "Buried or damaged buildings"]},
        "hi": {"happening": "भूस्खलन / अस्थिर ढलान का खतरा।",
               "do": ["तुरंत ढलान से दूर हटें - ढलान के नीचे की ओर नहीं, बल्कि बगल की ओर।",
                      "दरारें, झुकते पेड़, गंदा पानी या गड़गड़ाहट पर ध्यान दें। नदी-नालों से दूर रहें।",
                      "अधिकारियों के सुरक्षित कहने तक वापस न जाएँ।"],
               "safest": "ढलानों से दूर स्थिर ऊँची जगह जाएँ।",
               "risks": ["और भूस्खलन", "बंद सड़कें", "इमारतों को नुकसान"]}},
    "heatwave": {
        "en": {"happening": "Extreme heat.",
               "do": ["Move to a cool, shaded place. Drink water or ORS regularly.",
                      "Avoid going out 12-4 pm. Loosen clothing, cool the body with wet cloths.",
                      "Dizziness, confusion or no sweating = heatstroke: call 108 / 112 now."],
               "safest": "Stay indoors in the shade.",
               "risks": ["Heatstroke", "Dehydration", "Children, elderly most at risk"]},
        "hi": {"happening": "भीषण गर्मी।",
               "do": ["ठंडी, छायादार जगह जाएँ। पानी या ORS बार-बार पिएँ।",
                      "दोपहर 12-4 बजे बाहर न निकलें। ढीले कपड़े पहनें, गीले कपड़े से शरीर ठंडा करें।",
                      "चक्कर, भ्रम या पसीना बंद होना = लू: तुरंत 108 / 112 पर कॉल करें।"],
               "safest": "छाया में घर के अंदर रहें।",
               "risks": ["लू लगना", "डिहाइड्रेशन", "बच्चे और बुज़ुर्ग सबसे अधिक जोखिम में"]}},
}

KEYWORDS = {
    "earthquake": ["earthquake", "quake", "shaking", "tremor", "भूकंप", "भूकम्प", "झटके"],
    "flood": ["flood", "water level", "rising water", "inundat", "waterlog", "बाढ़", "बाढ", "पानी भर"],
    "fire": ["fire", "smoke", "burning", "blaze", "आग", "धुआँ", "धुआं"],
    "cyclone": ["cyclone", "hurricane", "storm", "typhoon", "चक्रवात", "तूफ़ान", "तूफान"],
    "landslide": ["landslide", "mudslide", "rockfall", "भूस्खलन", "पहाड़ खिसक"],
    "heatwave": ["heatwave", "heat wave", "heatstroke", "लू", "भीषण गर्मी"],
}
CRITICAL_WORDS = ["trapped", "stuck", "drowning", "bleeding", "unconscious", "can't breathe", "cannot breathe",
                  "collapsed", "help me", "sos", "dying", "fire in", "shaking now", "फंस", "बचाओ", "डूब", "खून"]


def heuristic_triage(msg: str) -> dict:
    m = msg.lower()
    disaster = next((d for d, ks in KEYWORDS.items() if any(k in m for k in ks)), "none")
    critical = any(k in m for k in CRITICAL_WORDS)
    urgency = "CRITICAL" if critical else ("HIGH" if disaster != "none" else "UNKNOWN")
    lang = "hi" if re.search(r"[\u0900-\u097F]", msg) else "en"
    return {"disaster": disaster, "urgency": urgency, "language": lang}


# --------------------------------------------------------------------------
# MODEL CALLS (Ollama)
# --------------------------------------------------------------------------
async def ollama_chat(model, messages, fmt=None, max_tokens=500, timeout=90):
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": 0.2, "num_predict": max_tokens}}
    if fmt:
        payload["format"] = fmt
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(f"{OLLAMA}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]


async def triage(msg: str) -> dict:
    base = heuristic_triage(msg)
    try:
        out = await ollama_chat(SLM, [{"role": "system", "content": TRIAGE_PROMPT},
                                      {"role": "user", "content": msg}], fmt="json", max_tokens=80, timeout=15)
        d = json.loads(out)
        res = {
            "disaster": d.get("disaster") if d.get("disaster") in list(KEYWORDS) + ["other", "none"] else base["disaster"],
            "urgency": str(d.get("urgency", "")).upper() if str(d.get("urgency", "")).upper() in
                       ["CRITICAL", "HIGH", "MODERATE", "LOW", "UNKNOWN"] else base["urgency"],
            "language": d.get("language", base["language"]),
        }
        if res["disaster"] in ("none", "other") and base["disaster"] != "none":
            res["disaster"] = base["disaster"]
        if base["urgency"] == "CRITICAL":  # never downgrade a safety keyword hit
            res["urgency"] = "CRITICAL"
        return res
    except Exception:
        return base


# --------------------------------------------------------------------------
# LIVE DATA + RISK SCORE
# --------------------------------------------------------------------------
async def fetch_context(lat: float, lon: float) -> dict:
    ctx = {"weather": None, "quakes": None, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="minutes")}
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
    async with httpx.AsyncClient(timeout=10) as c:
        try:
            r = await c.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 2,
                "current": "temperature_2m,wind_speed_10m,precipitation",
                "daily": "precipitation_sum,wind_gusts_10m_max,temperature_2m_max"})
            j = r.json()
            ctx["weather"] = {
                "temp_c": j["current"]["temperature_2m"],
                "wind_kmh": j["current"]["wind_speed_10m"],
                "rain_now_mm": j["current"]["precipitation"],
                "rain_today_mm": j["daily"]["precipitation_sum"][0],
                "rain_tomorrow_mm": j["daily"]["precipitation_sum"][1],
                "gust_max_kmh": j["daily"]["wind_gusts_10m_max"][0],
                "temp_max_c": j["daily"]["temperature_2m_max"][0],
                "source": "Open-Meteo"}
        except Exception:
            pass
        try:
            r = await c.get("https://earthquake.usgs.gov/fdsnws/event/1/query", params={
                "format": "geojson", "latitude": lat, "longitude": lon, "maxradiuskm": 300,
                "starttime": since, "minmagnitude": 2.5, "orderby": "time", "limit": 5})
            ctx["quakes"] = [{"mag": f["properties"]["mag"], "place": f["properties"]["place"],
                              "time": datetime.fromtimestamp(f["properties"]["time"] / 1000, timezone.utc)
                              .strftime("%Y-%m-%d %H:%M UTC")} for f in r.json()["features"]]
        except Exception:
            pass
    return ctx


def risk_score(ctx: dict):
    """Transparent heuristic. Returns None unless enough data. It is a PREDICTION, not a fact."""
    w, q = ctx.get("weather"), ctx.get("quakes")
    if not w and q is None:
        return None
    pts, factors = 0, []
    if w:
        rain = max(w["rain_today_mm"] or 0, w["rain_tomorrow_mm"] or 0)
        p = min(45, rain / 2)
        pts += p; factors.append(f"Rain forecast {rain:.0f} mm/day (+{p:.0f})")
        g = min(30, max(0, (w["gust_max_kmh"] or 0) - 30) * 0.5)
        pts += g; factors.append(f"Wind gusts {w['gust_max_kmh']:.0f} km/h (+{g:.0f})")
        h = min(15, max(0, (w["temp_max_c"] or 0) - 38) * 3)
        pts += h; factors.append(f"Max temperature {w['temp_max_c']:.0f} °C (+{h:.0f})")
    if q:
        m = max(x["mag"] or 0 for x in q)
        e = min(40, max(0, m - 3) * 10 + len(q))
        pts += e; factors.append(f"{len(q)} quake(s) within 300 km, max M{m} (+{e:.0f})")
    elif q == []:
        factors.append("No M2.5+ quakes within 300 km in 24 h")
    s = int(min(100, round(pts)))
    cat = ("LOW" if s <= 20 else "MODERATE" if s <= 40 else "HIGH" if s <= 60
           else "VERY HIGH" if s <= 80 else "CRITICAL")
    return {"score": s, "category": cat, "factors": factors,
            "note": "Prediction based on public data - not an official warning."}


def context_text(ctx, risk):
    lines = []
    w = ctx.get("weather")
    if w:
        lines.append(f"Weather ({w['source']}, {ctx['fetched_at']} UTC): {w['temp_c']}°C, wind {w['wind_kmh']} km/h, "
                     f"rain today {w['rain_today_mm']} mm, tomorrow {w['rain_tomorrow_mm']} mm, gusts up to {w['gust_max_kmh']} km/h.")
    if ctx.get("quakes") is not None:
        qs = ctx["quakes"]
        lines.append("USGS quakes within 300 km, last 24h: " +
                     ("; ".join(f"M{x['mag']} {x['place']} {x['time']}" for x in qs) if qs else "none"))
    if risk:
        lines.append(f"Computed risk (PREDICTION): {risk['score']}/100 {risk['category']}.")
    lines.append("No official alert/evacuation/shelter feed is connected - do not claim any exists.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# RESPONSE BUILDING
# --------------------------------------------------------------------------
ORDER = {"NOW": 0, "NEXT 6 HOURS": 1, "TODAY": 2}


def build_action_plan(disaster, ctx, risk, profile):
    """Turn data into a prioritised TODO list: what to do, by when, and WHY (trigger = data)."""
    w = ctx.get("weather") or {}
    rain = max(w.get("rain_today_mm") or 0, w.get("rain_tomorrow_mm") or 0)
    gust, tmax = w.get("gust_max_kmh") or 0, w.get("temp_max_c") or 0
    mag = max([x["mag"] or 0 for x in (ctx.get("quakes") or [])], default=0)
    A = []
    if rain >= 100 or disaster in ("flood", "landslide"):
        why = f"Heavy rain forecast ({rain:.0f} mm/day)" if rain else "Flood/landslide reported"
        A += [("NOW", "Charge phone + power bank; download an offline map of your area", why + " can cut power and networks"),
              ("NOW", "Identify your nearest higher-ground / upper-floor route and agree a meeting point", why),
              ("NEXT 6 HOURS", "Pack go-bag: water, food, torch, medicines, documents in a plastic bag", why),
              ("NEXT 6 HOURS", "Do not cross flooded roads; avoid slopes, river banks and underpasses", "Moving water 15 cm deep can knock a person down")]
        if rain >= 200:
            A.append(("NOW", "Be ready to evacuate immediately when authorities say so", "Extreme rainfall (200+ mm/day)"))
    if gust >= 70 or disaster == "cyclone":
        A += [("NOW", "Secure or bring in loose objects; stay away from coast and windows", f"Gusts up to {gust:.0f} km/h"),
              ("TODAY", "Stock 3 days of water/food; follow official evacuation orders only", "Cyclones bring surge, wind and power cuts")]
    if tmax >= 40 or disaster == "heatwave":
        A += [("NOW", "Drink water/ORS regularly and avoid going out 12-4 pm", f"Max temperature {tmax:.0f} °C"),
              ("TODAY", "Check on elderly, children and neighbours; know heatstroke signs (confusion, no sweating) - call 108/112", "Heatstroke can be fatal in hours")]
    if mag >= 4.5 or disaster == "earthquake":
        A += [("NOW", "Pick a safe spot in each room (under sturdy table, away from glass) - Drop, Cover, Hold On", f"Quake M{mag} nearby: aftershocks likely"),
              ("NEXT 6 HOURS", "Know how to shut off gas and electricity; keep shoes and torch by the bed", "Gas leaks and fires follow quakes")]
    P = {"ground_floor": ("NOW", "Lift documents, electronics and valuables above 1 m / to an upper floor", "Ground-floor homes flood first"),
         "elderly": ("NOW", "Arrange transport/help for elderly family early - do not wait for orders", "Slower evacuation needs a head start"),
         "child": ("NEXT 6 HOURS", "Pack child essentials: milk/food, medicines, ID tag with phone number", "Children need dedicated supplies"),
         "vehicle": ("NOW", "Park on high ground and keep fuel above half", "Fuel stations close in disasters"),
         "medical": ("NEXT 6 HOURS", "Pack 7 days of regular medicines + prescriptions; keep phone numbers of your doctor", "Pharmacies may be unreachable")}
    if A:
        A += [P[k] for k in profile if k in P]
    A.append(("TODAY", "Save emergency numbers offline and tell family where you will go", "Communication is the first thing to fail"))
    if not (rain >= 100 or gust >= 70 or tmax >= 40 or mag >= 4.5 or disaster not in ("none", "other")):
        A = [("TODAY", "No major hazard in current data. Keep go-bag ready and numbers saved", "Preparedness costs nothing")]
    seen, out = set(), []
    for when, act, why in sorted(A, key=lambda x: ORDER[x[0]]):
        if act not in seen:
            seen.add(act); out.append({"when": when, "action": act, "why": why})
    return out


def playbook_response(t: dict, loc: str, lang: str, status: str) -> str:
    pb = PLAYBOOKS.get(t["disaster"])
    hi = lang == "hi"
    if not pb:
        return ("🚨 ALERT LEVEL:\nUnknown\n\n📍 LOCATION:\n" + loc +
                "\n\n🛡️ WHAT TO DO NOW:\n- If anyone is in danger, call 112 now.\n- Move to a safe open place away from hazards.\n"
                "- Tell me what is happening (flood, fire, earthquake, storm...).\n\n📡 INFORMATION STATUS:\nUnknown")
    p = pb["hi" if hi else "en"]
    level = t["urgency"].capitalize()
    return (f"🚨 ALERT LEVEL:\n{level}\n\n📍 LOCATION:\n{loc}\n\n⚠️ WHAT IS HAPPENING:\n{p['happening']}\n\n"
            f"🛡️ WHAT TO DO NOW:\n" + "\n".join(f"- {a}" for a in p["do"]) +
            f"\n- {'जान को खतरा हो तो तुरंत 112 पर कॉल करें।' if hi else 'If life is in danger, call 112 now.'}"
            f"\n\n🏠 SAFEST OPTION:\n{p['safest']}\n\n📌 IMPORTANT RISKS:\n" + "\n".join(f"- {r}" for r in p["risks"]) +
            f"\n\n📡 INFORMATION STATUS:\n{status}")


app = FastAPI(title="ResQora - AI Disaster Shield")


class Chat(BaseModel):
    message: str
    lat: float | None = None
    lon: float | None = None
    lang: str = "auto"
    history: list[dict] = []
    profile: list[str] = []  # ground_floor, elderly, child, vehicle, medical


@app.post("/api/chat")
async def chat(req: Chat):
    t = await triage(req.message)
    lang = t["language"] if req.lang == "auto" else req.lang
    have_loc = req.lat is not None and req.lon is not None
    ctx = await fetch_context(req.lat, req.lon) if have_loc else {"weather": None, "quakes": None, "fetched_at": ""}
    risk = risk_score(ctx) if have_loc else None
    loc = f"{req.lat:.4f}, {req.lon:.4f} (from your device)" if have_loc else "Location unavailable"
    status = "Partially verified" if (ctx["weather"] or ctx["quakes"] is not None) else "Unknown"

    user_block = (f"USER MESSAGE: {req.message}\nTRIAGE: {t['disaster']} / {t['urgency']}\nLOCATION: {loc}\n"
                  f"LANGUAGE: {'reply in ' + lang if req.lang != 'auto' else 'reply in the language of the user message'}\n"
                  f"VERIFIED DATA:\n{context_text(ctx, risk)}")
    if t["urgency"] == "CRITICAL":
        user_block += "\nLife-threatening: keep reply VERY SHORT, max 5 bullets."
    model = SLM if t["urgency"] == "CRITICAL" else LLM  # speed for critical, depth otherwise
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + \
           [h for h in req.history[-6:] if h.get("role") in ("user", "assistant")] + \
           [{"role": "user", "content": user_block}]
    try:
        text = await ollama_chat(model, msgs, max_tokens=300 if t["urgency"] == "CRITICAL" else 600)
        engine = model
    except Exception:
        text, engine = playbook_response(t, loc, lang, status), "offline-playbook"
    plan = build_action_plan(t["disaster"], ctx, risk, req.profile)
    return {"text": text, "triage": t, "risk": risk, "engine": engine, "context": ctx, "action_plan": plan,
            "emergency_numbers": EMERGENCY_NUMBERS}


@app.get("/api/context")
async def context(lat: float, lon: float):
    ctx = await fetch_context(lat, lon)
    risk = risk_score(ctx)
    return {"context": ctx, "risk": risk, "action_plan": build_action_plan("none", ctx, risk, [])}


@app.get("/api/playbooks")
async def playbooks():
    return {"playbooks": PLAYBOOKS, "numbers": EMERGENCY_NUMBERS}


@app.get("/api/status")
async def status():
    out = {"ollama": False, "slm": SLM, "llm": LLM, "installed": []}
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{OLLAMA}/api/tags")
            out["installed"] = [m["name"] for m in r.json().get("models", [])]
            out["ollama"] = True
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------
# FRONTEND (single page, mobile-first, no external assets)
# --------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ResQora - AI Disaster Shield</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--tx:#14181f;--mut:#5b6573;--bd:#e1e5ea;--red:#d6252e;--ok:#1a7f4b}
@media(prefers-color-scheme:dark){:root{--bg:#0f1318;--card:#171d25;--tx:#eef1f5;--mut:#9aa5b4;--bd:#2a3340}}
*{box-sizing:border-box}body{margin:0;font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--tx)}
header{display:flex;gap:10px;align-items:center;justify-content:space-between;padding:12px 16px;background:var(--red);color:#fff;position:sticky;top:0;z-index:5}
header h1{font-size:18px;margin:0}header select{border-radius:8px;border:0;padding:6px}
main{max-width:1000px;margin:0 auto;padding:12px;display:grid;gap:12px;grid-template-columns:1fr}
@media(min-width:860px){main{grid-template-columns:1.5fr 1fr}}
.card{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:14px}
.card h2{font-size:15px;margin:0 0 8px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
button{font:inherit;cursor:pointer;border-radius:10px;border:1px solid var(--bd);background:var(--card);color:var(--tx);padding:8px 12px}
button.p{background:var(--red);border-color:var(--red);color:#fff}
#sos{width:100%;font-size:20px;font-weight:700;padding:16px;background:var(--red);color:#fff;border:0;border-radius:14px}
#log{height:46vh;min-height:280px;overflow:auto;display:flex;flex-direction:column;gap:8px;padding:4px}
.m{padding:10px 12px;border-radius:12px;max-width:94%;white-space:pre-wrap;word-wrap:break-word}
.u{align-self:flex-end;background:var(--red);color:#fff}.a{align-self:flex-start;background:var(--bg);border:1px solid var(--bd)}
.tag{font-size:12px;color:var(--mut);margin-top:6px}
.row{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap}textarea{flex:1;min-width:0;font:inherit;border-radius:10px;border:1px solid var(--bd);padding:10px;background:var(--bg);color:var(--tx);resize:none}
.chips button{font-size:14px;padding:6px 10px}
.badge{display:inline-block;padding:2px 10px;border-radius:99px;color:#fff;font-weight:700;font-size:13px}
.CRITICAL{background:#8b0000}.HIGH{background:#d6252e}.MODERATE{background:#c77a00}.LOW{background:var(--ok)}.UNKNOWN,.VERYHIGH{background:#5b6573}.VERYHIGH{background:#b3171f}
.bar{height:10px;border-radius:6px;background:var(--bd);overflow:hidden;margin:6px 0}.bar i{display:block;height:100%;background:var(--red)}
.num a{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--bd);color:var(--tx);text-decoration:none}
.num a b{color:var(--red)}small,.mut{color:var(--mut)}ul{padding-left:18px;margin:6px 0}label.c{display:flex;gap:8px;padding:3px 0}
.pl{padding:8px;border-left:4px solid var(--red);background:var(--bg);margin:6px 0;border-radius:6px;font-size:14px}.pl b{color:var(--red);font-size:12px}.pl small{display:block}
.tabs{display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap}.tabs button.on{background:var(--tx);color:var(--bg)}
</style></head><body>
<header><h1>🛡️ ResQora <span style="font-weight:400;font-size:13px;opacity:.9">· AI Disaster Shield</span></h1>
<select id="lang"><option value="auto">Auto language</option><option value="en">English</option><option value="hi">हिन्दी</option></select></header>
<main>
<section>
 <div class="card">
  <button id="sos">🆘 SOS — share my location</button>
  <div class="mut" style="margin-top:6px;font-size:13px">Always call <b>112</b> (or your local emergency number) first in life-threatening danger.</div>
 </div>
 <div class="card" style="margin-top:12px">
  <h2>Ask for help</h2>
  <div class="chips row" style="margin:0 0 8px">
   <button data-q="Earthquake shaking right now">🌍 Earthquake</button><button data-q="Flood water is rising around my house">🌊 Flood</button>
   <button data-q="There is a fire in my building">🔥 Fire</button><button data-q="Cyclone warning near me, what should I do?">🌀 Cyclone</button>
   <button data-q="Landslide risk, heavy rain in the hills">⛰️ Landslide</button><button data-q="भीषण गर्मी में क्या करें?">☀️ Heat</button>
  </div>
  <div id="log"><div class="m a">Hi, I'm ResQora, your AI disaster shield. Tell me what is happening and share your location (tap 📍) for live local data. I will never invent alerts or shelter locations.</div></div>
  <div class="row"><textarea id="in" rows="2" placeholder="Describe your situation… (English / हिन्दी)"></textarea></div>
  <div class="row"><button class="p" id="send">Send</button><button id="loc">📍 Location</button><button id="mic">🎤 Voice</button><button id="spk">🔊 Read aloud: off</button></div>
  <div id="loctxt" class="tag"></div>
 </div>
</section>
<aside style="display:grid;gap:12px;align-content:start">
 <div class="card"><h2>Live risk</h2><div id="risk" class="mut">Enable location to see the risk estimate (prediction from public weather + USGS earthquake data).</div><div id="plan"></div></div>
 <div class="card"><h2>Personalise my plan</h2><div id="prof"><label class="c"><input type="checkbox" value="ground_floor">Ground floor home</label><label class="c"><input type="checkbox" value="elderly">Elderly at home</label><label class="c"><input type="checkbox" value="child">Children at home</label><label class="c"><input type="checkbox" value="vehicle">I have a vehicle</label><label class="c"><input type="checkbox" value="medical">Regular medicines</label></div></div>
 <div class="card num"><h2>Emergency numbers</h2><div id="nums"></div><small>Defaults for India — verify for your region.</small></div>
 <div class="card"><h2>Nearby help (opens Maps)</h2><div class="row" style="margin:0">
  <button data-map="hospital">🏥 Hospitals</button><button data-map="relief shelter">⛺ Shelters</button><button data-map="police station">👮 Police</button><button data-map="fire station">🚒 Fire stn</button></div>
  <small>Map results are not verified shelter lists — confirm with local authorities.</small></div>
 <div class="card"><h2>Offline guides</h2><div class="tabs" id="tabs"></div><div id="guide"></div></div>
 <div class="card"><h2>Go-bag checklist</h2><div id="chk"></div></div>
 <div class="card"><h2>System</h2><div id="sys" class="mut">checking…</div></div>
</aside>
</main>
<script>
const $=s=>document.querySelector(s);let pos=null,hist=[],speak=false,PB=null;
const lang=()=>$('#lang').value;
function add(t,c,tag){const d=document.createElement('div');d.className='m '+c;d.textContent=t;
 if(tag){const g=document.createElement('div');g.className='tag';g.textContent=tag;d.appendChild(g)}
 $('#log').appendChild(d);$('#log').scrollTop=1e9;return d}
async function send(txt){txt=(txt||$('#in').value).trim();if(!txt)return;$('#in').value='';add(txt,'u');
 const w=add('Thinking…','a');
 try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({message:txt,lat:pos?.lat??null,lon:pos?.lon??null,lang:lang(),history:hist,profile:getProfile()})});
  const j=await r.json();w.remove();
  add(j.text,'a',`Triage: ${j.triage.disaster} / ${j.triage.urgency} · Engine: ${j.engine}`);
  hist.push({role:'user',content:txt},{role:'assistant',content:j.text});
  if(j.risk)showRisk(j.risk);showPlan(j.action_plan);
  if(speak){const u=new SpeechSynthesisUtterance(j.text.replace(/[🚨📍⚠️🛡️🏠📌📡]/g,''));
   u.lang=(j.triage.language==='hi'||lang()==='hi')?'hi-IN':'en-IN';speechSynthesis.cancel();speechSynthesis.speak(u)}
 }catch(e){w.remove();add('Server unreachable. Use the offline guides and call 112 if in danger.','a')}}
function showRisk(r){const k=r.category.replace(' ','');$('#risk').innerHTML=
 `<span class="badge ${k}">${r.category}</span> <b>${r.score}/100</b><div class="bar"><i style="width:${r.score}%"></i></div>
 <ul>${r.factors.map(f=>`<li>${f}</li>`).join('')}</ul><small>${r.note}</small>`}
const getProfile=()=>[...document.querySelectorAll('#prof input:checked')].map(i=>i.value);
function showPlan(p){$('#plan').innerHTML=p&&p.length?'<h2 style="margin-top:14px">Your action plan</h2>'+p.map(a=>`<div class="pl"><b>${a.when}</b> ${a.action}<small>Why: ${a.why}</small></div>`).join(''):''}
function getLoc(cb){if(!navigator.geolocation){$('#loctxt').textContent='Geolocation not supported';return}
 $('#loctxt').textContent='Getting location…';
 navigator.geolocation.getCurrentPosition(p=>{pos={lat:p.coords.latitude,lon:p.coords.longitude};
  $('#loctxt').textContent=`📍 ${pos.lat.toFixed(4)}, ${pos.lon.toFixed(4)} (±${Math.round(p.coords.accuracy)} m)`;
  fetch(`/api/context?lat=${pos.lat}&lon=${pos.lon}`).then(r=>r.json()).then(j=>{j.risk&&showRisk(j.risk);showPlan(j.action_plan)}).catch(()=>{});
  cb&&cb()},()=>{$('#loctxt').textContent='Location denied/unavailable — you can still describe where you are.'},{enableHighAccuracy:true,timeout:15000})}
$('#loc').onclick=()=>getLoc();$('#send').onclick=()=>send();
$('#in').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
document.querySelectorAll('[data-q]').forEach(b=>b.onclick=()=>send(b.dataset.q));
document.querySelectorAll('[data-map]').forEach(b=>b.onclick=()=>{const q=encodeURIComponent(b.dataset.map);
 const u=pos?`https://www.google.com/maps/search/${q}/@${pos.lat},${pos.lon},14z`:`https://www.google.com/maps/search/${q}+near+me`;window.open(u,'_blank')});
$('#spk').onclick=()=>{speak=!speak;$('#spk').textContent='🔊 Read aloud: '+(speak?'on':'off')};
$('#mic').onclick=()=>{const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
 if(!SR){alert('Voice input not supported in this browser (try Chrome).');return}
 const r=new SR();r.lang=lang()==='hi'?'hi-IN':'en-IN';r.onresult=e=>{$('#in').value=e.results[0][0].transcript;send()};r.start()};
$('#sos').onclick=()=>{const go=()=>{const link=pos?`https://maps.google.com/?q=${pos.lat},${pos.lon}`:'(location unavailable)';
 const msg=`🆘 SOS! I need help. My location: ${link} — Sent ${new Date().toLocaleString()}`;
 if(navigator.share){navigator.share({text:msg}).catch(()=>{})}
 else{window.open('https://wa.me/?text='+encodeURIComponent(msg),'_blank')}
 add('SOS message prepared. Also CALL 112 now. Move to a safe place and keep your phone charged.','a')};
 pos?go():getLoc(go);if(!pos)setTimeout(()=>{if(!pos)go()},16000)};
// numbers, guides, checklist, status
fetch('/api/playbooks').then(r=>r.json()).then(j=>{PB=j.playbooks;
 $('#nums').innerHTML=j.numbers.map(n=>`<a href="tel:${n.number}"><span>${n.name}</span><b>${n.number}</b></a>`).join('');
 localStorage.setItem('pb',JSON.stringify(j));drawGuides(Object.keys(PB)[0])}).catch(()=>{
 const c=localStorage.getItem('pb');if(c){const j=JSON.parse(c);PB=j.playbooks;
 $('#nums').innerHTML=j.numbers.map(n=>`<a href="tel:${n.number}"><span>${n.name}</span><b>${n.number}</b></a>`).join('');drawGuides(Object.keys(PB)[0])}});
function drawGuides(k){$('#tabs').innerHTML=Object.keys(PB).map(x=>`<button class="${x===k?'on':''}" onclick="drawGuides('${x}')">${x}</button>`).join('');
 const p=PB[k][lang()==='hi'?'hi':'en'];$('#guide').innerHTML=`<b>${p.happening}</b><ul>${p.do.map(a=>`<li>${a}</li>`).join('')}</ul><small><b>Risks:</b> ${p.risks.join(', ')}</small>`}
$('#lang').onchange=()=>{const on=document.querySelector('#tabs .on');PB&&drawGuides(on?on.textContent:Object.keys(PB)[0])};
const items=['Water (3 L/person/day)','Dry food & snacks','Torch + spare batteries','Power bank + charger','First-aid kit & medicines','ID / documents (copies)','Cash','Whistle','Warm clothes / raincoat','Radio'];
const saved=JSON.parse(localStorage.getItem('chk')||'{}');
$('#chk').innerHTML=items.map((t,i)=>`<label class="c"><input type="checkbox" data-i="${i}" ${saved[i]?'checked':''}>${t}</label>`).join('');
$('#chk').onchange=e=>{saved[e.target.dataset.i]=e.target.checked;localStorage.setItem('chk',JSON.stringify(saved))};
fetch('/api/status').then(r=>r.json()).then(j=>{const ok=m=>j.installed.some(x=>x.startsWith(m.split(':')[0])&&x.includes(m.split(':')[1]||''));
 $('#sys').innerHTML=j.ollama?`Ollama online<br>SLM ${j.slm}: ${ok(j.slm)?'✅':'❌ not pulled'}<br>LLM ${j.llm}: ${ok(j.llm)?'✅':'❌ not pulled'}`:'Ollama offline — using built-in offline playbooks.'}).catch(()=>$('#sys').textContent='Server unreachable');
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
