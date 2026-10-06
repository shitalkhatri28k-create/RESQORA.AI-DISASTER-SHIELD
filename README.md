# 🛡️ ResQora – AI Disaster Shield

> **From alerts to actions.** Most disaster apps tell you *what is happening*. ResQora tells you **what to do, by when, and why** – personalised to your location, your household and the live data around you.

Open-source • Runs locally with open-weight models • Works offline • English + हिन्दी

---

## ✨ What makes it different

| Pillar | What ResQora does |
|---|---|
| 🧠 **AI risk assessment** | Combines live rainfall, wind, heat and earthquake data into a transparent 0–100 risk score with the exact factors shown (no black box). Marked clearly as a *prediction*. |
| 📍 **Location-aware guidance** | Uses your device GPS to fetch local conditions and tailor advice. |
| 📡 **Real-time disaster information** | Open-Meteo (weather/rain/wind) + USGS (earthquakes within 300 km, last 24 h). No API keys needed. |
| 🆘 **Emergency assistance** | One-tap SOS (shares your Google Maps link), tap-to-call emergency numbers, nearby hospital/shelter/police/fire search, voice input & read-aloud. |
| ✅ **Actionable by design** | A **prioritised action plan** (`NOW` / `NEXT 6 HOURS` / `TODAY`) where every step carries the data trigger that caused it. Personalised for ground-floor homes, elderly, children, vehicles and medical needs. |

## 🧩 Architecture

```
 Browser (single page, mobile-first)
   │  GPS · voice · SOS · checklist · offline guides (cached)
   ▼
 FastAPI  ──────────────►  Open-Meteo  (weather)          [live, no key]
   │      ──────────────►  USGS        (earthquakes)       [live, no key]
   │
   ├─ 1. SLM  (llama3.2:3b)  → triage: disaster type, urgency, language (JSON)
   ├─ 2. Data fetch          → verified context block
   ├─ 3. Risk engine         → 0–100 score + factors (prediction)
   ├─ 4. Action-plan engine  → trigger → action → why, ordered by urgency
   └─ 5. LLM  (llama3.1:8b)  → detailed, empathetic guidance in the user's language
        (CRITICAL cases use the fast SLM for a short reply;
         if Ollama is down, built-in offline playbooks answer instead)
```

**Safety principles** (also encoded in the system prompt): protect life first · never invent alerts, evacuation orders, shelters or phone numbers · separate verified data from predictions · simple language · say "unknown" when unknown.

## 🚀 Quick start

```bash
# 1. Install Ollama (https://ollama.com) and pull open-weight models
ollama pull llama3.2:3b      # SLM – fast triage
ollama pull llama3.1:8b      # LLM – detailed guidance

# 2. Install dependencies
pip install fastapi uvicorn httpx

# 3. Run
python disaster_assistant.py
# open http://localhost:8000
```

Optional environment variables: `OLLAMA_URL`, `SLM_MODEL`, `LLM_MODEL`, `PORT`.
Any Ollama model works (e.g. `qwen2.5:7b`, `gemma2:9b`, `mistral`).

> Location, microphone and speech features need **HTTPS** or `localhost`.

## 🔌 API

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` | `{message, lat, lon, lang, history, profile}` → guidance text, triage, risk, **action_plan**, engine used |
| `GET /api/context?lat=&lon=` | Live data + risk + action plan for a location |
| `GET /api/playbooks` | Offline guides + emergency numbers |
| `GET /api/status` | Ollama and model availability |

## 📐 Risk score (transparent heuristic)

| Factor | Points |
|---|---|
| Rain forecast (max of today/tomorrow) | `min(45, mm / 2)` |
| Wind gusts | `min(30, (km/h − 30) × 0.5)` |
| Heat | `min(15, (max °C − 38) × 3)` when above 38 °C |
| Earthquakes (300 km, 24 h) | `min(40, (max magnitude − 3) × 10 + count)` |

Bands: 0–20 LOW · 21–40 MODERATE · 41–60 HIGH · 61–80 VERY HIGH · 81–100 CRITICAL.
The score only appears when real data was fetched, and is always labelled a prediction.

## ⚠️ Limitations & responsible use

- **Not a replacement for official services.** Call your local emergency number first in any life-threatening situation.
- Emergency numbers are **India defaults** – edit `EMERGENCY_NUMBERS` for your region.
- No official alert/evacuation/shelter feed is connected, and the assistant will not claim one exists. Map buttons are plain searches, not verified shelter lists.
- The risk formula is a hackathon-grade heuristic, not a calibrated scientific model.
- Small local models can make mistakes; the system prompt, safety keyword override and offline playbooks reduce – not remove – that risk.

## 🗺️ Roadmap

- Official feeds: IMD, NDMA SACHET (CAP alerts), GDACS, NASA FIRMS (wildfire)
- Flood-river gauge data and elevation (DEM) for sharper flood risk
- SMS / WhatsApp low-bandwidth mode and PWA install
- Community reports with verification, family check-in ("I'm safe") links
- More regional languages (Marathi, Tamil, Bengali, Telugu…)

## 🤝 Contributing

Hacktoberfest-friendly! Good first issues: add a language to `PLAYBOOKS`, add a new hazard rule to `build_action_plan`, improve the UI. Fork → branch → PR.

## 📄 License

MIT – free to use, modify and share. Built with FastAPI, Ollama, Open-Meteo and USGS open data.
