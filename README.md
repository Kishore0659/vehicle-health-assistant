# 🚗 AI Vehicle Health Assistant

> A production-grade RAG application that diagnoses vehicle issues for Bikes, Cars, and SUVs using **FAISS vector search**, **sentence-transformers embeddings**, and **Claude LLM** via FastAPI.

---

## 📁 Project Structure

```
ai-vehicle-health-assistant/
├── backend/
│   ├── api/
│   │   └── main.py          ← FastAPI app + all endpoints
│   ├── rag/
│   │   └── rag_pipeline.py  ← Full RAG pipeline (embed → retrieve → generate)
│   ├── db/
│   │   └── models.py        ← SQLAlchemy ORM (SQLite)
│   └── data/
│       ├── bike_issues.txt  ← 15 curated bike issue records
│       ├── car_issues.txt   ← 18 curated car issue records
│       └── suv_issues.txt   ← 18 curated SUV issue records
├── frontend/
│   └── index.html           ← Standalone UI (no framework needed)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.10+
- An [Anthropic API Key](https://console.anthropic.com/) (free tier works)

### 2. Create a virtual environment
```bash
cd ai-vehicle-health-assistant
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> The first run will download the `all-MiniLM-L6-v2` embedding model (~90 MB) from HuggingFace.

### 4. Set your Anthropic API key
```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows PowerShell
$env:ANTHROPIC_API_KEY="sk-ant-..."

# Windows CMD
set ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Run the backend
```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Open the frontend
Open `frontend/index.html` in your browser:
```bash
# macOS
open frontend/index.html

# Linux
xdg-open frontend/index.html

# Windows
start frontend/index.html
```

> The frontend talks to `http://localhost:8000` by default.

---

## 🔌 API Reference

### `POST /ask`
Diagnose a vehicle issue using RAG.

**Request:**
```json
{
  "query": "My bike engine makes a knocking noise when idling",
  "vehicle_type": "bike"
}
```

**Response:**
```json
{
  "vehicle_type": "Bike",
  "query": "My bike engine makes a knocking noise when idling",
  "problem_summary": "Engine knocking likely indicates...",
  "possible_causes": ["Low engine oil", "Worn piston", "Carbon buildup"],
  "severity": "High",
  "suggested_actions": ["Check oil level immediately", "Visit a mechanic"],
  "estimated_cost_inr": "INR 1000 – INR 15000",
  "additional_notes": "Do not ride the bike until inspected.",
  "retrieved_records_count": 4
}
```

`vehicle_type` must be one of: `bike`, `car`, `suv`

### `POST /vehicles`
Save a vehicle profile.

### `GET /vehicles`
List all saved vehicle profiles.

### `GET /history?limit=20`
View recent query history.

### `DELETE /history`
Clear all query history.

### `GET /docs`
Interactive Swagger UI — `http://localhost:8000/docs`

---

## 🧠 How the RAG Pipeline Works

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  1. EMBED (all-MiniLM-L6-v2)│  ← sentence-transformers
│     Query → 384-dim vector  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  2. RETRIEVE (FAISS)        │  ← cosine similarity search
│     Top-4 matching records  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  3. GENERATE (Claude)       │  ← Anthropic claude-sonnet-4
│     Structured JSON output  │
└─────────────────────────────┘
```

### Embedding model
`sentence-transformers/all-MiniLM-L6-v2` — lightweight 384-dimensional model that runs fully offline after the first download.

### Vector store
**FAISS IndexFlatIP** — exact inner-product search. Because embeddings are L2-normalised, this is equivalent to cosine similarity. Indexes are built at startup and cached in memory per vehicle type.

### LLM
Claude (`claude-sonnet-4-20250514`) via the Anthropic Python SDK. A structured system prompt forces JSON output with these keys: `problem_summary`, `possible_causes`, `severity`, `suggested_actions`, `estimated_cost_inr`, `additional_notes`.

---

## 🚀 Deployment

### Docker (recommended)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV ANTHROPIC_API_KEY=""
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
```bash
docker build -t vehicle-health-assistant .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... vehicle-health-assistant
```

### Cloud platforms
| Platform   | Command |
|------------|---------|
| Railway    | Connect GitHub repo → set `ANTHROPIC_API_KEY` env var → deploy |
| Render     | Web service → `uvicorn backend.api.main:app --host 0.0.0.0 --port $PORT` |
| Fly.io     | `fly launch` → `fly secrets set ANTHROPIC_API_KEY=sk-ant-...` |
| AWS EC2    | Install Python, clone repo, run with `gunicorn` behind nginx |

### Frontend hosting
The `frontend/index.html` is a single self-contained file. Host it on:
- **Netlify / Vercel**: Drag and drop the `frontend/` folder.
- **GitHub Pages**: Push to a `gh-pages` branch.
- Update the `API_BASE` constant in the JS to point to your deployed backend URL.

---

## ✨ Features

| Feature | Status |
|---------|--------|
| RAG pipeline (FAISS + sentence-transformers + Claude) | ✅ |
| Vehicle type selection (Bike / Car / SUV) | ✅ |
| Structured diagnosis (causes, severity, cost in INR) | ✅ |
| Voice input (Web Speech API, works in Chrome) | ✅ |
| Query history (in-memory + SQLite) | ✅ |
| Vehicle profile storage | ✅ |
| Swagger API docs | ✅ |
| Dark theme, production-grade UI | ✅ |

---

## 🔧 Extending the Dataset

Each dataset file (`backend/data/*.txt`) follows this format:
```
Problem: <description>
Cause: <cause description>
Solution: <solution>
Severity: Low | Moderate | High
Cost: INR X to INR Y

Problem: ...
```

Blank lines separate records. Add as many records as needed — the FAISS index rebuilds automatically on next startup.

---

## 📄 License
MIT — free to use, modify, and deploy.
