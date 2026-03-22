# Healthcare Planning Assistant

Full-stack demo: Flask + SQLite auth, **LangChain** agent (**Groq**), **FAISS** + HuggingFace embeddings, **pytesseract** OCR, and a vanilla HTML/CSS/JS frontend.

## Prerequisites

- **Python 3.10+** (use `py` on Windows if `python` is not on PATH)
- **Tesseract OCR** installed and on your `PATH` (required for `/verify-id`)
  - Windows: install from [GitHub UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add the install folder to PATH, or set `TESSERACT_CMD` in `.env` to the full path of `tesseract.exe`.
- **Groq API key** (optional but recommended): [Groq Console](https://console.groq.com/) — without it, the planner and symptom endpoints use **demo fallbacks**.

## Setup

1. Clone or copy this project and open a terminal in the project root (`Healtcare-Agentic Ai`).

2. Create a virtual environment (recommended):

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies (first run may take several minutes while downloading `torch` / `sentence-transformers`):

   ```powershell
   py -m pip install -r requirements.txt
   ```

4. Copy environment variables:

   ```powershell
   copy .env.example .env
   ```

   Edit `.env` and set `GROQ_API_KEY=your_key_here`.

## Run the server

From the project root:

```powershell
cd backend
py app.py
```

Open **http://127.0.0.1:5000** in your browser.

- **Home:** `/`
- **Login / Signup:** `/login.html` (username + password + role: Doctor or Patient)
- **Dashboard:** `/dashboard.html`
- **Doctor:** `/doctor.html` → ID verification `/verify.html`, Planner `/planner.html`
- **Patient:** `/patient.html`

## API routes (summary)

| Method | Path | Notes |
|--------|------|--------|
| POST | `/signup` | JSON: `username`, `password`, `role` (`doctor` \| `patient`) |
| POST | `/login` | JSON: `username`, `password` |
| POST | `/logout` | Clears session |
| GET | `/session` | Current session |
| POST | `/upload-id` | `multipart/form-data` file `file` — **doctor** only |
| POST | `/verify-id` | JSON optional `path` — **doctor** only |
| POST | `/planner-agent` | JSON `{ "goal": "..." }` — **doctor** only |
| POST | `/symptom-check` | JSON `{ "symptoms": "..." }` — **patient** only |

Additional helpers: `GET /doctor/appointments`, `GET /doctor/patient-queries`, `GET /doctor/profile` (mock + DB profile).

## Project layout

```
/backend
  app.py
  ocr_service.py
  vector_db.py
  planner_agent.py
  routes/
  services/
  models/
  utils/
  data/
  uploads/
/frontend
  index.html, login.html, dashboard.html, verify.html, planner.html, doctor.html, patient.html
  css/
  js/
requirements.txt
```

## Data and indexes

- **SQLite** file: `backend/data/healthcare.db` (created automatically).
- **Seed knowledge:** `backend/data/medical_knowledge.json` — edit and delete `backend/data/faiss_index/` to rebuild the FAISS index on next startup (or call `add_documents()` from code).
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (downloaded on first use).

## Notes

- This is an **educational** demo; it is **not** medical advice or a substitute for licensed care.
- No web scraping, no FaceID or social login — local username/password only.
