# AutoPrep AI — Intelligent Data Preparation Platform

AutoPrep AI lets you upload any messy dataset and automatically prepare it for
machine learning. This repository currently contains **Phase 1** — the project
foundation.

> **Phase 1 scope:** upload a CSV, store it, parse it with pandas, record its
> metadata in SQLite, and display an overview (row/column counts, column names,
> and the first 10 rows). No preprocessing, EDA, feature engineering,
> visualization, or AI is implemented yet.

---

## Project Overview

The platform is split into two independently runnable services:

- **Backend** — a FastAPI service that handles uploads, file storage, CSV
  parsing, and metadata persistence in SQLite via SQLAlchemy.
- **Frontend** — a Streamlit app that provides a clean UI for uploading a CSV
  and viewing its overview by calling the backend API.

---

## Folder Structure

```
AutoPrep-AI/
├── backend/
│   ├── app/
│   │   ├── api/                # HTTP routes (thin controllers)
│   │   │   └── datasets.py
│   │   ├── core/               # Configuration / settings
│   │   │   └── config.py
│   │   ├── database/           # Engine, session, declarative base
│   │   │   └── session.py
│   │   ├── models/             # SQLAlchemy ORM models
│   │   │   └── dataset.py
│   │   ├── schemas/            # Pydantic request/response schemas
│   │   │   └── dataset.py
│   │   ├── services/           # Business logic
│   │   │   └── dataset_service.py
│   │   ├── utils/              # Reusable helpers (file handling)
│   │   │   └── file_utils.py
│   │   └── main.py             # FastAPI app entry point
│   ├── uploads/                # Saved uploaded datasets (git-ignored)
│   └── requirements.txt
├── frontend/
│   ├── pages/                  # Reserved for future multi-page UI
│   ├── assets/                 # Reserved for static assets
│   └── app.py                  # Streamlit entry point
├── README.md
└── .gitignore
```

---

## Installation

Requires **Python 3.12+**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies (covers both backend and frontend)
pip install -r backend/requirements.txt
```

---

## Running the Backend

From the `backend/` directory:

```bash
cd backend
uvicorn app.main:app --reload
```

- API base URL: `http://127.0.0.1:8000`
- Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/`

The SQLite database (`backend/autoprep.db`) and the `uploads/` directory are
created automatically on first run.

---

## Running the Frontend

In a **second terminal** (with the backend running):

```bash
cd frontend
streamlit run app.py
```

Streamlit opens at `http://localhost:8501`. If your backend runs elsewhere, set
the `AUTOPREP_BACKEND_URL` environment variable before launching.

---

## Testing the Application

1. Start the backend and frontend as described above.
2. Open the Streamlit UI in your browser.
3. Upload a `.csv` file.
4. Confirm the overview shows the correct row count, column count, column
   names, and the first 10 rows.
5. (Optional) Verify persistence via the API: `GET /api/v1/datasets` returns
   the recorded metadata, and the file appears under `backend/uploads/`.
