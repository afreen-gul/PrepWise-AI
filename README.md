# PrepWise AI — Intelligent ML Data Preparation Platform

PrepWise AI turns raw, messy datasets into clean, reliable, machine-learning-ready
data — without manually performing every preprocessing step. Upload a CSV, and
PrepWise analyzes it, identifies data-quality issues, recommends useful
transformations, and guides you through the full preparation workflow until you
can export a dataset that is ready to train a model.

---

## What It Does

PrepWise walks a dataset through a complete, cumulative data-preparation pipeline:

1. **Upload** — Add your raw CSV dataset.
2. **Profile** — Understand columns, distributions, and relationships.
3. **Data Quality** — Detect missing values, duplicates, outliers, and other issues.
4. **Cleaning** — Apply appropriate data-cleaning and preprocessing strategies.
5. **Feature Engineering** — Review and generate useful feature transformations
   (numerical, categorical, datetime, and text).
6. **Feature Selection** — Identify the most useful features for modeling
   (redundancy, correlation, mutual information, and VIF analysis).
7. **Export** — Download the final, clean, ML-ready dataset.

Each stage consumes the validated output of the previous stage, so the workflow
behaves as one coherent, end-to-end pipeline.

---

## Project Overview

The platform is split into two independently runnable services:

- **Backend** — a FastAPI service that handles uploads, file storage, CSV
  parsing, profiling, quality analysis, cleaning, feature engineering, feature
  selection, and metadata persistence in SQLite via SQLAlchemy.
- **Frontend** — a Streamlit app that provides a clean, navy/white UI for moving
  a dataset through each stage of the preparation workflow by calling the
  backend API.

---

## Folder Structure

```
PrepWise-AI/
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
│   │   ├── services/           # Business logic (profiling, cleaning, FE, FS)
│   │   ├── utils/              # Reusable helpers (file handling)
│   │   └── main.py             # FastAPI app entry point
│   ├── uploads/                # Saved uploaded datasets (git-ignored)
│   ├── processed/              # Cleaned / exported datasets (git-ignored)
│   ├── tests/                  # Backend test suite
│   └── requirements.txt
├── frontend/
│   ├── ui/                     # Theme, shell, and per-stage dashboards
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

The SQLite database and the `uploads/` / `processed/` directories are created
automatically on first run.

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

## Testing

Run the backend test suite from the project root:

```bash
pytest backend/tests
```

---

## Typical Workflow

1. Start the backend and frontend as described above.
2. Open the Streamlit UI in your browser.
3. Upload a `.csv` file — PrepWise profiles it automatically.
4. Review data-quality issues, then apply cleaning.
5. Review feature-engineering opportunities and feature selection.
6. Export the final, ML-ready dataset for training.
