# Interview Trainer Agent 🎯

An AI-powered interview preparation assistant built on **IBM watsonx.ai** and **IBM Granite** foundation models.  
It uses Retrieval-Augmented Generation (RAG) to produce a fully tailored interview prep kit — technical questions, behavioral STAR-format questions, model answers, and improvement tips — from a curated local knowledge base.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                    │
│  [Name] [Role] [Experience] [Resume/JD PDF upload]              │
└────────────────────────┬────────────────────────────────────────┘
                         │ user profile + optional PDF
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     agent.py  (Orchestrator)                    │
│  1. resume_parser.py  → extract & summarize PDF text            │
│  2. retriever.py      → query ChromaDB for top-k chunks         │
│  3. prompt_templates.py → build structured Granite prompt       │
│  4. watsonx_client.py → call IBM Granite via ModelInference     │
│  5. parse JSON response → return structured prep kit dict       │
└──────┬────────────────────────────┬───────────────────────────┘
       │                            │
       ▼                            ▼
┌─────────────┐          ┌──────────────────────────┐
│  ChromaDB   │          │  IBM watsonx.ai (Granite) │
│  (local DB) │          │  ibm/granite-3-3-8b-instruct│
│  ./db/      │          │  ibm/slate-125m-english-  │
│             │          │  rtrvr (embeddings)       │
└─────────────┘          └──────────────────────────┘
       ▲
       │  (one-time setup)
┌─────────────┐
│  ingest.py  │
│  data/sample_kb/*.txt │
│  → chunk → embed → persist │
└─────────────┘
```

---

## IBM Cloud Lite Setup (Free Tier Only)

1. **Create an IBM Cloud account** at https://cloud.ibm.com (free, no credit card required for Lite tier).
2. **Provision Watson Machine Learning (Lite)**:
   - Go to **Catalog → AI / Machine Learning → Watson Machine Learning**.
   - Select the **Lite** plan → click **Create**.
3. **Create a watsonx.ai Project**:
   - Go to https://dataplatform.cloud.ibm.com → **Projects → New project → Create an empty project**.
   - Copy the **Project ID** from the project settings page.
4. **Generate an API Key**:
   - Go to **Manage → Access (IAM) → API keys → Create an IBM Cloud API key**.
   - Copy and save the key (shown only once).
5. **Find your regional URL**:
   - Default: `https://us-south.ml.cloud.ibm.com`
   - Adjust if your WML instance is in a different region (e.g., `eu-de`, `eu-gb`).

---

## Prerequisites

- Python 3.11+
- `pip` (or a virtual environment manager)
- IBM Cloud Lite account (free) with Watson Machine Learning provisioned

---

## Setup & Run

### 1. Clone / unzip the project

```bash
cd Interview_Agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your IBM Cloud credentials
```

### 5. Build the vector store (one-time)

```bash
python src/ingest.py
```

This reads all `.txt` files in `data/sample_kb/`, chunks them, embeds them using the IBM Slate model, and persists the Chroma vector store to `./db`.

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## How to Extend

| Goal | What to change |
|---|---|
| **Swap ChromaDB for watsonx.data** | Replace `Chroma` in `src/ingest.py` and `src/retriever.py` with the watsonx.data connector |
| **Add more knowledge base content** | Drop additional `.txt` files into `data/sample_kb/` and re-run `python src/ingest.py` |
| **Support more roles** | Add role-specific `.txt` files to `data/sample_kb/` with targeted Q&A content |
| **Deploy to Streamlit Cloud** | Push to GitHub; add `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL` as Streamlit secrets |
| **Add user authentication** | Wrap `app.py` with `streamlit-authenticator` or an OAuth provider |
| **Use a larger Granite model** | Change `GRANITE_MODEL_ID` in `.env` or `watsonx_client.py` to `ibm/granite-13b-instruct-v2` |
| **PDF output** | Replace the Markdown download with `fpdf2` or `reportlab` for a proper PDF export |
| **Persist sessions** | Store generated prep kits in SQLite keyed by session ID |

---

## Project Structure

```
Interview_Agent/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   └── sample_kb/
│       ├── software_engineer.txt
│       ├── data_analyst.txt
│       ├── hr_behavioral.txt
│       └── general_interview_tips.txt
├── src/
│   ├── __init__.py
│   ├── watsonx_client.py     # IBM watsonx.ai credential setup
│   ├── ingest.py             # KB chunking, embedding, Chroma persistence
│   ├── retriever.py          # Chroma query wrapper
│   ├── resume_parser.py      # PDF text extraction + LLM summarization
│   ├── prompt_templates.py   # Structured Granite prompt with few-shot example
│   └── agent.py              # End-to-end orchestration + JSON parsing
├── app.py                    # Streamlit frontend
└── tests/
    ├── __init__.py
    └── test_ingest.py
```

---

## License

MIT — free for hackathon and educational use.
