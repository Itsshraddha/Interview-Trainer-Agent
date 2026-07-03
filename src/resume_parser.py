"""
resume_parser.py
================
Two utilities for handling resume / job-description uploads:

1. extract_resume_text(file)     — extract raw text from a PDF (or plain text)
                                    file object (as returned by st.file_uploader).
2. summarize_resume(text, model) — call the Granite model to condense the raw
                                    text into a concise "key skills" summary that
                                    is small enough to fit cleanly in the main
                                    interview-prep prompt.
"""

import io
from typing import Union

import pdfplumber  # PDF text extraction

from ibm_watsonx_ai.foundation_models import ModelInference


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_resume_text(file: Union[io.BytesIO, bytes, str]) -> str:
    """
    Extract plain text from an uploaded file.

    Supports:
    - PDF files (binary): pdfplumber extracts text page by page.
    - Plain-text .txt files: decoded as UTF-8.

    Parameters
    ----------
    file : A file-like object (io.BytesIO), raw bytes, or a str path.
           When called from Streamlit, pass the object returned by
           st.file_uploader() directly — it behaves like a BytesIO.

    Returns
    -------
    Extracted text as a single string, with pages separated by newlines.
    Returns an empty string if extraction fails gracefully.
    """
    # Normalise input to a BytesIO object so pdfplumber can seek it.
    if isinstance(file, bytes):
        file = io.BytesIO(file)
    elif isinstance(file, str):
        # Treat as a file path
        with open(file, "rb") as fh:
            file = io.BytesIO(fh.read())
    # else: already a file-like object

    # Peek at the first 4 bytes to detect PDF magic bytes (%PDF).
    file.seek(0)
    header = file.read(4)
    file.seek(0)

    if header == b"%PDF":
        # ── PDF extraction via pdfplumber ─────────────────────────────────────
        # pdfplumber is robust to encrypted/scanned PDFs and handles tables and
        # multi-column layouts better than PyPDF2.
        pages: list[str] = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text.strip())
        return "\n\n".join(pages)
    else:
        # ── Plain text fallback ───────────────────────────────────────────────
        file.seek(0)
        try:
            return file.read().decode("utf-8", errors="replace")
        except Exception:
            return ""


# ── LLM-based summarization ───────────────────────────────────────────────────

_SUMMARIZE_PROMPT = """\
You are an expert career coach reading a candidate's resume or job description.
Extract and list the most important professional details in the following categories:

1. Core Technical Skills (programming languages, frameworks, tools, platforms)
2. Domain Expertise (industries, subject matter areas)
3. Years of Experience (estimate if not explicit)
4. Key Achievements (quantified accomplishments if present)
5. Certifications or Education (if relevant)

Be concise — the output will be injected into an interview preparation prompt.
Return ONLY the structured list, no preamble or closing remarks.

RESUME / JOB DESCRIPTION TEXT:
\"\"\"
{resume_text}
\"\"\"

STRUCTURED SKILLS SUMMARY:"""

# Maximum characters of resume text to send to the model — avoids exceeding
# context windows and keeps costs low on the Lite tier.
MAX_RESUME_CHARS = 3000


def summarize_resume(text: str, model: ModelInference) -> str:
    """
    Use the Granite model to condense resume/JD text into a key-skills summary.

    Parameters
    ----------
    text  : Raw text extracted from the uploaded PDF or .txt file.
    model : An initialised ModelInference instance (from watsonx_client.py).

    Returns
    -------
    A concise structured string listing the candidate's key skills,
    experience, and achievements — ready to be embedded in the main prompt.
    """
    # Truncate to avoid hitting context limits on the Lite tier.
    truncated = text[:MAX_RESUME_CHARS]
    if len(text) > MAX_RESUME_CHARS:
        truncated += "\n[... truncated for brevity ...]"

    prompt = _SUMMARIZE_PROMPT.format(resume_text=truncated)

    # Call IBM Granite for a short, focused summarization task.
    # ModelInference.generate_text() returns the generated text string directly.
    try:
        summary = model.generate_text(prompt=prompt)
        return summary.strip() if summary else ""
    except Exception as exc:
        # Non-fatal — if summarization fails, the main agent can continue
        # without a resume summary (it will just use the role + level).
        return f"[Resume summarization failed: {exc}]"
