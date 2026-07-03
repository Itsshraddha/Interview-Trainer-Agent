"""
agent.py
========
End-to-end orchestration for the Interview Trainer Agent.

Flow
----
1. Optionally parse & summarize the uploaded resume/JD (resume_parser.py).
2. Build a retrieval query from role + experience + resume skills.
3. Retrieve top-k context chunks from ChromaDB (retriever.py).
4. Assemble the structured Granite prompt (prompt_templates.py).
5. Call IBM Granite via ModelInference (watsonx_client.py).
6. Parse the JSON response into a typed Python dict.
7. Return the structured prep-kit dict to the Streamlit UI.

JSON repair strategy
--------------------
Granite sometimes wraps JSON in markdown fences (```json ... ```) or adds a
short preamble before the opening brace.  The _extract_json() helper strips
those artefacts before json.loads() so the app degrades gracefully instead of
crashing on minor formatting deviations.
"""

import io
import json
import re
from typing import Optional, TypedDict

from ibm_watsonx_ai.foundation_models import ModelInference

from src.watsonx_client import get_generation_model, get_embedding_model
from src.retriever import retrieve_context
from src.resume_parser import extract_resume_text, summarize_resume
from src.prompt_templates import build_prompt


# ── Typed output schema ───────────────────────────────────────────────────────

class TechnicalQuestion(TypedDict):
    question: str
    model_answer: str
    tips: list[str]


class BehavioralQuestion(TypedDict):
    question: str
    star_answer_outline: str
    tips: list[str]


class PrepKit(TypedDict):
    technical_questions: list[TechnicalQuestion]
    behavioral_questions: list[BehavioralQuestion]
    confidence_checklist: list[str]


# ── JSON extraction / repair ──────────────────────────────────────────────────

def _extract_json(raw: str) -> str:
    """
    Extract the first JSON object from a string that may contain:
    - Markdown code fences:  ```json { ... } ```  or  ``` { ... } ```
    - Preamble text before the opening brace
    - Trailing text after the closing brace

    Returns the cleaned JSON string, or raises ValueError if no JSON object
    can be located.
    """
    # 1. Strip markdown code fences (```json ... ``` or ``` ... ```)
    fenced = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    fenced = fenced.replace("```", "")

    # 2. Find the first '{' and the last '}' to extract the outermost object.
    start = fenced.find("{")
    end = fenced.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError(
            "Could not locate a JSON object in the model's response.\n"
            f"Raw output was:\n{raw[:500]}"
        )

    return fenced[start: end + 1]


def _parse_response(raw: str) -> PrepKit:
    """
    Parse the model's raw text output into a PrepKit dict.

    Applies _extract_json() first, then json.loads().  Falls back to a
    structured error dict if parsing fails completely so the UI always has
    something to display.
    """
    try:
        cleaned = _extract_json(raw)
        data = json.loads(cleaned)
    except (ValueError, json.JSONDecodeError) as exc:
        # Return a graceful error structure rather than crashing the app.
        return PrepKit(
            technical_questions=[
                {
                    "question": "⚠️ The model returned an unparseable response.",
                    "model_answer": (
                        f"Raw model output (first 800 chars):\n\n{raw[:800]}"
                    ),
                    "tips": [
                        "Try regenerating — the model occasionally produces "
                        "malformed JSON.",
                        f"Parse error detail: {exc}",
                    ],
                }
            ],
            behavioral_questions=[],
            confidence_checklist=[
                "Regenerate the prep kit using the button in the sidebar.",
                "If the error persists, check the watsonx.ai API status.",
            ],
        )

    # Validate and normalise the required top-level keys.
    technical = data.get("technical_questions", [])
    behavioral = data.get("behavioral_questions", [])
    checklist = data.get("confidence_checklist", [])

    # Ensure every question has the required subfields (fill defaults if absent).
    normalised_technical: list[TechnicalQuestion] = []
    for item in technical:
        normalised_technical.append(
            TechnicalQuestion(
                question=item.get("question", "(no question)"),
                model_answer=item.get("model_answer", "(no answer provided)"),
                tips=item.get("tips", []),
            )
        )

    normalised_behavioral: list[BehavioralQuestion] = []
    for item in behavioral:
        normalised_behavioral.append(
            BehavioralQuestion(
                question=item.get("question", "(no question)"),
                star_answer_outline=item.get("star_answer_outline", "(no outline provided)"),
                tips=item.get("tips", []),
            )
        )

    return PrepKit(
        technical_questions=normalised_technical,
        behavioral_questions=normalised_behavioral,
        confidence_checklist=checklist if isinstance(checklist, list) else [],
    )


# ── Main orchestration function ───────────────────────────────────────────────

def generate_prep_kit(
    candidate_name: str,
    target_role: str,
    experience_level: str,
    uploaded_file: Optional[io.BytesIO] = None,
    top_k: int = 5,
) -> PrepKit:
    """
    Generate a tailored interview preparation kit for the candidate.

    Parameters
    ----------
    candidate_name   : The candidate's name.
    target_role      : The job role they are interviewing for.
    experience_level : Self-reported level (e.g. "Mid Level (2-5 years)").
    uploaded_file    : Optional PDF or .txt file-like object from st.file_uploader.
    top_k            : Number of knowledge-base chunks to retrieve (default 5).

    Returns
    -------
    A PrepKit TypedDict containing:
    - technical_questions  : 5 role-specific questions with answers + tips
    - behavioral_questions : 3 STAR behavioral questions with outlines + tips
    - confidence_checklist : 5-7 actionable preparation items
    """
    # ── Step 1: Load the IBM Granite generation model ─────────────────────────
    # get_generation_model() reads credentials from env vars and returns a
    # configured ModelInference object ready for .generate_text() calls.
    model: ModelInference = get_generation_model()

    # ── Step 2: Parse & summarise the uploaded resume/JD (optional) ──────────
    resume_summary = ""
    if uploaded_file is not None:
        try:
            raw_text = extract_resume_text(uploaded_file)
            if raw_text.strip():
                # summarize_resume() makes an additional Granite API call to
                # condense the resume into a ~200-word skills summary.
                resume_summary = summarize_resume(raw_text, model)
        except Exception as exc:
            # Non-fatal — continue without a resume summary.
            resume_summary = f"[Resume parsing error: {exc}]"

    # ── Step 3: Build the retrieval query ─────────────────────────────────────
    # Combine role, experience level, and extracted skills into a single query
    # string that will be embedded and compared against the knowledge base.
    retrieval_query = f"{target_role} interview questions {experience_level}"
    if resume_summary and not resume_summary.startswith("["):
        # Append the first 200 chars of the summary to enrich the query.
        retrieval_query += f" skills: {resume_summary[:200]}"

    # ── Step 4: Retrieve context from ChromaDB ────────────────────────────────
    # retrieve_context() calls WatsonxEmbeddings.embed_query() on the query,
    # then runs an ANN search against the persisted Chroma index.
    try:
        context_chunks = retrieve_context(retrieval_query, k=top_k)
    except RuntimeError as exc:
        # Vector store not built yet — surface a clear error.
        raise RuntimeError(str(exc)) from exc

    # ── Step 5: Build the prompt ──────────────────────────────────────────────
    # build_prompt() fills the template with the user profile, resume summary,
    # and retrieved context chunks, plus the few-shot JSON example.
    prompt = build_prompt(
        candidate_name=candidate_name,
        target_role=target_role,
        experience_level=experience_level,
        context_chunks=context_chunks,
        resume_summary=resume_summary,
    )

    # ── Step 6: Call IBM Granite ──────────────────────────────────────────────
    # ModelInference.generate_text() sends a single synchronous POST request
    # to the watsonx.ai /ml/v1/text/generation endpoint and returns the
    # generated text as a plain string.
    try:
        raw_output = model.generate_text(prompt=prompt)
    except Exception as exc:
        raise RuntimeError(
            f"watsonx.ai API call failed: {exc}\n"
            "Check your credentials in .env and that your Watson Machine "
            "Learning instance is active on IBM Cloud."
        ) from exc

    # ── Step 7: Parse the JSON response ──────────────────────────────────────
    prep_kit = _parse_response(raw_output)

    return prep_kit
