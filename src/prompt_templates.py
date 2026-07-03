"""
prompt_templates.py
===================
Structured prompt template for the IBM Granite generation model.

Design goals
------------
- Instruct Granite to return ONLY valid JSON matching the schema below.
- Include a complete few-shot example so the model understands the format
  without ambiguity (critical for reliable JSON output from a chat model).
- Use clear section delimiters (###) that are robust to slight model quirks.
- Keep the system instruction concise — Granite 3.x is instruction-following
  but benefits from explicit, structured directives.

Output JSON schema
------------------
{
  "technical_questions": [
    {
      "question": "string",
      "model_answer": "string",
      "tips": ["string", "string"]
    }
    // × 5
  ],
  "behavioral_questions": [
    {
      "question": "string",
      "star_answer_outline": "string",
      "tips": ["string", "string"]
    }
    // × 3
  ],
  "confidence_checklist": ["string", ...]   // 5-7 items
}
"""

# ── System instruction ────────────────────────────────────────────────────────
# Granite 3.3 respects a clear system role separator.  We define the model's
# persona, output format, and strictness requirements here.

_SYSTEM_INSTRUCTION = """\
You are an expert interview coach with deep knowledge of technical hiring \
processes, behavioral assessment frameworks, and industry best practices. \
Your task is to generate a tailored interview preparation kit for a job candidate.

CRITICAL RULES — follow these exactly:
1. Respond with ONLY a single valid JSON object. No markdown, no code fences, \
no explanations before or after the JSON.
2. The JSON must match the schema shown in the example below precisely.
3. Generate exactly 5 technical questions and exactly 3 behavioral questions.
4. Each question must have exactly 2 tips.
5. The confidence_checklist must contain 5 to 7 short, actionable bullet points.
6. Do not truncate. Complete every field fully.
"""

# ── Few-shot example ──────────────────────────────────────────────────────────
# A single worked example greatly improves JSON reliability with Granite.
# The example uses a generic "Software Engineer / Entry Level" scenario so it
# is clearly distinguishable from the actual task injected later.

_FEW_SHOT_EXAMPLE = """\
### EXAMPLE (do not use this data — it is for format guidance only) ###

INPUT:
- Candidate: Alex, Entry Level Software Engineer
- Context: Python, REST APIs, data structures

OUTPUT:
{
  "technical_questions": [
    {
      "question": "Explain the difference between a list and a tuple in Python.",
      "model_answer": "Lists are mutable sequences; tuples are immutable. \
Use tuples for data that should not change (e.g., RGB values, coordinates) \
and lists when you need to append, remove, or sort items. Tuples are slightly \
faster to create and can be used as dictionary keys.",
      "tips": [
        "Mention that tuples are hashable and can serve as dict keys or set members.",
        "Give a concrete example: coordinates as a tuple vs. a shopping cart as a list."
      ]
    }
  ],
  "behavioral_questions": [
    {
      "question": "Tell me about a time you had to learn a new technology quickly.",
      "star_answer_outline": "Situation: joined a project two weeks before a \
deadline that required React (which I had not used before). Task: become \
productive enough to contribute meaningful frontend code. Action: completed \
the official React tutorial in two days, then built a small practice component, \
paired with a senior engineer for code review. Result: delivered two features \
on time; team lead noted the quality of my code in the retro.",
      "tips": [
        "Quantify how quickly you became productive (e.g., contributing within 3 days).",
        "Emphasise the learning strategy, not just the outcome."
      ]
    }
  ],
  "confidence_checklist": [
    "Review Python fundamentals: lists, dicts, comprehensions, and generators.",
    "Practice explaining Big-O complexity for common operations.",
    "Prepare two STAR stories covering learning agility and teamwork.",
    "Research the company's tech stack and recent engineering blog posts.",
    "Prepare three thoughtful questions to ask the interviewer."
  ]
}

### END EXAMPLE ###
"""

# ── Main prompt template ──────────────────────────────────────────────────────
# Placeholders are filled by build_prompt() below.

_MAIN_TEMPLATE = """\
{system_instruction}

{few_shot_example}

### YOUR TASK ###

Generate a personalised interview preparation kit for the following candidate.

CANDIDATE PROFILE:
- Name: {candidate_name}
- Target Role: {target_role}
- Experience Level: {experience_level}

{resume_section}

RETRIEVED KNOWLEDGE BASE CONTEXT (use this to ground your questions and answers):
---
{context}
---

INSTRUCTIONS:
- Generate 5 technical questions highly relevant to the "{target_role}" role \
at {experience_level} level. Base questions and model answers on the retrieved \
context above.
- Generate 3 behavioral / HR questions in STAR format appropriate for the \
experience level.
- For each question (both technical and behavioral), provide exactly 2 \
specific, actionable improvement tips.
- Generate a confidence_checklist of 5-7 short, actionable preparation items \
tailored to this candidate's profile.
- If a resume summary is provided, personalise questions to the candidate's \
stated skills and background.
- Return ONLY the JSON object. No markdown fences. No text before or after \
the JSON.

JSON OUTPUT:"""


def build_prompt(
    candidate_name: str,
    target_role: str,
    experience_level: str,
    context_chunks: list[str],
    resume_summary: str = "",
) -> str:
    """
    Assemble the full Granite prompt from the template and runtime values.

    Parameters
    ----------
    candidate_name   : The user's name (personalises the output).
    target_role      : e.g. "Software Engineer", "Data Analyst".
    experience_level : e.g. "Entry Level (0-2 years)".
    context_chunks   : Top-k text chunks from the Chroma retrieval step.
    resume_summary   : Optional condensed skills summary from the resume parser.

    Returns
    -------
    A fully assembled prompt string ready to pass to ModelInference.generate_text().
    """
    # Format the retrieved chunks into a numbered list for readability.
    formatted_context = "\n\n".join(
        f"[Chunk {i + 1}]\n{chunk.strip()}"
        for i, chunk in enumerate(context_chunks)
    )

    # Only add the resume section if a summary was provided.
    if resume_summary and resume_summary.strip():
        resume_section = (
            f"CANDIDATE RESUME / JD SUMMARY:\n{resume_summary.strip()}\n"
        )
    else:
        resume_section = (
            "CANDIDATE RESUME / JD SUMMARY: Not provided — use role and "
            "experience level only.\n"
        )

    return _MAIN_TEMPLATE.format(
        system_instruction=_SYSTEM_INSTRUCTION,
        few_shot_example=_FEW_SHOT_EXAMPLE,
        candidate_name=candidate_name,
        target_role=target_role,
        experience_level=experience_level,
        resume_section=resume_section,
        context=formatted_context,
    )
