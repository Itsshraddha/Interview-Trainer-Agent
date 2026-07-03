import io
import sys
import pathlib

import streamlit as st

# Ensure src/ is importable when running from the project root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.agent import generate_prep_kit

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interview Trainer Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS 
st.markdown(
    """
    <style>
    /* ── Global typography ── */
    html, body, [class*="css"] {
        font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        color: #ffffff;
        margin-bottom: 1.5rem;
    }
    .hero-banner h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        color: #ffffff !important;
    }
    .hero-banner p {
        font-size: 1.05rem;
        opacity: 0.88;
        margin: 0;
        line-height: 1.6;
    }

    /* ── Feature cards on landing ── */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin: 1.2rem 0;
    }
    .feature-card {
        background: #f7f9ff;
        border: 1px solid #dbe4ff;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        text-align: center;
    }
    .feature-card .fc-icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
    .feature-card .fc-title {
        font-weight: 600;
        font-size: 0.88rem;
        color: #1e3a5f;
        margin-bottom: 0.25rem;
    }
    .feature-card .fc-desc { font-size: 0.78rem; color: #57606a; line-height: 1.4; }

    /* ── Section header pills ── */
    .section-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: #1e3a5f;
        color: #ffffff;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        letter-spacing: 0.02em;
    }

    /* ── Question number badge ── */
    .q-badge {
        display: inline-block;
        background: #2563eb;
        color: #fff;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 12px;
        margin-right: 0.5rem;
        vertical-align: middle;
        letter-spacing: 0.04em;
    }
    .hr-badge {
        background: #7c3aed;
    }

    /* ── Answer / tip / star boxes ── */
    .answer-box {
        background: #f0f6ff;
        border-left: 4px solid #2563eb;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0 12px 0;
        font-size: 0.93rem;
        line-height: 1.65;
    }
    .tip-box {
        background: #fffbea;
        border-left: 4px solid #f59e0b;
        padding: 8px 13px;
        border-radius: 0 8px 8px 0;
        margin: 5px 0;
        font-size: 0.88rem;
        line-height: 1.55;
        color: #78350f;
    }
    .star-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0 12px 0;
        font-size: 0.93rem;
        line-height: 1.65;
    }

    /* ── Checklist items ── */
    .checklist-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        margin: 0.4rem 0;
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .checklist-card .check-icon { color: #22c55e; font-size: 1rem; flex-shrink: 0; }

    /* ── Profile summary card (sidebar) ── */
    .profile-card {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        color: #fff;
        margin-bottom: 0.8rem;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    .profile-card strong { font-size: 1rem; }

    /* ── Download button area ── */
    .dl-wrapper {
        background: #f7f9ff;
        border: 1px solid #dbe4ff;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.2rem;
    }
    .dl-wrapper .dl-label {
        font-weight: 600;
        color: #1e3a5f;
        font-size: 0.95rem;
    }
    .dl-wrapper .dl-sub { font-size: 0.8rem; color: #57606a; }

    /* ── Divider override ── */
    hr { border-color: #e5e7eb !important; }

    /* ── Streamlit expander header tweak ── */
    details summary {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar — Input form 
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 0.2rem 0;">
            <span style="font-size:2.2rem;">🎯</span>
            <div style="font-size:1.15rem; font-weight:700; color:#1e3a5f; margin-top:0.2rem;">
                Interview Trainer
            </div>
            <div style="font-size:0.78rem; color:#57606a; margin-top:0.15rem;">
                Powered by IBM Granite · watsonx.ai
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**👤 Your Profile**")

    candidate_name = st.text_input(
        "Full Name",
        placeholder="e.g. Jordan Lee",
        help="Used to personalise the generated prep kit.",
    )

    target_role = st.selectbox(
        "🏢 Target Role",
        options=[
            "Software Engineer",
            "Data Analyst",
            "Data Scientist",
            "Product Manager",
            "DevOps Engineer",
            "Machine Learning Engineer",
            "Other (specify below)",
        ],
        help="Choose the role you are interviewing for.",
    )

    if target_role == "Other (specify below)":
        target_role = st.text_input(
            "Specify Role",
            placeholder="e.g. Site Reliability Engineer",
        )

    experience_level = st.selectbox(
        "📈 Experience Level",
        options=[
            "Entry Level (0-2 years)",
            "Mid Level (2-5 years)",
            "Senior Level (5+ years)",
        ],
    )

    st.markdown("**📎 Resume / Job Description**")
    uploaded_file = st.file_uploader(
        "Upload PDF or TXT (optional)",
        type=["pdf", "txt"],
        help=(
            "Upload a PDF or .txt resume or job description. "
            "The agent will extract key skills and personalise the questions."
        ),
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.success(f"✅ {uploaded_file.name} uploaded")

    st.markdown("**⚙️ Settings**")
    top_k = st.slider(
        "Knowledge base chunks",
        min_value=3,
        max_value=10,
        value=5,
        help=(
            "More chunks = more context for the model, but slightly slower. "
            "5 is a good default."
        ),
    )

    st.divider()

    # Live profile preview
    if candidate_name and target_role:
        st.markdown(
            f"""
            <div class="profile-card">
                <strong>{candidate_name}</strong><br>
                {target_role}<br>
                <span style="opacity:0.8; font-size:0.82rem;">{experience_level}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    generate_btn = st.button(
        "✨ Generate Prep Kit",
        type="primary",
        use_container_width=True,
        disabled=not candidate_name or not target_role,
    )

    if not candidate_name:
        st.caption("⬆ Enter your name to enable generation.")

    st.divider()
    with st.expander("ℹ️ How it works"):
        st.markdown(
            """
            1. Your profile + optional resume are processed.
            2. Relevant interview Q&A is retrieved from a local knowledge base
               (ChromaDB + IBM Slate embeddings).
            3. IBM Granite generates a personalised prep kit via watsonx.ai.
            4. Results are shown below — expand each question to see the
               model answer and improvement tips.
            """
        )


# ── Main area 
# Initialise session state to persist the last generated kit across reruns.
if "prep_kit" not in st.session_state:
    st.session_state.prep_kit = None
if "last_profile" not in st.session_state:
    st.session_state.last_profile = {}


def _render_prep_kit(kit: dict, profile: dict) -> None:
    """Render the prep kit dict as structured Streamlit UI components."""

    # ── Kit header with download 
    markdown_content = _kit_to_markdown(kit, profile)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f"""
            <div style="margin-bottom:0.2rem;">
                <span style="font-size:1.35rem; font-weight:700; color:#1e3a5f;">
                    Interview Prep Kit
                </span><br>
                <span style="color:#57606a; font-size:0.92rem;">
                    {profile.get('role', '')} · {profile.get('level', '')} · {profile.get('name', '')}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.download_button(
            label="⬇ Download Markdown",
            data=markdown_content,
            file_name="interview_prep_kit.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()

    # ── Technical Questions 
    st.markdown(
        '<div class="section-pill">💻 Technical Questions</div>',
        unsafe_allow_html=True,
    )
    tech_questions = kit.get("technical_questions", [])
    if not tech_questions:
        st.info("No technical questions were generated.")
    else:
        for i, q in enumerate(tech_questions, start=1):
            with st.expander(f"Q{i}:  {q.get('question', '(no question)')}"):
                st.markdown(
                    '<span style="font-weight:600; color:#2563eb;">📘 Model Answer</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="answer-box">{q.get("model_answer", "_No answer provided._")}</div>',
                    unsafe_allow_html=True,
                )

                tips = q.get("tips", [])
                if tips:
                    st.markdown(
                        '<span style="font-weight:600; color:#b45309;">💡 Improvement Tips</span>',
                        unsafe_allow_html=True,
                    )
                    for tip in tips:
                        st.markdown(
                            f'<div class="tip-box">💡 {tip}</div>',
                            unsafe_allow_html=True,
                        )

    st.divider()

    # ── Behavioral / HR Questions
    st.markdown(
        '<div class="section-pill" style="background:#7c3aed;">🤝 Behavioral / HR Questions — STAR Format</div>',
        unsafe_allow_html=True,
    )
    beh_questions = kit.get("behavioral_questions", [])
    if not beh_questions:
        st.info("No behavioral questions were generated.")
    else:
        for i, q in enumerate(beh_questions, start=1):
            with st.expander(f"HR{i}:  {q.get('question', '(no question)')}"):
                st.markdown(
                    '<span style="font-weight:600; color:#15803d;">⭐ STAR Answer Outline</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="star-box">{q.get("star_answer_outline", "_No outline provided._")}</div>',
                    unsafe_allow_html=True,
                )

                tips = q.get("tips", [])
                if tips:
                    st.markdown(
                        '<span style="font-weight:600; color:#b45309;">💡 Improvement Tips</span>',
                        unsafe_allow_html=True,
                    )
                    for tip in tips:
                        st.markdown(
                            f'<div class="tip-box">💡 {tip}</div>',
                            unsafe_allow_html=True,
                        )

    st.divider()

    # ── Confidence Checklist 
    st.markdown(
        '<div class="section-pill" style="background:#15803d;">✅ Confidence Checklist</div>',
        unsafe_allow_html=True,
    )
    checklist = kit.get("confidence_checklist", [])
    if not checklist:
        st.info("No checklist items were generated.")
    else:
        for item in checklist:
            st.markdown(
                f'<div class="checklist-card"><span class="check-icon">✔</span> {item}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div style="text-align:center; margin-top:2rem; padding-top:1rem;
                    border-top:1px solid #e5e7eb; font-size:0.78rem; color:#57606a;">
            Generated by Interview Trainer Agent · IBM Granite · watsonx.ai
        </div>
        """,
        unsafe_allow_html=True,
    )


def _kit_to_markdown(kit: dict, profile: dict) -> str:
    """Convert the prep kit dict to a Markdown string for download."""
    lines: list[str] = []
    lines.append("# Interview Prep Kit")
    lines.append(
        f"**Role:** {profile.get('role', '')}  |  "
        f"**Level:** {profile.get('level', '')}  |  "
        f"**Candidate:** {profile.get('name', '')}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 💻 Technical Questions")
    lines.append("")
    for i, q in enumerate(kit.get("technical_questions", []), start=1):
        lines.append(f"### Q{i}: {q.get('question', '')}")
        lines.append("")
        lines.append("**Model Answer**")
        lines.append("")
        lines.append(q.get("model_answer", ""))
        lines.append("")
        tips = q.get("tips", [])
        if tips:
            lines.append("**Improvement Tips**")
            lines.append("")
            for tip in tips:
                lines.append(f"- {tip}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 🤝 Behavioral / HR Questions")
    lines.append("")
    for i, q in enumerate(kit.get("behavioral_questions", []), start=1):
        lines.append(f"### HR{i}: {q.get('question', '')}")
        lines.append("")
        lines.append("**STAR Answer Outline**")
        lines.append("")
        lines.append(q.get("star_answer_outline", ""))
        lines.append("")
        tips = q.get("tips", [])
        if tips:
            lines.append("**Improvement Tips**")
            lines.append("")
            for tip in tips:
                lines.append(f"- {tip}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## ✅ Confidence Checklist")
    lines.append("")
    for item in kit.get("confidence_checklist", []):
        lines.append(f"- [ ] {item}")
    lines.append("")
    lines.append("---")
    lines.append("_Generated by Interview Trainer Agent · IBM Granite · watsonx.ai_")

    return "\n".join(lines)


# ── Generate on button press
if generate_btn:
    profile = {
        "name": candidate_name,
        "role": target_role,
        "level": experience_level,
    }

    # Pass the uploaded file's bytes as a BytesIO if present.
    file_obj = None
    if uploaded_file is not None:
        file_obj = io.BytesIO(uploaded_file.getvalue())

    with st.spinner(
        "🔍 Retrieving relevant knowledge base chunks …\n"
        "⚙️ Generating your prep kit with IBM Granite …\n"
        "This usually takes 15-30 seconds."
    ):
        try:
            kit = generate_prep_kit(
                candidate_name=candidate_name,
                target_role=target_role,
                experience_level=experience_level,
                uploaded_file=file_obj,
                top_k=top_k,
            )
            st.session_state.prep_kit = kit
            st.session_state.last_profile = profile
            st.success("✅ Your personalised prep kit is ready!")

        except EnvironmentError as exc:
            st.error(
                "**Credentials not configured.**\n\n"
                f"{exc}\n\n"
                "See README.md → *IBM Cloud Lite Setup* for step-by-step instructions."
            )
            st.stop()

        except RuntimeError as exc:
            error_msg = str(exc)
            if "Vector store not found" in error_msg:
                st.error(
                    "**Knowledge base not built yet.**\n\n"
                    "Run the following command once before starting the app:\n\n"
                    "```bash\npython src/ingest.py\n```"
                )
            else:
                st.error(f"**Generation failed.**\n\n{error_msg}")
            st.stop()

        except Exception as exc:
            st.error(
                f"**Unexpected error:** {exc}\n\n"
                "Check the terminal for the full traceback."
            )
            st.stop()


# ── Render the stored prep kit (persists across reruns) 
if st.session_state.prep_kit is not None:
    _render_prep_kit(st.session_state.prep_kit, st.session_state.last_profile)
else:
    # ── Hero banner
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🎯 Interview Trainer Agent</h1>
            <p>
                Get a <strong>personalised interview prep kit</strong> — technical questions,
                behavioral STAR-format questions, model answers, and improvement tips —
                powered by <strong>IBM Granite</strong> on <strong>watsonx.ai</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Feature cards
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card">
                <div class="fc-icon">💻</div>
                <div class="fc-title">Technical Questions</div>
                <div class="fc-desc">5 role-specific questions with model answer outlines</div>
            </div>
            <div class="feature-card">
                <div class="fc-icon">🤝</div>
                <div class="fc-title">Behavioral / HR</div>
                <div class="fc-desc">3 STAR-format questions with structured answer frameworks</div>
            </div>
            <div class="feature-card">
                <div class="fc-icon">💡</div>
                <div class="fc-title">Improvement Tips</div>
                <div class="fc-desc">2 targeted tips per question to sharpen your delivery</div>
            </div>
            <div class="feature-card">
                <div class="fc-icon">✅</div>
                <div class="fc-title">Confidence Checklist</div>
                <div class="fc-desc">Personalised preparation actions before your interview</div>
            </div>
            <div class="feature-card">
                <div class="fc-icon">⬇</div>
                <div class="fc-title">Download Kit</div>
                <div class="fc-desc">Export the full prep kit as a Markdown file</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin-top:1.5rem; padding:1rem 1.2rem; background:#f7f9ff;
                    border:1px solid #dbe4ff; border-radius:10px; color:#57606a; font-size:0.9rem;">
            👈 <strong>Fill in your profile</strong> in the sidebar and click
            <strong>✨ Generate Prep Kit</strong> to get started.
        </div>
        """,
        unsafe_allow_html=True,
    )
