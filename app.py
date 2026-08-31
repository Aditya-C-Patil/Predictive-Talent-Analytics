import streamlit as st
import numpy as np
import pickle
import os
import difflib
from train import train_and_export

# 1. Page Configuration
st.set_page_config(
    page_title="TalentPulse AI | Predictive Career Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern Custom Styling (Glassmorphism, Neon Accents, Card Radii)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Glow Badge */
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        color: #818cf8;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    
    /* Glassmorphism Analysis Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s ease, border-color 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .role-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 8px;
    }
    
    .coverage-number {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    
    .tag-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        margin: 2px 3px 2px 0;
    }
    
    .gap-badge {
        display: inline-block;
        background: rgba(244, 63, 94, 0.12);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.25);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        margin: 2px 3px 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model & Knowledge Base Loader
@st.cache_resource
def load_resources():
    if not os.path.exists("models/classifier.pkl") or not os.path.exists("models/all_skills.pkl"):
        train_and_export()
    with open("models/classifier.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("models/role_skills_map.pkl", "rb") as f:
        role_skills_map = pickle.load(f)
    with open("models/all_skills.pkl", "rb") as f:
        all_skills = pickle.load(f)
    return model, tfidf, role_skills_map, all_skills

model, tfidf, role_skills_map, all_skills = load_resources()
display_skill_options = [s.title() for s in all_skills]

# 4. Telemetry Sidebar
with st.sidebar:
    st.markdown("### ⚡ System Telemetry")
    st.markdown("---")
    st.markdown(f"**Indexed Job Profiles:** `{len(role_skills_map)} Roles`")
    st.markdown(f"**Master Skill Catalog:** `{len(all_skills)} Entities`")
    st.markdown(f"**Classifier Backend:** `RandomForest (n=150)`")
    st.markdown(f"**Sampling Strategy:** `SMOTE (k_neighbors=1)`")
    st.markdown(f"**Vector Space:** `Sublinear TF-IDF (1,2)`")
    st.markdown("---")
    st.info("💡 **Tip:** Select 3+ competencies to simulate accurate semantic role alignment and detect missing curriculum gaps.")

# 5. Hero Header Section
st.markdown('<div class="hero-badge">⚡ Next-Gen Talent Intelligence</div>', unsafe_allow_html=True)
st.title("Predictive Talent Analytics & Semantic Skill-Gap Engine")
st.markdown("Enterprise ML analytics platform for candidate competency vectorization, class-imbalance mitigation, and curriculum readiness scoring.")

st.markdown("<br>", unsafe_allow_html=True)

# 6. Skill Selector Form
default_demo = ["Python", "Sql", "Machine Learning"] if "Python" in display_skill_options else []
selected_skills = st.multiselect(
    "Select candidate competencies from taxonomy:",
    options=display_skill_options,
    default=default_demo,
    placeholder="Type or browse competencies (e.g., Python, C++, Docker, PyTorch)..."
)

user_skills = [s.lower().strip() for s in selected_skills]

def fuzzy_skill_match(user_skill, target_skill, threshold=0.75):
    u, t = user_skill.lower().strip(), target_skill.lower().strip()
    if u in t or t in u:
        return True
    return difflib.SequenceMatcher(None, u, t).ratio() >= threshold

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("🚀 Analyze Career Alignment & Gaps", type="primary", use_container_width=True)

# 7. Inference & Analytics Rendering
if analyze_btn:
    if not user_skills:
        st.warning("⚠️ Please select at least one technical competency from the taxonomy above.")
        st.stop()

    input_text = " ".join(user_skills)
    vec = tfidf.transform([input_text])

    probs = model.predict_proba(vec)[0]
    top_indices = np.argsort(probs)[::-1][:3]

    st.markdown("---")
    st.subheader("🎯 Optimal Role Alignments & Target Curriculum")
    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(3, gap="large")

    for idx, col in zip(top_indices, cols):
        role_name = model.classes_[idx]
        confidence = probs[idx] * 100
        required_skills = role_skills_map.get(role_name, [])

        # Calculate exact matched skills vs gaps
        matched = []
        missing = []
        for req in required_skills:
            if any(fuzzy_skill_match(usr, req) for usr in user_skills):
                matched.append(req)
            else:
                missing.append(req)

        total_req = len(required_skills)
        coverage_pct = (len(matched) / total_req * 100) if total_req > 0 else 0

        matched_tags = "".join([f'<span class="tag-badge">✓ {m.title()}</span>' for m in matched[:6]])
        gap_tags = "".join([f'<span class="gap-badge">⚠ {g.title()}</span>' for g in missing[:6]])

        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div>
                    <div class="role-title">{role_name}</div>
                    <div style="color: #94a3b8; font-size: 0.82rem; margin-bottom: 12px;">
                        Model Likelihood: <strong style="color: #cbd5e1;">{confidence:.1f}%</strong>
                    </div>
                    <div style="margin-bottom: 16px;">
                        <div style="font-size: 0.78rem; font-weight: 700; color: #64748b; text-transform: uppercase;">
                            Competency Coverage
                        </div>
                        <div class="coverage-number">{coverage_pct:.1f}%</div>
                    </div>
                    <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 12px 0;">
                    <div style="margin-bottom: 14px;">
                        <div style="font-size: 0.8rem; font-weight: 700; color: #38bdf8; margin-bottom: 6px;">
                            Matched Competencies ({len(matched)}):
                        </div>
                        {matched_tags if matched_tags else '<span style="color: #64748b; font-size: 0.8rem;">None matched yet</span>'}
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; font-weight: 700; color: #fb7185; margin-bottom: 6px;">
                            Identified Skill Gaps ({len(missing)}):
                        </div>
                        {gap_tags if gap_tags else '<span style="color: #22c55e; font-size: 0.8rem; font-weight: 600;">Full competency match!</span>'}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)