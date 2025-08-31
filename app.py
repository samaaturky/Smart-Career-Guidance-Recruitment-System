import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

from utils import (
    load_artifacts,
    recommend_jobs,
    extract_user_skills,
    suggest_missing_skills,
)

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Smart Career Advisor", layout="wide")

# -----------------------
# Load model
# -----------------------
model, tfidf = load_artifacts()

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    selected = option_menu(
        menu_title=" Navigation",
        options=[" Home", " Dashboard", " About"],
        icons=["house", "bar-chart", "info-circle"],
        default_index=0,
    )

# -----------------------
# Home
# -----------------------
if selected == " Home":
    st.markdown(
        """
        <h1 style="text-align: center; color: orange;"> Smart Career Advisor</h1>
        <p style="text-align: center; font-size:18px;">
        Upload your CV and get career advice ✨
        </p>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("📂 Upload your CV", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        # unified function handles extraction + prediction
        preds, cv_text = recommend_jobs(uploaded_file, model, tfidf, top_k=3)
        career, prob = preds[0]

        with st.expander("📄 Preview Extracted Text", expanded=False):
            st.text(cv_text[:1000] + ("..." if len(cv_text) > 1000 else ""))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.success(f"🎯 **Career Path:** {career} ({prob*100:.1f}%)")

        with col2:
            user_skills = extract_user_skills(cv_text)
            missing = suggest_missing_skills(career, user_skills)
            st.warning(f"🛠 **Missing Skills:** {', '.join(missing) if missing else 'None 🎉'}")

        with col3:
            jobs = [career.title(), f"{career} Specialist", f"{career} Expert"]
            st.info(f"💼 **Job Roles:** {', '.join(jobs)}")

        st.session_state["career"] = career
        st.session_state["missing"] = missing
        st.session_state["jobs"] = jobs

# -----------------------
# Dashboard
# -----------------------
elif selected == " Dashboard":
    st.header("📊 Visualization Dashboard")
    if "jobs" in st.session_state and "missing" in st.session_state:
        jobs = st.session_state["jobs"]
        missing = st.session_state["missing"]

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=jobs,
                values=[1]*len(jobs),
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set2)
            )])
            fig.update_layout(title="Relevant Job Roles")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if missing:
                importance = list(range(len(missing), 0, -1))
                fig = px.bar(
                    x=importance, y=missing, orientation="h",
                    color=importance, text=missing,
                    color_continuous_scale="Tealgrn"
                )
                fig.update_layout(
                    title="Skill Gaps (0 = Least important , 10 = Most important)",
                    xaxis_title="Importance Level",
                    yaxis_title="Skill",
                    coloraxis_showscale=False
                )
                fig.update_traces(textposition="inside")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing skills detected!")
    else:
        st.warning("⚠️ Please upload a CV first from Home page.")

# -----------------------
# About
# -----------------------
elif selected == " About":
    st.markdown(
        """
        ### ℹ️ About this Project
        - 🎯 Predicts the best **career path** based on your CV.
        - 🛠 Suggests missing **skills** you need to improve.
        - 💼 Recommends possible **job roles**.
        ---
        👩‍💻 *Made with love by our Data Science Team.*
        """
    )
