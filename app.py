# AI Career Path Predictor + Skill Gap Analyzer — Streamlit App (Streamlit)
# ---------------------------------------------------------------
# How to run locally:
# 1) Create and activate a virtualenv
#    python -m venv venv
#    # Windows: venv\Scripts\activate   |  Mac/Linux: source venv/bin/activate
# 2) Save this file as app.py in your project folder
# 3) Create requirements.txt with:
#    streamlit
#    requests
#    pandas
#    numpy
#    spacy
#    scikit-learn
# 4) python -m spacy download en_core_web_sm
# 5) streamlit run app.py
# ---------------------------------------------------------------

import os
import time
import json
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st

# Optional: load spaCy lazily to speed startup
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            _nlp = None
    return _nlp

# -------------------------
# Skill dictionary (expand)
# -------------------------
DEFAULT_SKILLS = {
    # Programming / DS
    "python", "java", "c++", "r", "scala", "sql", "nosql",
    "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn",
    "tensorflow", "pytorch", "keras", "xgboost", "lightgbm",
    "opencv", "spacy", "nltk",
    # Data Eng / Cloud
    "airflow", "spark", "hadoop", "kafka", "dbt", "etl",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    # Web / API
    "fastapi", "flask", "django", "rest api", "graphql",
    # MLOps
    "mlflow", "dvc", "feature store", "model serving",
    # Analytics / BI
    "excel", "power bi", "tableau", "looker",
    # NLP / CV
    "nlp", "transformers", "llm", "computer vision", "clip",
    # Soft-ish (keep minimal)
    "git", "linux", "jira"
}

# -------------------------
# Helpers
# -------------------------

def adzuna_fetch_jobs(app_id: str,
                      app_key: str,
                      country: str = "gb",
                      query: str = "data scientist",
                      location: str = "",
                      results_per_page: int = 50,
                      pages: int = 1,
                      max_days_old: int | None = None) -> pd.DataFrame:
    """Fetch jobs from Adzuna Jobs API with pagination.
    Returns a DataFrame with key fields.
    """
    base = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{{page}}"

    all_rows = []
    headers = {"User-Agent": "CareerPathPredictor/1.0"}

    for p in range(1, pages + 1):
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "what": query,
            "results_per_page": results_per_page,
            "content-type": "application/json",
        }
        if location:
            params["where"] = location
        if max_days_old is not None:
            params["max_days_old"] = max_days_old

        url = base.format(page=p)
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Adzuna API error on page {p}: {resp.status_code} — {resp.text[:200]}")
        data = resp.json()
        for job in data.get("results", []):
            all_rows.append({
                "title": job.get("title", ""),
                "company": (job.get("company") or {}).get("display_name", ""),
                "location": (job.get("location") or {}).get("display_name", ""),
                "created": job.get("created", ""),
                "category": (job.get("category") or {}).get("label", ""),
                "salary_min": job.get("salary_min"),
                "salary_max": job.get("salary_max"),
                "description": job.get("description", ""),
                "redirect_url": job.get("redirect_url", ""),
            })
        # Respectful small delay
        time.sleep(0.5)

    return pd.DataFrame(all_rows)


def extract_skills_freeform(text: str, dictionary: set[str]) -> list[str]:
    """Simple lowercase dictionary match; robust & fast.
    Add stemming/lemmatization via spaCy if available.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    t = text.lower()
    found = [s for s in dictionary if s in t]
    return sorted(set(found))


def nlp_boost(text: str) -> list[str]:
    """Optional: use spaCy to pick proper nouns & nouns as candidate skills.
    This is *very* heuristic; we still rely on dictionary intersection.
    """
    nlp = get_nlp()
    if nlp is None or not isinstance(text, str):
        return []
    doc = nlp(text)
    candidates = set()
    for tok in doc:
        if tok.pos_ in {"PROPN", "NOUN"} and len(tok.text) > 2:
            candidates.add(tok.text.lower())
    return sorted(candidates)


def compare_skills(job_skills: list[str], user_skills: set[str]) -> dict:
    req = set([s.strip().lower() for s in job_skills])
    mine = set([s.strip().lower() for s in user_skills])
    matched = sorted(req & mine)
    missing = sorted(req - mine)
    total = len(req) if req else 1
    match_pct = 100.0 * len(matched) / total
    return {
        "match_pct": round(match_pct, 1),
        "matched": matched,
        "missing": missing,
    }


def rank_roles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate match% by job title to provide a simple role ranking."""
    if df.empty:
        return df
    agg = (
        df.groupby("title")["match_pct"].mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    agg.rename(columns={"match_pct": "avg_match_pct"}, inplace=True)
    return agg


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="AI Career Path Predictor", page_icon="🧭", layout="wide")

st.title("🧭 AI Career Path Predictor + Skill Gap Analyzer")
st.caption("Fetch live job data, extract required skills, compare with your skills, and rank career roles by fit.")

with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**Adzuna API Credentials**")
    default_app_id = os.getenv("ADZUNA_APP_ID", "")
    default_app_key = os.getenv("ADZUNA_APP_KEY", "")
    app_id = st.text_input("APP_ID", value=default_app_id, type="default")
    app_key = st.text_input("APP_KEY", value=default_app_key, type="password")

    st.markdown("---")
    st.markdown("**Job Query**")
    country = st.selectbox("Country", options=["gb", "us", "in", "ca", "au", "de", "fr"], index=0, help="API uses 2-letter codes (lowercase)")
    query = st.text_input("What (keywords)", value="data scientist")
    location = st.text_input("Where (optional)", value="")
    results_per_page = st.slider("Results per page", min_value=10, max_value=50, value=20, step=10)
    pages = st.slider("Pages to fetch", min_value=1, max_value=5, value=1)
    max_days_old = st.slider("Max days old (optional)", min_value=0, max_value=60, value=30, help="0 = newest only; 30 is a good default")

    st.markdown("---")
    st.markdown("**Your Skills** (comma-separated)")
    skills_text = st.text_area("e.g., Python, SQL, Pandas, TensorFlow, Docker", height=80)
    user_skills = {s.strip().lower() for s in skills_text.split(",") if s.strip()}

    st.markdown("**Skill Dictionary (optional)**")
    dict_text = st.text_area("Add/override skills dictionary (comma-separated)", value=", ".join(sorted(DEFAULT_SKILLS)), height=120)
    custom_dict = {s.strip().lower() for s in dict_text.split(",") if s.strip()}

    st.markdown("---")
    st.caption("Tip: You can also upload an existing CSV of jobs (with a 'description' column) below.")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("1) Get Job Data")
    uploaded = st.file_uploader("Upload existing job CSV (optional)", type=["csv"])

    df_jobs: pd.DataFrame
    if uploaded is not None:
        df_jobs = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_jobs)} jobs from uploaded CSV")
    else:
        if st.button("🔎 Fetch from Adzuna API", use_container_width=True):
            if not app_id or not app_key:
                st.error("Please enter APP_ID and APP_KEY in the sidebar.")
            else:
                with st.spinner("Fetching jobs..."):
                    try:
                        df_jobs = adzuna_fetch_jobs(
                            app_id=app_id,
                            app_key=app_key,
                            country=country,
                            query=query,
                            location=location,
                            results_per_page=results_per_page,
                            pages=pages,
                            max_days_old=max_days_old,
                        )
                        if df_jobs.empty:
                            st.warning("No jobs returned. Try adjusting your query/filters.")
                        else:
                            st.success(f"Fetched {len(df_jobs)} jobs.")
                            st.dataframe(df_jobs.head(50), use_container_width=True)
                            csv = df_jobs.to_csv(index=False).encode("utf-8")
                            st.download_button("Download jobs.csv", data=csv, file_name="jobs.csv")
                            st.session_state["jobs_df"] = df_jobs
                    except Exception as e:
                        st.error(str(e))
        else:
            # If button not pressed and nothing uploaded, try to reuse previous session data
            df_jobs = st.session_state.get("jobs_df", pd.DataFrame())

with col2:
    st.subheader("2) Extract Skills")
    st.caption("We use a fast dictionary match; optionally boosted with spaCy if installed.")
    use_nlp = st.checkbox("Use spaCy noun/proper-noun boost (optional)", value=False)
    if st.button("🧠 Run Skill Extraction", use_container_width=True):
        df_jobs = st.session_state.get("jobs_df", uploaded and pd.read_csv(uploaded) or pd.DataFrame())
        if df_jobs is None or df_jobs.empty:
            st.error("No jobs loaded yet. Fetch via API or upload a CSV first.")
        else:
            dict_set = custom_dict if custom_dict else DEFAULT_SKILLS
            req_skills_col = []
            for desc in df_jobs.get("description", ""):
                base = set(extract_skills_freeform(desc, dict_set))
                if use_nlp:
                    boost = set(nlp_boost(desc))
                    # consider only candidates that already exist in dictionary to avoid noise
                    base = base.union({c for c in boost if c in dict_set})
                req_skills_col.append(sorted(base))
            df_jobs = df_jobs.copy()
            df_jobs["skills_required"] = req_skills_col
            st.session_state["jobs_df"] = df_jobs
            st.success("Skills extracted.")
            st.dataframe(df_jobs[["title", "company", "skills_required"]].head(50), use_container_width=True)
            csv = df_jobs.to_csv(index=False).encode("utf-8")
            st.download_button("Download career_skill_analysis.csv", data=csv, file_name="career_skill_analysis.csv")

st.markdown("---")

st.subheader("3) Compare With Your Skills & Rank Roles")
if st.button("📊 Analyze Fit & Rank Careers", type="primary"):
    df_jobs = st.session_state.get("jobs_df", pd.DataFrame())
    if df_jobs is None or df_jobs.empty:
        st.error("No jobs with extracted skills. Please complete steps 1 and 2.")
    else:
        if "skills_required" not in df_jobs.columns:
            st.error("Please run Skill Extraction first to create the 'skills_required' column.")
        else:
            results = []
            for _, row in df_jobs.iterrows():
                comp = compare_skills(row["skills_required"], user_skills)
                results.append(comp)
            out = pd.DataFrame(results)
            df_out = pd.concat([df_jobs.reset_index(drop=True), out], axis=1)
            df_out.sort_values("match_pct", ascending=False, inplace=True)

            st.success("Analysis complete.")
            st.dataframe(df_out[["title", "company", "match_pct", "matched", "missing"]].head(50), use_container_width=True)

            # Role ranking
            ranking = rank_roles(df_out)
            if not ranking.empty:
                st.markdown("#### 🏆 Top Career Matches (by average skill match %)")
                st.dataframe(ranking.head(15), use_container_width=True)

            # Download
            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("Download full_results.csv", data=csv, file_name="full_results.csv")

st.markdown("---")

with st.expander("Notes & Tips"):
    st.markdown(
        """
        - **Accuracy**: This MVP uses dictionary keyword matching. Improve by expanding the skill dictionary or training a skill NER model.
        - **Privacy**: Do not commit API keys to GitHub. Use environment variables (e.g., set ADZUNA_APP_ID / ADZUNA_APP_KEY) or Streamlit secrets.
        - **Scaling up**: Fetch multiple queries (e.g., "data analyst", "ml engineer") and merge to broaden role coverage.
        - **Deployment**: You can deploy on Streamlit Cloud or any VM. Set environment variables there for credentials.
        - **Portfolio polish**: Add screenshots, example CSVs, and a short demo video/GIF in your README.
        """
    )
