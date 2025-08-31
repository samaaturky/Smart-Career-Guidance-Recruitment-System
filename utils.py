import os
import re
import numpy as np
import joblib
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO

# ------------------------
# Text cleaning
# ------------------------
def cleanresume(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    txt = txt.lower()
    txt = re.sub(r"http\S+|www\S+", " ", txt)
    txt = re.sub(r"\S+@\S+", " ", txt)
    txt = re.sub(r"[^a-z\s+#.]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


# ------------------------
# Extract text from Streamlit UploadedFile
# ------------------------
def extract_text_from_cv_filelike(upload) -> str:
    name = upload.name
    ext = os.path.splitext(name)[1].lower()
    data = upload.read()
    try:
        upload.seek(0)
    except Exception:
        pass

    if ext == ".pdf":
        reader = PdfReader(BytesIO(data))
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + " "
        return text

    elif ext == ".docx":
        doc = Document(BytesIO(data))
        return " ".join(p.text for p in doc.paragraphs)

    else:  # txt fallback
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""


# ------------------------
# Load model + vectorizer
# ------------------------
def load_artifacts(model_path="career_path_svm.pkl", vec_path="tfidf.pkl"):
    clf = joblib.load(model_path)
    vec = joblib.load(vec_path)
    return clf, vec


# ------------------------
# Predict Top-k categories
# ------------------------
def predict_topk(text: str, clf, vec, top_k: int = 3):
    X = vec.transform([text])
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
        classes = clf.classes_
        idx = np.argsort(probs)[::-1][:top_k]
        return [(classes[i], float(probs[i])) for i in idx]
    else:
        scores = clf.decision_function(X).ravel()
        classes = clf.classes_
        idx = np.argsort(scores)[::-1][:top_k]
        s = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return [(classes[i], float(s[i])) for i in idx]


# ------------------------
# Skills extraction
# ------------------------
BASIC_SKILLS = sorted({
    "python","r","sql","excel","power bi","tableau",
    "pandas","numpy","matplotlib","scikit-learn","sklearn","tensorflow","pytorch",
    "nlp","computer vision","deep learning","machine learning",
    "svm","random forest","xgboost","lightgbm","logistic regression",
    "git","github","docker","linux","bash",
    "etl","airflow","spark","hadoop",
    "flask","django","fastapi","rest api",
    "aws","azure","gcp","bigquery","s3","ec2",
    "html","css","javascript","react",
})

def extract_user_skills(raw_text: str) -> set:
    t = raw_text.lower()
    found = set()
    for skill in BASIC_SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", t):
            found.add(skill)
    if "scikit-learn" in found: found.add("sklearn")
    if "sklearn" in found: found.add("scikit-learn")
    return found


def suggest_missing_skills(predicted_category: str, user_skills: set) -> list:
    tracks = {
        "data science": {"python","sql","pandas","numpy","scikit-learn","matplotlib","nlp","ml","git","docker"},
        "data analyst": {"sql","excel","power bi","tableau","python","pandas","statistics"},
        "machine learning engineer": {"python","numpy","pandas","scikit-learn","tensorflow","pytorch","docker","aws","ml","deep learning"},
        "web developer": {"html","css","javascript","react","django","flask","git"},
        "big data engineer": {"spark","hadoop","aws","airflow","etl","python","sql"},
    }
    target = predicted_category.lower()
    target_skills = set()
    for k, v in tracks.items():
        if k in target:
            target_skills = v
            break
    if not target_skills:
        target_skills = {"python","sql","pandas","numpy","scikit-learn","git"}
    return sorted(list(target_skills - user_skills))


# ------------------------
# Unified recommend_jobs
# ------------------------
def recommend_jobs(input_data, clf, vec, top_k=3):
    """
    Accepts either raw text (str) or a Streamlit UploadedFile.
    Returns predictions and extracted text.
    """
    if hasattr(input_data, "read"):  # file upload
        cv_text = extract_text_from_cv_filelike(input_data)
    else:  # assume string
        cv_text = str(input_data)

    cleaned_text = cleanresume(cv_text)
    preds = predict_topk(cleaned_text, clf, vec, top_k=top_k)
    return preds, cv_text
