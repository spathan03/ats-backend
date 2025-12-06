import pdfplumber
from docx import Document
from fuzzywuzzy import fuzz

def extract_text(path):
    if path.endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])

    if path.endswith(".docx"):
        doc = Document(path)
        return "\n".join([para.text for para in doc.paragraphs])

    return ""
    

def calculate_ats_score(resume_text, jd_text):
    resume_words = set(resume_text.lower().split())
    jd_words = set(jd_text.lower().split())

    matched = resume_words.intersection(jd_words)

    keyword_score = round((len(matched) / len(jd_words)) * 100, 2) if jd_words else 0
    similarity = fuzz.ratio(resume_text, jd_text)

    final_score = round((keyword_score * 0.6) + (similarity * 0.4), 2)

    return {
        "final_score": final_score,
        "similarity": similarity,
        "keyword_score": keyword_score,
        "matched_keywords": list(matched)
    }
