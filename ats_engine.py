#!/usr/bin/env python3
"""
ATS Resume Scoring Engine
Supports PDF, DOCX, TXT + AI similarity + keyword match
"""

import pdfplumber
import docx
import re
from difflib import SequenceMatcher

def read_pdf(path):
    try:
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    except:
        return ""

def read_docx(path):
    try:
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except:
        return ""

def read_txt(path):
    try:
        return open(path, "r", encoding="utf-8").read()
    except:
        return ""

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return read_pdf(file_path)

    elif file_path.endswith(".docx"):
        return read_docx(file_path)

    elif file_path.endswith(".txt"):
        return read_txt(file_path)

    return ""


def similarity(a, b):
    return round(SequenceMatcher(None, a, b).ratio() * 100, 2)


def keyword_match(resume, jd):
    r_words = set(re.findall(r"\w+", resume.lower()))
    j_words = set(re.findall(r"\w+", jd.lower()))

    matches = r_words.intersection(j_words)
    score = round((len(matches) / len(j_words)) * 100, 2)

    return score, list(matches)


def calculate_ats_score(resume_text, jd_text):
    sim = similarity(resume_text, jd_text)
    key_score, keywords = keyword_match(resume_text, jd_text)

    final_score = round((sim * 0.6) + (key_score * 0.4), 2)

    return {
        "similarity": sim,
        "keyword_score": key_score,
        "final_score": final_score,
        "matched_keywords": keywords,
    }
