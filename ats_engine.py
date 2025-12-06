#!/usr/bin/env python3
"""
ATS Resume Scoring System (Enhanced V2)
Evaluates resume against job description with comprehensive scoring and recommendations.

Key features:
- Supports PDF, DOCX, and TXT for both resume & job description
- Better PDF reading (pdfplumber + PyPDF2 + OCR fallback)
- DOCX with paragraphs + tables
- TXT auto-detected and read
- AI-powered similarity (sentence-transformers) with TF-IDF fallback
- Better keyword extraction & skills detection
- Smarter bullet detection (supports ➢, •, -, etc.)
- Grammar check with filters (ignores British vs US variants, ALL CAPS, tech terms, emails)
- Clear, calibrated scoring out of 10
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter
from datetime import datetime

# -----------------------------
# Optional / External Libraries
# -----------------------------

# Document Processing
try:
    import pdfplumber
    from PyPDF2 import PdfReader
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
except ImportError as e:
    print(f"Missing PDF libraries (optional but recommended): {e}")
    print("Install with: pip install pdfplumber PyPDF2 pdf2image pillow pytesseract")

try:
    from docx import Document
except ImportError:
    print("Missing python-docx. Install with: pip install python-docx")

# NLP Processing
try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
except ImportError:
    print("Missing spaCy. Install with: pip install spacy")
    print("Then run: python -m spacy download en_core_web_sm")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Missing scikit-learn. Install with: pip install scikit-learn")

# Optional: better semantic similarity via local embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
    print("✅ sentence-transformers available: using AI embeddings for better similarity.")
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("ℹ Optional: Install sentence-transformers for better AI similarity:")
    print("   pip install sentence-transformers")

try:
    import language_tool_python
except ImportError:
    print("Missing language_tool_python. Install with: pip install language-tool-python")


# ===========================
# Utility functions
# ===========================

def clean_extracted_text(text: str) -> str:
    """
    Clean raw text extracted from PDFs/DOCX/TXT:
    - Fix hyphenation across line breaks (e.g. "devel-\nopment" -> "development")
    - Collapse excessive whitespace
    - Normalize spaces and newlines
    """
    if not text:
        return ""

    # Remove hyphenation at line breaks: word- \n next -> wordnext
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    text = text.replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)  # collapse 3+ newlines to 2
    text = re.sub(r'[ \t]+', ' ', text)    # multiple spaces -> single

    return text.strip()


# ====================================
# DocumentReader with improved handling
# ====================================

class DocumentReader:
    """Handles reading different document formats with multiple fallbacks."""
    
    @staticmethod
    def _read_pdf_with_pdfplumber(filepath: str) -> str:
        """Primary PDF text extraction using pdfplumber."""
        collected = []
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    page_text = page_text.strip()

                    # If page_text is suspiciously empty, try word-based reconstruction
                    if len(page_text) < 20:
                        try:
                            words = page.extract_words()
                            if words:
                                lines_map = {}
                                for w in words:
                                    line_key = round(w.get("top", 0) / 3)
                                    lines_map.setdefault(line_key, []).append(w)
                                
                                line_texts = []
                                for _, line_words in sorted(lines_map.items(), key=lambda kv: kv[0]):
                                    line_words_sorted = sorted(line_words, key=lambda w: w.get("x0", 0))
                                    line_texts.append(" ".join(w["text"] for w in line_words_sorted))
                                page_text = "\n".join(line_texts)
                        except Exception:
                            pass

                    if page_text:
                        collected.append(page_text)
        except Exception as e:
            print(f"pdfplumber failed: {e}")
        return "\n\n".join(collected).strip()

    @staticmethod
    def _read_pdf_with_pypdf2(filepath: str) -> str:
        """Secondary PDF text extraction using PyPDF2."""
        try:
            reader = PdfReader(filepath)
            texts = []
            for page in reader.pages:
                try:
                    txt = page.extract_text() or ""
                    texts.append(txt)
                except Exception:
                    continue
            return "\n\n".join(texts).strip()
        except Exception as e:
            print(f"PyPDF2 failed: {e}")
            return ""

    @staticmethod
    def _read_pdf_with_ocr(filepath: str) -> str:
        """Fallback: OCR using pytesseract on each page image."""
        print("⚠ Detected low text content. Trying OCR on scanned PDF (this may be slow)...")
        try:
            images = convert_from_path(filepath)
            ocr_texts = []
            for i, image in enumerate(images):
                print(f"  OCR processing page {i+1}/{len(images)}...")
                ocr_texts.append(pytesseract.image_to_string(image))
            return "\n\n".join(ocr_texts).strip()
        except Exception as e:
            print(f"❌ OCR failed: {e}")
            print("Make sure Tesseract is installed: https://github.com/tesseract-ocr/tesseract")
            return ""

    @staticmethod
    def read_pdf(filepath: str) -> str:
        """Read PDF file with multiple extraction strategies + OCR fallback."""
        text = DocumentReader._read_pdf_with_pdfplumber(filepath)

        if len(text.strip()) < 200:
            print("ℹ PDF text seems low; trying PyPDF2 as fallback...")
            text2 = DocumentReader._read_pdf_with_pypdf2(filepath)
            if len(text2) > len(text):
                text = text2

        if len(text.strip()) < 150:
            text_ocr = DocumentReader._read_pdf_with_ocr(filepath)
            if len(text_ocr) > len(text):
                text = text_ocr

        return clean_extracted_text(text)

    @staticmethod
    def read_docx(filepath: str) -> str:
        """Read DOCX file, including paragraphs and table cell text."""
        try:
            doc = Document(filepath)
        except Exception as e:
            print(f"❌ Error reading DOCX: {e}")
            return ""

        chunks = []

        for p in doc.paragraphs:
            if p.text:
                chunks.append(p.text)

        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text:
                        row_text.append(cell.text.strip())
                if row_text:
                    chunks.append(" | ".join(row_text))

        text = "\n".join(chunks)
        return clean_extracted_text(text)

    @staticmethod
    def read_txt(filepath: str) -> str:
        """Read plain text file."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return clean_extracted_text(text)
        except Exception as e:
            print(f"❌ Error reading TXT: {e}")
            return ""

    @staticmethod
    def read_document(filepath: str) -> str:
        """Auto-detect and read document."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        ext = Path(filepath).suffix.lower()
        
        if ext == '.pdf':
            return DocumentReader.read_pdf(filepath)
        elif ext == '.docx':
            return DocumentReader.read_docx(filepath)
        elif ext == '.txt':
            return DocumentReader.read_txt(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use PDF, DOCX, or TXT.")


# ===========================
# NLP Processor + Embeddings
# ===========================

class NLPProcessor:
    """Handles NLP operations, keywords, skills, and semantic similarity."""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("❌ spaCy model not found. Run: python -m spacy download en_core_web_sm")
            sys.exit(1)
        
        self.stop_words = STOP_WORDS

        self.embed_model = None
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"⚠ Could not load sentence-transformers model: {e}")
                self.embed_model = None
        
        self.technical_skills = {
            'python', 'java', 'javascript', 'c++', 'c#', 'sql', 'aws', 'azure', 'gcp',
            'docker', 'kubernetes', 'react', 'angular', 'vue', 'nodejs', 'django', 'flask',
            'fastapi', 'spring', 'microservices', 'machine learning', 'deep learning', 'ai',
            'data science', 'analytics', 'pandas', 'numpy', 'scikit-learn', 'pytorch',
            'tensorflow', 'spark', 'hadoop', 'kafka', 'airflow', 'tableau', 'power bi',
            'git', 'ci/cd', 'devops', 'rest', 'graphql', 'api', 'linux', 'bash',
            'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'jira'
        }
        
        self.soft_skills = {
            'leadership', 'communication', 'teamwork', 'problem solving',
            'analytical', 'creative', 'adaptable', 'organized', 'detail-oriented',
            'collaborative', 'strategic', 'innovative', 'motivated', 'ownership',
            'stakeholder management', 'mentoring', 'presentation', 'negotiation'
        }
    
    def extract_keywords(self, text: str, top_n: int = 50) -> List[Tuple[str, float]]:
        """Extract important keywords using frequency + POS filtering."""
        if not text.strip():
            return []

        doc = self.nlp(text.lower())
        phrases = []

        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            if 1 <= len(phrase.split()) <= 4 and phrase not in self.stop_words:
                phrases.append(phrase)

        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'SKILL', 'PERSON']:
                phrases.append(ent.text.lower())

        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and
                token.text.lower() not in self.stop_words and
                len(token.text) > 2 and token.is_alpha):
                phrases.append(token.lemma_.lower())
        
        phrase_freq = Counter(phrases)

        if not phrase_freq:
            # Fallback: simple word frequency
            words = [w.lower() for w in re.findall(r'\w+', text) if w.lower() not in self.stop_words and len(w) > 3]
            phrase_freq = Counter(words)

        return phrase_freq.most_common(top_n)
    
    def extract_skills(self, text: str) -> Dict[str, Set[str]]:
        """Extract technical and soft skills by dictionary matching."""
        text_lower = text.lower()
        found_technical = {s for s in self.technical_skills if s in text_lower}
        found_soft = {s for s in self.soft_skills if s in text_lower}
        return {
            'technical': found_technical,
            'soft': found_soft
        }
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Semantic similarity between two texts."""
        text1 = text1.strip()
        text2 = text2.strip()
        if not text1 or not text2:
            return 0.0

        if self.embed_model is not None:
            try:
                embeddings = self.embed_model.encode([text1, text2], convert_to_tensor=False)
                from numpy import dot
                from numpy.linalg import norm
                v1, v2 = embeddings[0], embeddings[1]
                sim = float(dot(v1, v2) / (norm(v1) * norm(v2) + 1e-9))
                return max(0.0, min(1.0, sim))
            except Exception as e:
                print(f"⚠ Embedding similarity failed, falling back to TF-IDF: {e}")

        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(max(0.0, min(1.0, similarity)))
        except Exception:
            return 0.0


# ======================
# Grammar & Readability
# ======================

TECH_TOKENS = {
    'jira', 'bugzilla', 'mysql', 'mssql', 'aws', 'azure', 'git', 'github', 'gitlab',
    'html', 'css', 'javascript', 'python', 'java', 'sql', 'saas'
}

class GrammarChecker:
    """Check grammar and spelling using LanguageTool; filter noisy hits."""
    
    def __init__(self):
        try:
            self.tool = language_tool_python.LanguageTool('en-US')
        except Exception as e:
            print(f"⚠ Grammar checker not available: {e}")
            print("Install: pip install language-tool-python")
            self.tool = None
    
    def check(self, text: str) -> List[Dict]:
        """Check for grammar and spelling errors with filters."""
        if not self.tool or not text.strip():
            return []
        
        try:
            matches = self.tool.check(text)
            errors = []
            for match in matches[:80]:
                msg = match.message or ""
                # Filter British vs US variants
                if "British English" in msg:
                    continue

                ctx = match.context or ""
                offset = match.offset or 0
                length = match.errorLength or 0
                snippet = ctx[offset:offset + length] if 0 <= offset < len(ctx) else ""

                # Skip ALL CAPS words (headings, names)
                if snippet.isupper() and len(snippet) > 2:
                    continue

                # Skip emails, URLs
                if "@" in snippet or "http" in snippet.lower():
                    continue

                # Skip obvious tech tokens
                if snippet.lower() in TECH_TOKENS:
                    continue

                errors.append({
                    'type': match.ruleId,
                    'message': msg,
                    'context': ctx,
                    'offset': offset,
                    'errorLength': length,
                    'snippet': snippet,
                    'suggestions': match.replacements[:3]
                })
            return errors
        except Exception as e:
            print(f"Grammar check error: {e}")
            return []


def estimate_readability(text: str) -> Dict[str, float]:
    """Rough Flesch Reading Ease estimation."""
    if not text.strip():
        return {"score": 0.0, "label": "Unknown"}
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r'\w+', text)
    if not sentences or not words:
        return {"score": 0.0, "label": "Unknown"}

    num_sentences = len(sentences)
    num_words = len(words)

    vowels = "aeiouy"
    syllables = 0
    for word in words:
        w = word.lower()
        groups = re.findall(r'[aeiouy]+', w)
        syllables += max(1, len(groups))

    words_per_sentence = num_words / num_sentences
    syllables_per_word = syllables / num_words

    score = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word

    if score >= 70:
        label = "Easy to read"
    elif score >= 50:
        label = "Fairly readable"
    elif score >= 30:
        label = "Difficult"
    else:
        label = "Very difficult"

    return {"score": round(score, 1), "label": label}


# ==========================
# Resume Structure Analyzer
# ==========================

class ResumeStructureAnalyzer:
    """Analyze resume structure and formatting."""
    
    REQUIRED_SECTIONS = [
        'experience', 'education', 'skills',
        'work experience', 'employment', 'professional experience'
    ]
    
    CONTACT_PATTERNS = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b',
        r'\blinkedin\.com/in/[A-Za-z0-9\-_/]+\b'
    ]

    BULLET_PREFIXES = ('•', '-', '*', '·', '○', '▪', '➢', '▶', '→', '■', '‣')
    
    def analyze(self, text: str) -> Dict:
        """Analyze resume structure."""
        text_lower = text.lower()
        lines = [l for l in text.split('\n') if l.strip()]
        
        sections_found = []
        for section in self.REQUIRED_SECTIONS:
            if section in text_lower:
                sections_found.append(section)
        
        has_email = bool(re.search(self.CONTACT_PATTERNS[0], text))
        has_phone = bool(re.search(self.CONTACT_PATTERNS[1], text))
        has_linkedin = bool(re.search(self.CONTACT_PATTERNS[2], text.lower()))
        
        bullet_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(self.BULLET_PREFIXES):
                bullet_lines.append(line)
            else:
                # also treat lines with ➢ later in the text as bullets
                if any(b in stripped[:5] for b in self.BULLET_PREFIXES):
                    bullet_lines.append(line)

        has_bullets = len(bullet_lines) > 0
        
        date_pattern = r'\b(19|20)\d{2}\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'
        dates_found = len(re.findall(date_pattern, text, re.IGNORECASE))

        all_caps_lines = [l for l in lines if l.isupper() and len(l.split()) <= 6]

        return {
            'sections_found': list(set(sections_found)),
            'sections_missing': [s for s in ['experience', 'education', 'skills'] 
                                if not any(s in sf for sf in sections_found)],
            'has_contact': {
                'email': has_email,
                'phone': has_phone,
                'linkedin': has_linkedin
            },
            'has_bullets': has_bullets,
            'bullet_count': len(bullet_lines),
            'dates_found': dates_found,
            'has_chronology': dates_found >= 2,
            'all_caps_heading_count': len(all_caps_lines),
            'line_count': len(lines)
        }


# ================
# ATS Scoring Core
# ================

class ATSScorer:
    """Main ATS scoring engine."""
    
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.grammar_checker = GrammarChecker()
        self.structure_analyzer = ResumeStructureAnalyzer()
    
    def score_keyword_matching(self, resume_text: str, jd_text: str) -> Dict:
        """Score: 30% - Keyword matching."""
        jd_keywords_list = self.nlp_processor.extract_keywords(jd_text, 40)
        resume_keywords_list = self.nlp_processor.extract_keywords(resume_text, 80)

        jd_keywords = {k for k, _ in jd_keywords_list}
        resume_keywords = {k for k, _ in resume_keywords_list}
        
        if not jd_keywords:
            # Avoid divide-by-zero; treat as neutral
            return {
                'score': 1.5,  # half of 3.0
                'max_score': 3.0,
                'percentage': 0.0,
                'matched': [],
                'missing': [],
                'total_jd_keywords': 0,
                'total_matched': 0
            }

        matched_keywords = sorted(list(jd_keywords.intersection(resume_keywords)))
        missing_keywords = sorted(list(jd_keywords.difference(resume_keywords)))
        
        match_ratio = len(matched_keywords) / len(jd_keywords)
        score = match_ratio * 3.0
        
        return {
            'score': round(score, 2),
            'max_score': 3.0,
            'percentage': round(match_ratio * 100, 1),
            'matched': matched_keywords[:15],
            'missing': missing_keywords[:15],
            'total_jd_keywords': len(jd_keywords),
            'total_matched': len(matched_keywords)
        }
    
    def score_skills_match(self, resume_text: str, jd_text: str) -> Dict:
        """Score: 20% - Skills matching."""
        resume_skills = self.nlp_processor.extract_skills(resume_text)
        jd_skills = self.nlp_processor.extract_skills(jd_text)
        
        matched_technical = resume_skills['technical'].intersection(jd_skills['technical'])
        missing_technical = jd_skills['technical'] - resume_skills['technical']
        
        matched_soft = resume_skills['soft'].intersection(jd_skills['soft'])
        missing_soft = jd_skills['soft'] - resume_skills['soft']
        
        total_jd_skills = len(jd_skills['technical']) + len(jd_skills['soft'])
        total_matched = len(matched_technical) + len(matched_soft)
        
        match_ratio = total_matched / total_jd_skills if total_jd_skills > 0 else 0.0
        score = match_ratio * 2.0
        
        return {
            'score': round(score, 2),
            'max_score': 2.0,
            'percentage': round(match_ratio * 100, 1),
            'matched_technical': sorted(list(matched_technical)),
            'missing_technical': sorted(list(missing_technical)),
            'matched_soft': sorted(list(matched_soft)),
            'missing_soft': sorted(list(missing_soft))
        }
    
    def score_structure(self, resume_text: str) -> Dict:
        """Score: 15% - Resume structure."""
        structure = self.structure_analyzer.analyze(resume_text)
        
        score = 0.0
        max_score = 1.5
        feedback = []
        
        sections_score = len(structure['sections_found']) / 3 * 0.5
        score += min(0.5, sections_score)
        if structure['sections_missing']:
            feedback.append(f"Missing sections: {', '.join(structure['sections_missing'])}")
        
        contact_count = sum(structure['has_contact'].values())
        contact_score = contact_count / 3 * 0.3
        score += min(0.3, contact_score)
        if contact_count < 3:
            missing = [k for k, v in structure['has_contact'].items() if not v]
            if missing:
                feedback.append(f"Missing contact info: {', '.join(missing)}")
        
        bullet_score = 0.3 if structure['has_bullets'] else 0.0
        score += bullet_score
        if not structure['has_bullets']:
            feedback.append("No bullet points found - use bullets for responsibilities and achievements.")
        
        chrono_score = 0.4 if structure['has_chronology'] else 0.0
        score += chrono_score
        if not structure['has_chronology']:
            feedback.append("Add dates to work experience and education for clear chronology.")
        
        if structure['all_caps_heading_count'] > 30:
            feedback.append("Too many ALL-CAPS headings; keep headings clear and concise.")
        
        return {
            'score': round(score, 2),
            'max_score': max_score,
            'percentage': round((score / max_score) * 100, 1) if max_score > 0 else 0.0,
            'structure': structure,
            'feedback': feedback
        }
    
    def score_grammar(self, resume_text: str) -> Dict:
        """Score: 15% - Grammar and readability."""
        errors = self.grammar_checker.check(resume_text)
        error_count = len(errors)
        
        if error_count == 0:
            score = 1.5
        elif error_count <= 5:
            score = 1.2
        elif error_count <= 10:
            score = 0.9
        elif error_count <= 15:
            score = 0.6
        else:
            score = 0.3
        
        readability = estimate_readability(resume_text)
        
        return {
            'score': round(score, 2),
            'max_score': 1.5,
            'error_count': error_count,
            'errors': errors[:10],
            'feedback': f"Found {error_count} grammar/spelling issues" if error_count > 0 else "Good grammar",
            'readability_score': readability['score'],
            'readability_label': readability['label']
        }
    
    def score_experience_relevance(self, resume_text: str, jd_text: str) -> Dict:
        """Score: 10% - Experience relevance."""
        similarity = self.nlp_processor.calculate_similarity(resume_text, jd_text)
        score = similarity * 1.0
        
        feedback = []
        if similarity < 0.3:
            feedback.append("Resume content doesn't closely match the job requirements. Add more relevant keywords and projects.")
        elif similarity < 0.6:
            feedback.append("Resume has moderate relevance to the job description. Consider tailoring bullet points to the role.")
        else:
            feedback.append("Resume content closely matches the job requirements.")
        
        return {
            'score': round(score, 2),
            'max_score': 1.0,
            'similarity': round(similarity, 3),
            'percentage': round(similarity * 100, 1),
            'feedback': feedback
        }
    
    def score_missing_requirements(self, resume_text: str, jd_text: str) -> Dict:
        """Score: 10% - Missing critical requirements."""
        jd_lower = jd_text.lower()
        resume_lower = resume_text.lower()
        
        required_pattern = r'(?:required|must have|mandatory|must-have)[:\s]+([^\n.]+)'
        required_matches = re.findall(required_pattern, jd_lower)
        
        if not required_matches:
            return {
                'score': 0.8,  # neutral-ish if JD has no explicit "required"
                'max_score': 1.0,
                'missing_count': 0,
                'critical_missing': [],
                'feedback': "No explicit 'must have' requirements detected in JD."
            }
        
        critical_missing = []
        for req in required_matches:
            words = [w for w in re.findall(r'\w+', req) if len(w) > 3 and w not in self.nlp_processor.stop_words]
            found = any(word in resume_lower for word in words[:4])
            if not found:
                critical_missing.append(req.strip()[:80])
        
        missing_count = len(critical_missing)
        if missing_count == 0:
            score = 1.0
        elif missing_count <= 2:
            score = 0.7
        elif missing_count <= 4:
            score = 0.4
        else:
            score = 0.1
        
        return {
            'score': round(score, 2),
            'max_score': 1.0,
            'missing_count': missing_count,
            'critical_missing': critical_missing[:5],
            'feedback': f"Missing {missing_count} critical requirements" if missing_count > 0 else "All critical requirements appear to be covered."
        }
    
    def generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if analysis['keyword_matching']['total_jd_keywords'] > 0 and analysis['keyword_matching']['percentage'] < 70:
            missing = analysis['keyword_matching']['missing'][:7]
            if missing:
                recommendations.append(f"📌 Add or naturally incorporate these job-related keywords (if relevant): {', '.join(missing)}")
        
        skills = analysis['skills_match']
        if skills['missing_technical']:
            recommendations.append(f"🔧 Highlight or add technical skills (if you actually have them): {', '.join(list(skills['missing_technical'])[:5])}")
        if skills['missing_soft']:
            recommendations.append(f"💡 Emphasize soft skills like: {', '.join(list(skills['missing_soft'])[:3])}")
        
        for fb in analysis['structure']['feedback'][:4]:
            recommendations.append(f"📋 {fb}")
        
        if analysis['grammar']['error_count'] > 0:
            recommendations.append(f"✏️ Fix {analysis['grammar']['error_count']} grammar/spelling issues. Use tools like LanguageTool or Grammarly.")
        if analysis['grammar']['readability_score'] < 50:
            recommendations.append(f"📖 Improve readability ({analysis['grammar']['readability_label']}). Use shorter sentences and clear bullet points.")
        
        if analysis['experience_relevance']['similarity'] < 0.5:
            recommendations.append("📝 Tailor your experience bullets to mirror responsibilities and skills in the job description.")
        
        if analysis['missing_requirements']['missing_count'] > 0:
            recommendations.append("⚠️ Explicitly address the 'required' or 'must have' items where you genuinely meet them.")
        
        return recommendations
    
    def calculate_total_score(self, analysis: Dict) -> float:
        """Calculate total ATS score out of 10."""
        total = 0.0
        total += analysis['keyword_matching']['score']
        total += analysis['skills_match']['score']
        total += analysis['structure']['score']
        total += analysis['grammar']['score']
        total += analysis['experience_relevance']['score']
        total += analysis['missing_requirements']['score']
        return round(total, 2)
    
    def analyze_resume(self, resume_text: str, jd_text: str) -> Dict:
        """Complete resume analysis."""
        print("\n🔍 Analyzing resume against job description...")
        
        analysis = {
            'keyword_matching': self.score_keyword_matching(resume_text, jd_text),
            'skills_match': self.score_skills_match(resume_text, jd_text),
            'structure': self.score_structure(resume_text),
            'grammar': self.score_grammar(resume_text),
            'experience_relevance': self.score_experience_relevance(resume_text, jd_text),
            'missing_requirements': self.score_missing_requirements(resume_text, jd_text)
        }
        
        analysis['total_score'] = self.calculate_total_score(analysis)
        analysis['recommendations'] = self.generate_recommendations(analysis)
        
        return analysis


# =====================
# Reporting / Printing
# =====================

def print_analysis(analysis: Dict):
    """Print formatted analysis to console."""
    print("\n" + "="*70)
    print("🎯 ATS RESUME ANALYSIS REPORT")
    print("="*70)
    
    score = analysis['total_score']
    print(f"\n📊 OVERALL ATS SCORE: {score} / 10.0")
    
    if score >= 8.0:
        print("   ✅ EXCELLENT - Strong match!")
    elif score >= 6.5:
        print("   ✓ GOOD - Competitive resume")
    elif score >= 5.0:
        print("   ⚠ FAIR - Needs improvement")
    else:
        print("   ❌ POOR - Major improvements needed")
    
    print("\n📈 SCORE BREAKDOWN:")
    print(f"   Keyword Matching (30%):      {analysis['keyword_matching']['score']:.1f}/3.0 ({analysis['keyword_matching']['percentage']:.1f}%)")
    print(f"   Skills Match (20%):          {analysis['skills_match']['score']:.1f}/2.0 ({analysis['skills_match']['percentage']:.1f}%)")
    print(f"   Resume Structure (15%):      {analysis['structure']['score']:.1f}/1.5 ({analysis['structure']['percentage']:.1f}%)")
    print(f"   Grammar & Readability (15%): {analysis['grammar']['score']:.1f}/1.5  (Readability: {analysis['grammar']['readability_label']})")
    print(f"   Experience Relevance (10%):  {analysis['experience_relevance']['score']:.1f}/1.0 ({analysis['experience_relevance']['percentage']:.1f}%)")
    print(f"   Critical Requirements (10%): {analysis['missing_requirements']['score']:.1f}/1.0")
    
    print("\n🔑 KEYWORDS:")
    print(f"   JD Keywords: {analysis['keyword_matching']['total_jd_keywords']}")
    print(f"   Matched: {analysis['keyword_matching']['total_matched']}/{analysis['keyword_matching']['total_jd_keywords']}")
    if analysis['keyword_matching']['matched']:
        print(f"   Top Matches: {', '.join(analysis['keyword_matching']['matched'][:7])}")
    if analysis['keyword_matching']['missing']:
        print(f"   ⚠ Missing (examples): {', '.join(analysis['keyword_matching']['missing'][:7])}")
    
    print("\n🛠 SKILLS:")
    skills = analysis['skills_match']
    if skills['matched_technical']:
        print(f"   ✓ Technical: {', '.join(skills['matched_technical'][:7])}")
    if skills['missing_technical']:
        print(f"   ✗ Missing Technical (from JD): {', '.join(skills['missing_technical'][:7])}")
    if skills['matched_soft']:
        print(f"   ✓ Soft Skills: {', '.join(skills['matched_soft'][:5])}")
    
    if analysis['grammar']['error_count'] > 0:
        print(f"\n✏️ GRAMMAR ISSUES: {analysis['grammar']['error_count']} found")
        for i, error in enumerate(analysis['grammar']['errors'][:3], 1):
            print(f"   {i}. {error['message'][:80]}  (\"{error['snippet']}\")")
    
    print("\n💡 RECOMMENDATIONS:")
    if not analysis['recommendations']:
        print("   Looks great! Only minor tweaks may be needed.")
    else:
        for i, rec in enumerate(analysis['recommendations'][:10], 1):
            print(f"   {i}. {rec}")
    
    print("\n" + "="*70)


def save_report(analysis: Dict, output_file: str = "ATS_Report.txt"):
    """Save detailed report to text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ATS RESUME ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"OVERALL ATS SCORE: {analysis['total_score']} / 10.0\n\n")
        
        f.write("DETAILED SCORE BREAKDOWN:\n")
        f.write("-" * 70 + "\n")
        
        f.write(f"\n1. KEYWORD MATCHING: {analysis['keyword_matching']['score']}/3.0 ({analysis['keyword_matching']['percentage']:.1f}%)\n")
        f.write(f"   JD Keywords: {analysis['keyword_matching']['total_jd_keywords']}\n")
        f.write(f"   Matched: {analysis['keyword_matching']['total_matched']}/{analysis['keyword_matching']['total_jd_keywords']}\n")
        f.write(f"   Matched Keywords: {', '.join(analysis['keyword_matching']['matched'])}\n")
        f.write(f"   Missing Keywords: {', '.join(analysis['keyword_matching']['missing'])}\n")
        
        f.write(f"\n2. SKILLS MATCH: {analysis['skills_match']['score']}/2.0 ({analysis['skills_match']['percentage']:.1f}%)\n")
        f.write(f"   Technical Skills Matched: {', '.join(analysis['skills_match']['matched_technical'])}\n")
        f.write(f"   Technical Skills Missing: {', '.join(analysis['skills_match']['missing_technical'])}\n")
        f.write(f"   Soft Skills Matched: {', '.join(analysis['skills_match']['matched_soft'])}\n")
        f.write(f"   Soft Skills Missing: {', '.join(analysis['skills_match']['missing_soft'])}\n")
        
        f.write(f"\n3. RESUME STRUCTURE: {analysis['structure']['score']}/1.5 ({analysis['structure']['percentage']:.1f}%)\n")
        f.write(f"   Sections Found: {', '.join(analysis['structure']['structure']['sections_found'])}\n")
        f.write(f"   Sections Missing: {', '.join(analysis['structure']['structure']['sections_missing'])}\n")
        f.write(f"   Contact Info Present: {analysis['structure']['structure']['has_contact']}\n")
        f.write(f"   Bullet Count: {analysis['structure']['structure']['bullet_count']}\n")
        f.write(f"   Chronology Detected: {analysis['structure']['structure']['has_chronology']}\n")
        for fb in analysis['structure']['feedback']:
            f.write(f"   - {fb}\n")
        
        f.write(f"\n4. GRAMMAR & READABILITY: {analysis['grammar']['score']}/1.5\n")
        f.write(f"   Errors Found: {analysis['grammar']['error_count']}\n")
        f.write(f"   Readability: {analysis['grammar']['readability_score']} ({analysis['grammar']['readability_label']})\n")
        for error in analysis['grammar']['errors']:
            f.write(f"   - {error['message']} | Snippet: \"{error['snippet']}\" | Context: {error['context']}\n")
        
        f.write(f"\n5. EXPERIENCE RELEVANCE: {analysis['experience_relevance']['score']}/1.0 ({analysis['experience_relevance']['percentage']:.1f}%)\n")
        for fb in analysis['experience_relevance']['feedback']:
            f.write(f"   {fb}\n")
        
        f.write(f"\n6. CRITICAL REQUIREMENTS: {analysis['missing_requirements']['score']}/1.0\n")
        f.write(f"   {analysis['missing_requirements']['feedback']}\n")
        if analysis['missing_requirements']['critical_missing']:
            f.write("   Missing (examples):\n")
            for item in analysis['missing_requirements']['critical_missing']:
                f.write(f"   - {item}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("ACTIONABLE RECOMMENDATIONS:\n")
        f.write("="*70 + "\n")
        for i, rec in enumerate(analysis['recommendations'], 1):
            f.write(f"{i}. {rec}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✅ Detailed report saved to: {output_file}")


def save_json_report(analysis: Dict, output_file: str = "ATS_Report.json"):
    """Save report as JSON."""
    json_analysis = json.loads(json.dumps(analysis, default=lambda x: list(x) if isinstance(x, set) else x))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_analysis, f, indent=2)
    print(f"✅ JSON report saved to: {output_file}")


# =====================
# File Discovery / Main
# =====================

def find_files() -> Tuple[str, str]:
    """
    Find resume and job description files in current directory.

    Rules:
      - Resume: file starting with 'resume' and ending with .pdf/.docx/.txt
      - JD: file containing 'job' or 'jd' in name, with .pdf/.docx/.txt
    """
    resume_files = []
    jd_files = []
    
    for file in os.listdir('.'):
        lower = file.lower()
        if lower.startswith('resume') and lower.endswith(('.pdf', '.docx', '.txt')):
            resume_files.append(file)
        elif ('job' in lower or 'jd' in lower) and lower.endswith(('.pdf', '.docx', '.txt')):
            jd_files.append(file)
    
    if not resume_files:
        raise FileNotFoundError("No resume file found. Please name it like 'resume.pdf', 'resume.docx' or 'resume.txt'.")
    if not jd_files:
        raise FileNotFoundError("No job description file found. Please name it like 'job_description.pdf', 'job_description.docx', 'jd.txt', etc.")
    
    return resume_files[0], jd_files[0]


def main():
    """Main execution function."""
    print("="*70)
    print("🚀 ATS RESUME SCORING SYSTEM (Enhanced V2)")
    print("="*70)
    
    try:
        print("\n📁 Searching for files in current folder...")
        resume_file, jd_file = find_files()
        print(f"   ✓ Found resume: {resume_file}")
        print(f"   ✓ Found job description: {jd_file}")
        
        print("\n📄 Reading documents...")
        doc_reader = DocumentReader()
        
        print(f"   Reading {resume_file}...")
        resume_text = doc_reader.read_document(resume_file)
        if len(resume_text) < 100:
            print("   ⚠ Warning: Resume text seems very short. Check if the file is scanned-only or empty.")
        print(f"   ✓ Resume loaded ({len(resume_text)} characters)")
        
        print(f"   Reading {jd_file}...")
        jd_text = doc_reader.read_document(jd_file)
        if len(jd_text) < 50:
            print("   ⚠ Warning: Job description text seems very short. If you pasted it, ensure the file is saved correctly.")
        print(f"   ✓ Job description loaded ({len(jd_text)} characters)")
        
        scorer = ATSScorer()
        analysis = scorer.analyze_resume(resume_text, jd_text)
        
        print_analysis(analysis)
        
        print("\n💾 Saving reports...")
        save_report(analysis, "ATS_Report.txt")
        save_json_report(analysis, "ATS_Report.json")
        
        print("\n✅ Analysis complete!")
        print("📊 Check ATS_Report.txt for detailed human-readable analysis")
        print("📊 Check ATS_Report.json for machine-readable format")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n📋 Expected file structure:")
        print("   your_folder/")
        print("   ├── ats_resume_scorer.py")
        print("   ├── resume.pdf (or resume.docx / resume.txt)")
        print("   └── job_description.pdf (or job_description.docx / jd.txt)")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
