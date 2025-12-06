from flask import Flask, request, jsonify
from ats import ATSScorer
import tempfile
import os

app = Flask(__name__)

scorer = ATSScorer()

@app.route("/")
def home():
    return {"status": "ATS API running successfully on Render!"}

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "resume" not in request.files or "jd" not in request.files:
            return jsonify({"error": "Upload both resume and job description"}), 400

        resume_file = request.files["resume"]
        jd_file = request.files["jd"]

        # Save temp files
        with tempfile.NamedTemporaryFile(delete=False) as r:
            resume_path = r.name
            resume_file.save(resume_path)

        with tempfile.NamedTemporaryFile(delete=False) as j:
            jd_path = j.name
            jd_file.save(jd_path)

        # Read files using your ATS engine
        from ats import DocumentReader
        resume_text = DocumentReader.read_document(resume_path)
        jd_text = DocumentReader.read_document(jd_path)

        analysis = scorer.analyze_resume(resume_text, jd_text)

        # Delete temp files
        os.remove(resume_path)
        os.remove(jd_path)

        return jsonify(analysis)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
