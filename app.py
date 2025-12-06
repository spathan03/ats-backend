from flask import Flask, request, jsonify
from flask_cors import CORS
from ats_engine import extract_text, calculate_ats_score
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/score", methods=["POST"])
def score():
    try:
        resume = request.files.get("resume")
        jd = request.files.get("jd")

        if not resume or not jd:
            return jsonify({"error": "Resume or JD missing"}), 400

        resume_path = os.path.join(UPLOAD_FOLDER, resume.filename)
        jd_path = os.path.join(UPLOAD_FOLDER, jd.filename)

        resume.save(resume_path)
        jd.save(jd_path)

        resume_text = extract_text(resume_path)
        jd_text = extract_text(jd_path)

        result = calculate_ats_score(resume_text, jd_text)

        return jsonify({"status": "success", "result": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/")
def home():
    return jsonify({"message": "ATS API working!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
