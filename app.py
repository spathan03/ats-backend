from flask import Flask, request, jsonify
from ats_engine import extract_text, calculate_ats_score
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/score", methods=["POST"])
def score():
    resume = request.files["resume"]
    jd = request.files["jd"]

    resume_path = os.path.join(UPLOAD_FOLDER, resume.filename)
    jd_path = os.path.join(UPLOAD_FOLDER, jd.filename)

    resume.save(resume_path)
    jd.save(jd_path)

    resume_text = extract_text(resume_path)
    jd_text = extract_text(jd_path)

    result = calculate_ats_score(resume_text, jd_text)

    return jsonify({
        "status": "success",
        "result": result
    })


@app.route("/")
def home():
    return {"message": "ATS API working!"}


if __name__ == "__main__":
    app.run(debug=True)
