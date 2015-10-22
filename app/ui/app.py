# app/ui/app.py
"""
app.py

Flask UI for the Clinical Eligibility Tool.

Purpose:
- Provide a local browser-based interface
- Trigger execution of the eligibility engine
- Display basic run status

This file intentionally keeps logic minimal.
"""

from pathlib import Path

from flask import Flask, render_template, redirect, url_for

from app.run_engine import run_study

app = Flask(__name__)

# Paths are relative to repository root
BASE_DIR = Path(__file__).resolve().parents[2]
STUDY_CONFIG_PATH = BASE_DIR / "app" / "engine" / "config" / "study_config.yaml"
OUTPUT_DIR = BASE_DIR / "outputs"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run")
def run():
    result = run_study(
        study_config_path=STUDY_CONFIG_PATH,
        output_dir=OUTPUT_DIR,
    )
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

