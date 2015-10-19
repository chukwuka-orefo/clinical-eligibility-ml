# app/ui/app.py
"""
app.py

Minimal Flask application skeleton for Clinical Eligibility ML.

Purpose:
- Establish local browser-based UI entry point
- Provide a foundation for later engine integration

This file intentionally does NOT:
- run the eligibility engine
- load study configuration
- perform any data processing
"""

from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    # Local development entry point
    app.run(host="127.0.0.1", port=5000, debug=False)
