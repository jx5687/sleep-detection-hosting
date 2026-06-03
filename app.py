import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Log drowsiness events (optional persistence)
LOG_FILE = os.environ.get("LOG_FILE", "drowsiness_log.jsonl")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/log_event", methods=["POST"])
def log_event():
    """
    Terima event dari frontend (misal: DROWSY detected)
    dan simpan ke log file.
    """
    try:
        data = request.get_json(silent=True) or {}
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": data.get("event", "unknown"),
            "ear_left": data.get("ear_left"),
            "ear_right": data.get("ear_right"),
            "perclos": data.get("perclos"),
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/get_logs")
def get_logs():
    """Ambil 50 log terakhir."""
    try:
        if not os.path.exists(LOG_FILE):
            return jsonify({"logs": []})
        with open(LOG_FILE) as f:
            lines = f.readlines()
        logs = [json.loads(l) for l in lines[-50:] if l.strip()]
        return jsonify({"logs": logs[::-1]})
    except Exception as e:
        return jsonify({"logs": [], "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)