from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

# ✅ IMPORTANT: correct import path
from backend.predict_disease import predict_plant

app = Flask(__name__)

# ✅ CORS (frontend connect fix)
CORS(app)

# ---------------- Q&A DATA ----------------
qa_data = {
    "tomato_early_blight": {
        "en": {"answer": "Caused by fungus. Use fungicide like Mancozeb."},
        "ta": {"answer": "இது பூஞ்சை நோய். Mancozeb போன்ற மருந்து பயன்படுத்தவும்."}
    },
    "tomato_late_blight": {
        "en": {"answer": "Spreads in wet weather. Use copper fungicide."},
        "ta": {"answer": "இது ஈரமான காலநிலையில் பரவும். Copper fungicide பயன்படுத்தவும்."}
    }
}

# ---------------- Q&A Route ----------------
@app.route("/get_answer", methods=["POST"])
def get_answer():
    try:
        data = request.get_json()
        qid = data.get("qid")
        lang = data.get("lang", "en")

        answer = qa_data.get(qid, {}).get(lang, {}).get("answer", "Answer not available")

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- Predict Route ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # ✅ temp file safe path
        file_path = os.path.join(os.getcwd(), "temp.jpg")
        file.save(file_path)

        disease, confidence, affected, reason, organic, chemical = predict_plant(file_path)

        return jsonify({
            "disease": disease,
            "confidence": confidence,
            "affected": affected,
            "reason": reason,
            "organic": organic,
            "chemical": chemical
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- ROOT CHECK ----------------
@app.route("/")
def home():
    return "✅ Agrovision AI Backend Running"


# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)