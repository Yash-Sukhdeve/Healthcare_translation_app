from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import openai
import os
import tempfile
import logging
from datetime import datetime

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
def initialize_openai_client():
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return openai.OpenAI(api_key=openai_api_key)

client = initialize_openai_client()

def medical_transcription(audio_file_path: str) -> str:
    """Transcribe and refine medical audio"""
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        prompt = (
            "Refine this medical transcript for accuracy, "
            "correcting medical terminology and errors:\n\n" 
            f"{transcription}\n\n"
            "Provide only the refined transcript."
        )

        refinement = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return refinement.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise

def translate_text(text: str, target_language: str) -> str:
    """Translate text to target language"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": f"Translate this to {target_language} precisely, "
                              "maintaining medical terminology accuracy."
                },
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=["POST"])
def upload():
    if "audio_data" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio_data"]
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = f"{tempfile.gettempdir()}/audio_{timestamp}.webm"
        audio_file.save(temp_path)
        
        transcript = medical_transcription(temp_path)
        return jsonify({
            "transcript": transcript,
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": "Audio processing failed",
            "details": str(e)
        }), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/translate', methods=["POST"])
def translate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        text = data.get("text", "").strip()
        target_language = data.get("targetLanguage", "").strip()
        
        if not text:
            return jsonify({"error": "Missing text to translate"}), 400
        if not target_language:
            return jsonify({"error": "Missing target language"}), 400
        
        translated_text = translate_text(text, target_language)
        return jsonify({
            "translatedText": translated_text,
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": "Translation failed",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))