from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import openai
import os
import tempfile
import logging
import time
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__, template_folder='templates')

# Vercel-specific middleware configuration
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
CORS(app, resources={
    r"/upload": {"origins": "*"},
    r"/translate": {"origins": "*"}
})

# Configure logging for Vercel
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vercel-compatible OpenAI initialization
def initialize_openai_client():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise RuntimeError("OpenAI API key not configured")
    
    return openai.OpenAI(
        api_key=api_key,
        timeout=30.0,
        max_retries=3
    )

client = initialize_openai_client()

def medical_transcription(audio_file_path: str) -> str:
    """Full transcription pipeline with refinement"""
    try:
        # Step 1: Transcribe with Whisper
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        # Step 2: Refine with GPT-3.5
        refinement = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"Refine this medical transcript for accuracy, correcting terminology:\n\n{transcription}"
            }],
            temperature=0.3
        )
        
        return refinement.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio_data' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['audio_data']
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    temp_path = None
    try:
        # Create temp file in Vercel's /tmp directory
        temp_path = os.path.join('/tmp', f"audio_{int(time.time())}.webm")
        file.save(temp_path)
        
        # Process transcription
        transcript = medical_transcription(temp_path)
        return jsonify({
            "transcript": transcript,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "solution": "Try again or check console for details"
        }), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass

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
        
        translated_text = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": f"Translate this to {target_language} precisely, maintaining medical terminology."
                },
                {"role": "user", "content": text}
            ],
            temperature=0.3
        ).choices[0].message.content.strip()

        return jsonify({
            "translatedText": translated_text,
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": "Translation failed",
            "details": str(e)
        }), 500

# Vercel requires this handler
handler = app