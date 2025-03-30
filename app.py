from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import openai
import os
import tempfile
import logging
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI Client Configuration
def initialize_openai_client():
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return openai.OpenAI(api_key=openai_api_key)

client = initialize_openai_client()

def medical_transcription(audio_file_path: str) -> str:
    """
    Transcribes and refines an audio file for medical accuracy.
    Returns refined transcript or raises exception.
    """
    try:
        # Step 1: Transcribe audio
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",  # Updated to current Whisper model
                file=audio_file,
                response_format="text"
            )
        raw_transcript = transcription

        # Step 2: Refine transcript
        prompt = (
            "Refine this medical transcript for accuracy, "
            "correcting any medical terminology and transcription errors:\n\n" 
            f"{raw_transcript}\n\n"
            "Provide only the refined transcript with no additional commentary."
        )

        refinement_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.3
        )

        return refinement_response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise

def translate_text(text: str, target_language: str) -> str:
    """
    Translates text to target language.
    Returns translation or raises exception.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": f"Translate this accurately to {target_language}. "
                              "Maintain medical terminology precision."
                },
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise

@app.route('/')
def index():
    """Serve the main application page."""
    return render_template("index.html")

@app.route('/upload', methods=["POST"])
def upload():
    """Handle audio upload and transcription."""
    # Validate request
    if "audio_data" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio_data"]
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Process file
    try:
        # Create temp file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = f"{tempfile.gettempdir()}/audio_{timestamp}.webm"
        
        audio_file.save(temp_path)
        logger.info(f"Saved temporary file: {temp_path}")
        
        # Transcribe and refine
        transcript = medical_transcription(temp_path)
        logger.info("Successfully generated transcript")
        
        return jsonify({
            "transcript": transcript,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Upload processing error: {str(e)}")
        return jsonify({
            "error": "Failed to process audio",
            "details": str(e)
        }), 500
        
    finally:
        # Clean up temp file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.info(f"Removed temporary file: {temp_path}")
        except Exception as e:
            logger.error(f"Error removing temp file: {str(e)}")

@app.route('/translate', methods=["POST"])
def translate():
    """Handle translation requests."""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        text = data.get("text", "").strip()
        target_language = data.get("targetLanguage", "").strip()
        
        if not text:
            return jsonify({"error": "Missing text to translate"}), 400
        if not target_language:
            return jsonify({"error": "Missing target language"}), 400
        
        # Perform translation
        translated_text = translate_text(text, target_language)
        logger.info(f"Translated to {target_language}")
        
        return jsonify({
            "translatedText": translated_text,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify({
            "error": "Translation failed",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))