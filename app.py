from flask import Flask, render_template, request, jsonify
import openai
import os
import tempfile
import mysecret as mysecret

app = Flask(__name__)
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
    openai_api_key = mysecret.OPENAI_API_KEY
client = openai.OpenAI(api_key=openai_api_key)


def medical_transcription(audio_file_path: str) -> str:
    """
    Transcribes an audio file using OpenAI's Whisper API (gpt-4o-transcribe) 
    and refines the transcript for medical accuracy.
    """
    # Step 1: Transcribe audio using the Whisper API
    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file
        )
    raw_transcript = transcription.text

    # Step 2: Refine the transcript with a chat prompt
    prompt = (
        "Please refine the following transcript for accuracy, "
        "especially ensuring that any medical terminology is correct, "
        "and fix any transcription errors:\n\n" + raw_transcript
    )

    refinement_response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or another suitable chat model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.3,
        n=1
    )

    refined_transcript = refinement_response.choices[0].message.content.strip()
    return refined_transcript

def translate_text(text: str, target_language: str) -> str:
    """
    Translates the given text into the target language using a chat-based approach.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Translate the following text into {target_language}:"},
            {"role": "user", "content": text}
        ],
        temperature=0.3,
        max_tokens=500,
        n=1
    )
    translation = response.choices[0].message.content.strip()
    return translation

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=["POST"])
def upload():
    """
    Receives the recorded audio file from the front end, saves it temporarily,
    transcribes and refines it, and returns the refined transcript as JSON.
    """
    if "audio_data" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio_data"]
    fd, temp_path = tempfile.mkstemp(suffix=".webm")
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(audio_file.read())
        transcript = medical_transcription(temp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            import time
            time.sleep(0.1)
            os.remove(temp_path)
        except Exception as remove_error:
            print("Error removing temporary file:", remove_error)
    return jsonify({"transcript": transcript})

@app.route('/translate', methods=["POST"])
def translate():
    """
    Receives a JSON payload containing the transcript and target language,
    translates the transcript, and returns the translated text.
    """
    data = request.get_json()
    text = data.get("text", "")
    target_language = data.get("targetLanguage", "")
    if not text or not target_language:
        return jsonify({"error": "Missing text or target language"}), 400
    try:
        translated_text = translate_text(text, target_language)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"translatedText": translated_text})

if __name__ == '__main__':
    app.run(debug=True)

