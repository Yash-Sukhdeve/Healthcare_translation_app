from flask import Flask, render_template, request, jsonify
import openai
import os
import tempfile

app = Flask(__name__)

# Set your OpenAI API key (ensure this is set in your environment)
openai.api_key = os.environ.get("OPENAI_API_KEY")

def medical_transcription(audio_file_path: str) -> str:
    """
    Transcribes an audio file using OpenAI's Whisper API and then refines the transcript
    for accuracy with an emphasis on correct medical terminology.
    """
    # Step 1: Transcribe using Whisper API
    with open(audio_file_path, "rb") as audio_file:
        transcript_response = openai.Audio.transcribe("whisper-1", audio_file)
    raw_transcript = transcript_response.get("text", "")
    
    # Step 2: Refine the transcript with a custom prompt
    prompt = (
        "Please refine the following transcript for accuracy, "
        "especially ensuring that any medical terminology is correct, "
        "and fix any transcription errors:\n\n" + raw_transcript
    )
    
    refinement_response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.3,
        n=1
    )
    refined_transcript = refinement_response.choices[0].text.strip()
    return refined_transcript

def translate_text(text: str, target_language: str) -> str:
    """
    Translates the given text into the target language using a prompt-based approach.
    """
    prompt = f"Translate the following text into {target_language}: {text}"
    translation_response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0.5,
        n=1
    )
    translated_text = translation_response.choices[0].text.strip()
    return translated_text

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=["POST"])
def upload():
    """
    Receives the recorded audio file from the front end, saves it temporarily,
    and returns the refined transcript as JSON.
    """
    if "audio_data" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio_data"]
    
    # Create a temporary file using mkstemp.
    fd, temp_path = tempfile.mkstemp(suffix=".webm")
    try:
        # Write the audio data to the temporary file and close the file descriptor.
        with os.fdopen(fd, 'wb') as f:
            f.write(audio_file.read())
        
        # Now call your transcription function with the temporary file path.
        transcript = medical_transcription(temp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            # Optionally, add a small delay to ensure the OS releases the file.
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
    calls the translation function, and returns the translated text.
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
