# Healthcare Translation Application

## Overview

The Healthcare Translation Application is designed to facilitate multilingual communication in medical consultations. It records audio, generates accurate medical transcripts, translates content into multiple languages, and provides audio playback of translations.

---

## Key Features

- **Medical Audio Recording:** Easily capture patient-provider conversations.
- **AI-Enhanced Transcription:** Accurate speech-to-text with OpenAI Whisper, refined for medical terminology using GPT-3.5-turbo.
- **Real-Time Translation:** Translate transcripts into multiple languages with GPT-3.5-turbo.
- **Text-to-Speech Playback:** Hear translated transcripts through integrated audio playback.
- **Responsive Design:** Mobile and desktop-friendly UI built with Bootstrap.

---

## Technologies

- **Frontend:** HTML5, CSS3 (Bootstrap), JavaScript
- **Backend:** Python, Flask
- **AI Services:** OpenAI Whisper, GPT-3.5-turbo
- **Deployment:** Vercel Serverless Functions

---

## Project Structure

```
healthcare_translation_app/
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
└── vercel.json
```

---

## Setup and Installation

### Requirements

- Python 3.9+
- OpenAI API Key

### Installation Steps

Clone the repository:

```bash
git clone <repository-url>
cd healthcare_translation_app
```

Create a virtual environment and activate it:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Unix/macOS
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your OpenAI API Key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'

# Windows (PowerShell)
$env:OPENAI_API_KEY='your-api-key-here'
```

Run the Flask server:

```bash
flask run
```

Visit the application:

```
http://127.0.0.1:5000
```

---

## Deployment

Deploy easily with Vercel:

Install Vercel CLI:

```bash
npm install -g vercel
```

Log in and deploy:

```bash
vercel login
vercel --prod
```

Ensure environment variables (`OPENAI_API_KEY`) are set in Vercel dashboard.

---

## Security Considerations

- No permanent storage of audio or transcripts.
- All API keys are managed securely via environment variables.
- CORS and file validation are properly configured.

---


