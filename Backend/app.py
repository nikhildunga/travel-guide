import base64
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request


load_dotenv(Path(__file__).with_name(".env"))

app = Flask(__name__)

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MURF_API_URL = "https://global.api.murf.ai/v1/speech/stream"
REQUEST_TIMEOUT = (10, 120)
GEMINI_MAX_ATTEMPTS = 3
GEMINI_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
ALLOWED_ORIGINS = {
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
}

MURF_ERROR_MESSAGES = {
    400: "Murf rejected the speech request. Check the text, voice ID, and locale.",
    402: "Murf requires an active payment method or additional credits.",
    403: "Murf rejected the API key or access to the requested voice.",
    500: "Murf encountered an internal error while generating speech.",
    503: "Murf speech generation is temporarily unavailable.",
}


class GeminiAPIError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Vary"] = "Origin"
    return response


def generate_description(place, answer_type, language):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini is not configured. Set GEMINI_API_KEY in Backend/.env.")

    model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    prompt = (
        f"Write a {answer_type.lower()} travel audio-guide description of {place}. "
        f"Respond in {language}. Make it engaging, accurate, and suitable for listening."
    )
    for attempt in range(GEMINI_MAX_ATTEMPTS):
        try:
            response = requests.post(
                GEMINI_API_URL.format(model=model),
                params={"key": api_key},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.Timeout as exc:
            if attempt == GEMINI_MAX_ATTEMPTS - 1:
                raise TimeoutError("Gemini timed out while generating the travel description.") from exc
        except requests.exceptions.ConnectionError as exc:
            if attempt == GEMINI_MAX_ATTEMPTS - 1:
                raise ConnectionError("Could not connect to Gemini to generate the travel description.") from exc
        except requests.exceptions.RequestException as exc:
            raise ConnectionError("Gemini could not complete the travel-description request.") from exc
        else:
            if response.status_code not in GEMINI_RETRYABLE_STATUSES or attempt == GEMINI_MAX_ATTEMPTS - 1:
                break

        time.sleep(attempt + 1)

    if not response.ok:
        status_code = response.status_code
        if status_code in GEMINI_RETRYABLE_STATUSES:
            message = "Gemini is temporarily busy. Please try again shortly."
        elif status_code in (400, 401, 403):
            message = "Gemini rejected the request. Check the API key and model configuration."
        else:
            message = "Gemini could not generate the travel description."
        raise GeminiAPIError(message, status_code)

    try:
        result = response.json()
        description = result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Gemini returned an invalid travel description.") from exc
    if not description:
        raise RuntimeError("Gemini returned an empty travel description.")
    return description


def generate_speech(text, voice_id, locale):
    api_key = os.getenv("MURF_API_KEY")
    if not api_key:
        raise RuntimeError("Murf is not configured. Set MURF_API_KEY in Backend/.env.")

    try:
        response = requests.post(
            MURF_API_URL,
            headers={"api-key": api_key, "Content-Type": "application/json"},
            json={
                "text": text,
                "voiceId": voice_id,
                "model": "falcon-2",
                "locale": locale,
                "format": "MP3",
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout as exc:
        raise TimeoutError("Murf timed out while generating speech.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError("Could not connect to Murf to generate speech.") from exc
    except requests.exceptions.RequestException as exc:
        raise ConnectionError("Murf could not complete the speech request.") from exc

    if not response.ok:
        message = MURF_ERROR_MESSAGES.get(
            response.status_code,
            "Murf could not generate speech for this request.",
        )
        raise MurfAPIError(message, response.status_code)
    if not response.content:
        raise MurfAPIError("Murf returned an empty audio response.", 502)
    return response.content


class MurfAPIError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


@app.post("/generate-audio-guide")
def generate_audio_guide():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Request body must be a JSON object."), 400

    required_fields = ("place", "answerType", "language", "voiceId", "locale")
    values = {}
    for field in required_fields:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            return jsonify(error=f"'{field}' is required and must be a non-empty string."), 400
        values[field] = value.strip()

    try:
        description = generate_description(
            values["place"],
            values["answerType"],
            values["language"],
        )
        audio_bytes = generate_speech(
            description,
            values["voiceId"],
            values["locale"],
        )
    except TimeoutError as exc:
        return jsonify(error=str(exc)), 504
    except ConnectionError as exc:
        return jsonify(error=str(exc)), 502
    except GeminiAPIError as exc:
        status_code = exc.status_code if exc.status_code in (400, 401, 403, 429, 500, 502, 503) else 502
        return jsonify(error=str(exc)), status_code
    except MurfAPIError as exc:
        status_code = exc.status_code if exc.status_code in (400, 402, 403, 500, 503) else 502
        return jsonify(error=str(exc)), status_code
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 500

    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    return jsonify(description=description, audio=audio_base64)



if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)