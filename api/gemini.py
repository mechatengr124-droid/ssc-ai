import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import httpx
import google.generativeai as genai
from flask import Flask, request, Response, jsonify, stream_with_context

app = Flask(__name__)

# ── MOCK MODE (set to True to bypass Gemini) ────────────────
MOCK_MODE = True   # Change to False to use real API

# ── API Keys (only used if MOCK_MODE is False) ──────────────
API_KEYS = [k.strip() for k in os.environ.get('GEMINI_API_KEYS', '').split(',') if k.strip()]
PRIMARY_KEY = os.environ.get('GEMINI_API_KEY', '')
ALL_KEYS = list(set([PRIMARY_KEY] + API_KEYS if PRIMARY_KEY else API_KEYS))

FALLBACK_MODELS = ['gemini-2.5-flash', 'gemini-2.0-flash-lite', 'gemini-2.0-flash', 'gemini-1.5-flash']
REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT_MS', 30000)) / 1000.0
SEARCH_API_KEY = os.environ.get('SEARCH_API_KEY', '')
SEARCH_ENGINE_ID = os.environ.get('SEARCH_ENGINE_ID', '')

# ── Mock Generator ────────────────────────────────────────────
def mock_generator():
    """Yields a test message to verify streaming."""
    chunks = [
        "Hello from ",
        "SSC AI ",
        "backend! ",
        "This is a test stream. ",
        "If you see this, the SSE pipeline works. "
        "Now we can debug the real Gemini calls."
    ]
    for chunk in chunks:
        yield chunk
        time.sleep(0.1)  # simulate streaming

# ── Health Check ──────────────────────────────────────────────
def health_check():
    return {
        'status': 'ok',
        'timestamp': datetime.utcnow().isoformat(),
        'keys': len(ALL_KEYS),
        'models': FALLBACK_MODELS,
        'mock_mode': MOCK_MODE,
    }

# ── Flask Routes ──────────────────────────────────────────────
@app.route('/api/gemini', methods=['GET', 'POST', 'OPTIONS'])
def handler():
    if request.method == 'OPTIONS':
        return '', 200

    if request.method == 'GET':
        return jsonify(health_check())

    if request.method == 'POST':
        # Use mock if enabled
        if MOCK_MODE:
            def generate_mock():
                full_text = ''
                for chunk in mock_generator():
                    full_text += chunk
                    payload = {'candidates': [{'content': {'parts': [{'text': full_text}]}}]}
                    yield f"data: {json.dumps(payload)}\n\n"
                    time.sleep(0.05)
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(generate_mock()), mimetype='text/event-stream')

        # ── Real Gemini code (only runs if MOCK_MODE is False) ──
        # (Keep your existing generation logic here – we'll restore later)
        # For now, we'll just return a fallback.
        return jsonify({'error': 'Real Gemini not implemented in debug mode'}), 501

# ── Vercel entry point ────────────────────────────────────────
