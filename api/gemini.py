import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import httpx
import google.generativeai as genai
from flask import Flask, request, Response, jsonify, stream_with_context

# ── Configuration ─────────────────────────────────────────────

app = Flask(__name__)

# API keys
API_KEYS = [k.strip() for k in os.environ.get('GEMINI_API_KEYS', '').split(',') if k.strip()]
PRIMARY_KEY = os.environ.get('GEMINI_API_KEY', '')
ALL_KEYS = list(set([PRIMARY_KEY] + API_KEYS if PRIMARY_KEY else API_KEYS))

# Hybrid settings
USE_SDK_FIRST = os.environ.get('USE_SDK_FIRST', 'true').lower() != 'false'
SDK_FALLBACK = os.environ.get('SDK_FALLBACK', 'true').lower() != 'false'
USE_FETCH_FIRST = os.environ.get('USE_FETCH_FIRST', 'false').lower() == 'true'

# Models
FALLBACK_MODELS = [
    'gemini-2.5-flash',
    'gemini-2.0-flash-lite',
    'gemini-2.0-flash',
    'gemini-1.5-flash',
]

# Timeout
REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT_MS', 30000)) / 1000.0

# Web Search
SEARCH_API_KEY = os.environ.get('SEARCH_API_KEY', '')
SEARCH_ENGINE_ID = os.environ.get('SEARCH_ENGINE_ID', '')

# ── Web Search (synchronous) ─────────────────────────────────

def perform_web_search(query: str) -> Optional[str]:
    if not SEARCH_API_KEY or not SEARCH_ENGINE_ID:
        return None
    url = f"https://www.googleapis.com/customsearch/v1?key={SEARCH_API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&num=5"
    with httpx.Client(timeout=8.0) as client:
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                return None
            data = resp.json()
            items = data.get('items', [])
            if not items:
                return None
            results = []
            for i, item in enumerate(items, 1):
                results.append(f"[{i}] {item['title']}\n{item.get('snippet', '')}\nURL: {item['link']}")
            return '\n\n'.join(results)
        except Exception as e:
            print(f"⚠️ Web search failed: {e}")
            return None

# ── Error Detection ────────────────────────────────────────────

def is_quota_error(error: Exception) -> bool:
    msg = str(error).lower()
    return any(k in msg for k in ['429', 'quota', 'rate limit', 'resource exhausted', 'billing'])

def is_network_error(error: Exception) -> bool:
    msg = str(error).lower()
    return any(k in msg for k in ['timeout', 'connection', 'econnrefused', 'econnreset', 'socket', 'dns'])

# ── SDK Generation (synchronous streaming) ────────────────────

def generate_with_sdk(contents: List[Dict], model: str, config: Dict, api_key: str):
    """Streaming generation using the official SDK (synchronous)."""
    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(
        model_name=model,
        generation_config={
            'temperature': config.get('temperature', 0.8),
            'max_output_tokens': config.get('max_output_tokens', 2048),
            'top_p': config.get('top_p', 0.95),
            'top_k': config.get('top_k', 40),
        }
    )
    response = gen_model.generate_content(
        contents=contents,
        stream=True,
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text

# ── Direct Fetch Generation (fixed streaming) ────────────────

def generate_with_fetch(contents: List[Dict], model: str, config: Dict, api_key: str):
    """Direct REST API call (fallback) with synchronous streaming."""
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:streamGenerateContent?alt=sse"
    headers = {'x-goog-api-key': api_key, 'Content-Type': 'application/json'}
    payload = {
        'contents': contents,
        'generationConfig': {
            'temperature': config.get('temperature', 0.8),
            'maxOutputTokens': config.get('max_output_tokens', 2048),
            'topP': config.get('top_p', 0.95),
            'topK': config.get('top_k', 40),
        }
    }

    # Use httpx.stream directly with proper context
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        with client.stream('POST', url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', str(response.status_code))
                except:
                    error_msg = response.text
                raise Exception(f"HTTP {response.status_code}: {error_msg}")

            # Read the streaming response correctly
            buffer = ''
            for chunk in response.iter_bytes():
                buffer += chunk.decode()
                lines = buffer.split('\n')
                buffer = lines.pop() if lines else ''
                for line in lines:
                    if line.startswith('data: '):
                        data_str = line[6:].strip()
                        if not data_str or data_str == '[DONE]':
                            continue
                        try:
                            data = json.loads(data_str)
                            text = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                            if text:
                                yield text
                        except:
                            continue

# ── Hybrid Generator ──────────────────────────────────────────

def generate_hybrid(contents: List[Dict], model: str, config: Dict):
    """
    Try SDK first (if enabled), then fallback to direct fetch.
    Yields text chunks synchronously.
    """
    if USE_FETCH_FIRST:
        methods = ['fetch', 'sdk']
    elif USE_SDK_FIRST:
        methods = ['sdk', 'fetch']
    else:
        methods = ['sdk', 'fetch']

    models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
    last_error = None

    for current_model in models_to_try:
        for method in methods:
            for idx, api_key in enumerate(ALL_KEYS):
                if not api_key:
                    continue
                try:
                    print(f"🔄 Trying {method} on {current_model} with key {idx+1}/{len(ALL_KEYS)}")
                    if method == 'sdk':
                        for chunk in generate_with_sdk(contents, current_model, config, api_key):
                            yield chunk
                        print(f"✅ SDK success on {current_model}")
                        return
                    else:  # fetch
                        for chunk in generate_with_fetch(contents, current_model, config, api_key):
                            yield chunk
                        print(f"✅ Fetch success on {current_model}")
                        return
                except Exception as e:
                    print(f"⚠️ {method} failed: {e}")
                    last_error = e
                    if is_quota_error(e) or is_network_error(e):
                        continue
                    else:
                        if method == 'sdk' and SDK_FALLBACK:
                            print("🔄 Non-quota SDK error, falling back to fetch")
                            break
                        else:
                            raise

    raise Exception(f"All methods failed. Last error: {last_error}")

# ── Health Check ──────────────────────────────────────────────

def health_check():
    result = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'keys': len(ALL_KEYS),
        'models': FALLBACK_MODELS,
        'config': {
            'useSdkFirst': USE_SDK_FIRST,
            'sdkFallback': SDK_FALLBACK,
            'useFetchFirst': USE_FETCH_FIRST,
        },
        'methods': {}
    }
    if ALL_KEYS:
        try:
            genai.configure(api_key=ALL_KEYS[0])
            genai.list_models()
            result['methods']['sdk'] = {'status': 'healthy', 'tested': True}
        except Exception as e:
            result['methods']['sdk'] = {'status': 'unhealthy', 'error': str(e)}
            result['status'] = 'degraded'
    else:
        result['methods']['sdk'] = {'status': 'no_key', 'tested': False}

    try:
        resp = httpx.get(
            'https://generativelanguage.googleapis.com/v1/models',
            headers={'x-goog-api-key': ALL_KEYS[0] if ALL_KEYS else ''},
            timeout=5.0
        )
        result['methods']['fetch'] = {
            'status': 'healthy' if resp.status_code == 200 else 'unhealthy',
            'tested': True,
            'statusCode': resp.status_code
        }
        if resp.status_code != 200:
            result['status'] = 'degraded'
    except Exception as e:
        result['methods']['fetch'] = {'status': 'unhealthy', 'error': str(e)}
        result['status'] = 'degraded'

    return result

# ── Flask Routes ──────────────────────────────────────────────

@app.route('/api/gemini', methods=['GET', 'POST', 'OPTIONS'])
def handler():
    if request.method == 'OPTIONS':
        return '', 200

    if request.method == 'GET':
        return jsonify(health_check())

    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400

        contents = data.get('contents')
        if not contents or not isinstance(contents, list) or len(contents) == 0:
            return jsonify({'error': 'Missing or invalid "contents"'}), 400

        # Validate and clean contents – remove any unknown fields like 'searchUsed'
        cleaned_contents = []
        for msg in contents:
            if not msg.get('role') or not msg.get('parts') or not isinstance(msg['parts'], list) or len(msg['parts']) == 0:
                return jsonify({'error': 'Each message must have "role" and non-empty "parts"'}), 400
            if not msg['parts'][0].get('text'):
                return jsonify({'error': 'Each part must have "text" property'}), 400
            # Only keep 'role' and 'parts' – remove any extra fields
            cleaned_contents.append({
                'role': msg['role'],
                'parts': msg['parts']
            })
        contents = cleaned_contents

        if not ALL_KEYS:
            return jsonify({'error': 'No API keys configured'}), 500

        model = data.get('model', 'gemini-2.5-flash')
        temperature = data.get('temperature', 0.8)
        max_tokens = data.get('maxTokens', 2048)
        web_search = data.get('webSearch', False)
        mastermind = data.get('mastermind', False)

        # ── Real-time clock injection ──────────────────────────
        now = datetime.utcnow()
        utc_str = now.strftime('%a, %d %b %Y %H:%M:%S GMT')
        wat_str = now.replace(tzinfo=None).astimezone().strftime('%A, %B %d, %Y %I:%M:%S %p')
        live_context = f"[LIVE SYSTEM CLOCK - injected by server]\nUTC: {utc_str}\nWest Africa Time (WAT, UTC+1, Nigeria): {wat_str}\nYou MUST use this timestamp to answer any question about current time or date.\nNEVER claim you cannot access real-time time data."

        if contents and contents[0].get('role') == 'user':
            contents[0]['parts'][0]['text'] += '\n\n' + live_context

        # ── Web Search ──────────────────────────────────────────
        search_status = None
        if web_search:
            user_msgs = [m for m in contents if m['role'] == 'user']
            if user_msgs:
                last_user = user_msgs[-1]
                query = last_user['parts'][0]['text']
                query = query.replace('📄 File:', '').split('---')[0].strip()
                query = query.split('\n')[0] if '\n' in query else query
                query = query[:200]
                if query:
                    search_results = perform_web_search(query)
                    if search_results:
                        search_inject = {
                            'role': 'user',
                            'parts': [{'text': f"[LIVE WEB SEARCH - {utc_str}]\nQuery: \"{query[:150]}\"\n\n{search_results}\n\n[END SEARCH RESULTS]\nUse these fresh results for your answer. Cite sources."}]
                        }
                        last_user_idx = max(i for i, m in enumerate(contents) if m['role'] == 'user')
                        contents.insert(last_user_idx, search_inject)
                        search_status = {'succeeded': True, 'query': query[:150], 'resultCount': len(search_results.split('\n\n'))}
                    else:
                        last_user_idx = max(i for i, m in enumerate(contents) if m['role'] == 'user')
                        contents.insert(last_user_idx, {
                            'role': 'user',
                            'parts': [{'text': '[NOTE] Live web search API is not configured. Use the system clock for time questions.'}]
                        })
                        search_status = {'succeeded': False, 'error': 'CSE not configured'}

        # ── Mastermind Mode ─────────────────────────────────────
        if mastermind:
            thinking_instruction = {
                'role': 'user',
                'parts': [{'text': """[MASTERMIND MODE - CONDENSED REASONING]

DO NOT output thinking tags. Instead:
1. Reason BRIEFLY (1-2 sentences max) about the core issue
2. Consider 1 alternative perspective
3. State your final answer with confidence

Focus on QUALITY over length."""}]
            }
            contents.insert(0, thinking_instruction)

        # ── Build config ──────────────────────────────────────
        config = {
            'temperature': temperature,
            'max_output_tokens': max_tokens if not mastermind else max(max_tokens, 3000),
            'stream': True,
            'top_p': 0.95,
            'top_k': 40,
        }

        # ── Generate streaming response ──────────────────────
        def generate():
            if search_status:
                yield f"data: {json.dumps({'searchStatus': search_status})}\n\n"

            full_text = ''
            try:
                for chunk in generate_hybrid(contents, model, config):
                    full_text += chunk
                    payload = {'candidates': [{'content': {'parts': [{'text': full_text}]}}]}
                    yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                error_msg = str(e)
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                yield "data: [DONE]\n\n"

        return Response(stream_with_context(generate()), mimetype='text/event-stream')

# ── Vercel entry point ────────────────────────────────────────
# Vercel looks for `app` (Flask instance)
