import os
import json
import time
import asyncio
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

# CORS
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get('ALLOWED_ORIGINS', '*').split(',')]

# Web Search
SEARCH_API_KEY = os.environ.get('SEARCH_API_KEY', '')
SEARCH_ENGINE_ID = os.environ.get('SEARCH_ENGINE_ID', '')

# ── Web Search ──────────────────────────────────────────────────

async def perform_web_search(query: str) -> Optional[str]:
    if not SEARCH_API_KEY or not SEARCH_ENGINE_ID:
        return None
    url = f"https://www.googleapis.com/customsearch/v1?key={SEARCH_API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&num=5"
    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            resp = await client.get(url)
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

# ── SDK Generation ─────────────────────────────────────────────

async def generate_with_sdk(contents: List[Dict], model: str, config: Dict, api_key: str):
    """Streaming generation using the official SDK."""
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
    # Use synchronous SDK – we'll wrap in a thread if needed, but it's fine.
    response = gen_model.generate_content(
        contents=contents,
        stream=True,
    )
    # Return an async generator that yields chunks
    for chunk in response:
        if chunk.text:
            yield chunk.text

# ── Direct Fetch Generation ────────────────────────────────────

async def generate_with_fetch(contents: List[Dict], model: str, config: Dict, api_key: str):
    """Direct REST API call (fallback)."""
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
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        async with client.stream('POST', url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                try:
                    error_data = json.loads(error_text)
                    error_msg = error_data.get('error', {}).get('message', str(response.status_code))
                except:
                    error_msg = error_text.decode()
                raise Exception(f"HTTP {response.status_code}: {error_msg}")
            buffer = ''
            async for chunk in response.aiter_bytes():
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

async def generate_hybrid(contents: List[Dict], model: str, config: Dict,
                          web_search_enabled: bool = False,
                          mastermind_enabled: bool = False):
    """
    Try SDK first (if enabled), then fallback to direct fetch.
    Yields text chunks and also sends search status as an event (we'll include in SSE).
    """
    # Determine method order
    if USE_FETCH_FIRST:
        methods = ['fetch', 'sdk']
    elif USE_SDK_FIRST:
        methods = ['sdk', 'fetch']
    else:
        methods = ['sdk', 'fetch']

    models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
    last_error = None

    # If web search is enabled, we already injected it before calling this function.

    for current_model in models_to_try:
        for method in methods:
            # Try all keys
            for idx, api_key in enumerate(ALL_KEYS):
                if not api_key:
                    continue
                try:
                    print(f"🔄 Trying {method} on {current_model} with key {idx+1}/{len(ALL_KEYS)}")
                    if method == 'sdk':
                        # Generate stream from SDK
                        async for chunk in generate_with_sdk(contents, current_model, config, api_key):
                            yield chunk
                        print(f"✅ SDK success on {current_model}")
                        return  # Success
                    else:  # fetch
                        async for chunk in generate_with_fetch(contents, current_model, config, api_key):
                            yield chunk
                        print(f"✅ Fetch success on {current_model}")
                        return
                except Exception as e:
                    print(f"⚠️ {method} failed: {e}")
                    last_error = e
                    if is_quota_error(e) or is_network_error(e):
                        continue  # try next key
                    else:
                        if method == 'sdk' and SDK_FALLBACK:
                            print("🔄 Non-quota SDK error, falling back to fetch")
                            break  # break out of key loop to try fetch
                        else:
                            raise  # rethrow fatal errors

    # If we got here, all methods failed
    error_msg = f"All methods failed. Last error: {last_error}"
    raise Exception(error_msg)

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
    # Test SDK (if primary key exists)
    if ALL_KEYS:
        try:
            genai.configure(api_key=ALL_KEYS[0])
            genai.list_models()  # lightweight call
            result['methods']['sdk'] = {'status': 'healthy', 'tested': True}
        except Exception as e:
            result['methods']['sdk'] = {'status': 'unhealthy', 'error': str(e)}
            result['status'] = 'degraded'
    else:
        result['methods']['sdk'] = {'status': 'no_key', 'tested': False}

    # Test fetch (direct call to list models)
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
    # CORS preflight
    if request.method == 'OPTIONS':
        return '', 200

    # GET = Health check
    if request.method == 'GET':
        return jsonify(health_check())

    # POST = Generation
    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400

        contents = data.get('contents')
        if not contents or not isinstance(contents, list) or len(contents) == 0:
            return jsonify({'error': 'Missing or invalid "contents"'}), 400

        for msg in contents:
            if not msg.get('role') or not msg.get('parts') or not isinstance(msg['parts'], list) or len(msg['parts']) == 0:
                return jsonify({'error': 'Each message must have "role" and non-empty "parts"'}), 400
            if not msg['parts'][0].get('text'):
                return jsonify({'error': 'Each part must have "text" property'}), 400

        if not ALL_KEYS:
            return jsonify({'error': 'No API keys configured'}), 500

        # ── Extract parameters ─────────────────────────────────
        model = data.get('model', 'gemini-2.5-flash')
        temperature = data.get('temperature', 0.8)
        max_tokens = data.get('maxTokens', 2048)
        web_search = data.get('webSearch', False)
        mastermind = data.get('mastermind', False)

        # ── Real-time clock injection ──────────────────────────
        now = datetime.utcnow()
        utc_str = now.strftime('%a, %d %b %Y %H:%M:%S GMT')
        wat_str = now.replace(tzinfo=None).astimezone().strftime('%A, %B %d, %Y %I:%M:%S %p')  # simplified
        live_context = f"[LIVE SYSTEM CLOCK - injected by server]\nUTC: {utc_str}\nWest Africa Time (WAT, UTC+1, Nigeria): {wat_str}\nYou MUST use this timestamp to answer any question about current time or date.\nNEVER claim you cannot access real-time time data."

        # Inject into first user message
        if contents and contents[0].get('role') == 'user':
            contents[0]['parts'][0]['text'] += '\n\n' + live_context

        # ── Web Search ──────────────────────────────────────────
        search_status = None
        if web_search:
            # Get last user message (skip potential system prompt)
            user_msgs = [m for m in contents if m['role'] == 'user']
            if user_msgs:
                last_user = user_msgs[-1]
                query = last_user['parts'][0]['text']
                # Clean query
                query = query.replace('📄 File:', '').split('---')[0].strip()
                query = query.split('\n')[0] if '\n' in query else query
                query = query[:200]
                if query:
                    search_results = await perform_web_search(query)
                    if search_results:
                        # Inject search results before the last user message
                        search_inject = {
                            'role': 'user',
                            'parts': [{'text': f"[LIVE WEB SEARCH - {utc_str}]\nQuery: \"{query[:150]}\"\n\n{search_results}\n\n[END SEARCH RESULTS]\nUse these fresh results for your answer. Cite sources."}]
                        }
                        # Find the last user message index and insert before it
                        last_user_idx = max(i for i, m in enumerate(contents) if m['role'] == 'user')
                        contents.insert(last_user_idx, search_inject)
                        search_status = {'succeeded': True, 'query': query[:150], 'resultCount': len(search_results.split('\n\n'))}
                    else:
                        # Fallback note
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
            # Send search status first (if any)
            if search_status:
                yield f"data: {json.dumps({'searchStatus': search_status})}\n\n"

            full_text = ''
            try:
                # Run hybrid generator
                async def async_generator():
                    async for chunk in generate_hybrid(contents, model, config, web_search, mastermind):
                        yield chunk
                # Use asyncio to run the async generator in Flask's sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                async_gen = async_generator()
                while True:
                    try:
                        chunk = loop.run_until_complete(async_gen.__anext__())
                        full_text += chunk
                        # Send incremental text
                        payload = {'candidates': [{'content': {'parts': [{'text': full_text}]}}]}
                        yield f"data: {json.dumps(payload)}\n\n"
                    except StopAsyncIteration:
                        break
                    except Exception as e:
                        # Send error
                        error_payload = {'error': str(e)}
                        yield f"data: {json.dumps(error_payload)}\n\n"
                        break
                # Final usage metadata (simulate – we don't have token counts from SDK)
                # We could add a mock token count, but not necessary.
                # yield f"data: {json.dumps({'usageMetadata': {'totalTokenCount': len(full_text)//4}})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                error_msg = f"Backend error: {str(e)}"
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                yield "data: [DONE]\n\n"

        return Response(stream_with_context(generate()), mimetype='text/event-stream')

# ── Vercel expects `app` as the WSGI application ─────────────

# We'll keep the app as is.

# ── For local testing ──────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)
