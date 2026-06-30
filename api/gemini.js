// ============================================================
// SSC AI HYBRID BACKEND - SDK + Direct Fetch with Smart Fallback
// ============================================================
// Deployment: Vercel, Netlify, AWS Lambda, Cloudflare Workers
// Node.js 18+ Required
// ============================================================

const { GoogleGenerativeAI } = require('@google/generative-ai');

// ── CONFIGURATION ──────────────────────────────────────────────────────────────

// Parse API keys from environment
const API_KEYS = (process.env.GEMINI_API_KEYS || '')
    .split(',')
    .map(k => k.trim())
    .filter(k => k.length > 0);

// Primary API key (can be same as first key or separate)
const PRIMARY_KEY = process.env.GEMINI_API_KEY || API_KEYS[0] || '';

// All keys (deduplicated)
const ALL_KEYS = [...new Set([PRIMARY_KEY, ...API_KEYS].filter(Boolean))];

// Hybrid configuration
const USE_SDK_FIRST = process.env.USE_SDK_FIRST !== 'false'; // default: true
const SDK_FALLBACK = process.env.SDK_FALLBACK !== 'false'; // default: true
const USE_FETCH_FIRST = process.env.USE_FETCH_FIRST === 'true'; // experimental

// Model fallback chain
const FALLBACK_MODELS = [
    'gemini-2.5-flash',
    'gemini-2.0-flash-lite',
    'gemini-2.0-flash',
    'gemini-1.5-flash',
];

// CORS configuration
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || '*')
    .split(',')
    .map(o => o.trim());

// Request configuration
const REQUEST_TIMEOUT_MS = parseInt(process.env.REQUEST_TIMEOUT_MS || '30000', 10);
const MAX_RETRIES = parseInt(process.env.MAX_RETRIES || '3', 10);

// ── WEB SEARCH (Google Custom Search) ─────────────────────────────────────────

const SEARCH_API_KEY = process.env.SEARCH_API_KEY || '';
const SEARCH_ENGINE_ID = process.env.SEARCH_ENGINE_ID || '';

async function performWebSearch(query) {
    if (!SEARCH_API_KEY || !SEARCH_ENGINE_ID) return null;
    try {
        const url = `https://www.googleapis.com/customsearch/v1?key=${SEARCH_API_KEY}&cx=${SEARCH_ENGINE_ID}&q=${encodeURIComponent(query)}&num=5`;
        const resp = await fetch(url, { signal: AbortSignal.timeout(8000) });
        if (!resp.ok) return null;
        const data = await resp.json();
        const items = data.items || [];
        if (!items.length) return null;
        return items.map((it, i) =>
            `[${i + 1}] ${it.title}\n${it.snippet}\nURL: ${it.link}`
        ).join('\n\n');
    } catch (e) {
        console.warn('⚠️ Web search failed:', e.message);
        return null;
    }
}

// ── ERROR DETECTION ─────────────────────────────────────────────────────────────

function isQuotaError(error) {
    const message = error.message?.toLowerCase() || '';
    const status = error.status || error.code || 0;
    return status === 429 || status === 403 ||
        message.includes('429') || message.includes('quota') ||
        message.includes('rate limit') || message.includes('api key') ||
        message.includes('forbidden') || message.includes('resource exhausted') ||
        message.includes('daily limit') || message.includes('billing');
}

function isNetworkError(error) {
    const message = error.message?.toLowerCase() || '';
    return message.includes('fetch') || message.includes('network') ||
        message.includes('connect') || message.includes('timeout') ||
        message.includes('econnrefused') || message.includes('econnreset') ||
        message.includes('socket') || message.includes('dns');
}

// ── CORS ────────────────────────────────────────────────────────────────────────

function getCorsHeaders(origin) {
    if (ALLOWED_ORIGINS.includes('*')) {
        return { 'Access-Control-Allow-Origin': '*' };
    }
    if (origin && ALLOWED_ORIGINS.includes(origin)) {
        return { 'Access-Control-Allow-Origin': origin };
    }
    return {};
}

// ── SDK GENERATION ─────────────────────────────────────────────────────────────

async function generateWithSDK(contents, model, config, apiKey, signal) {
    try {
        const genAI = new GoogleGenerativeAI(apiKey);
        const genModel = genAI.getGenerativeModel({
            model: model,
            generationConfig: {
                temperature: config.temperature,
                maxOutputTokens: config.maxOutputTokens,
                topP: config.topP || 0.95,
                topK: config.topK || 40
            }
        });

        if (config.stream) {
            const result = await genModel.generateContentStream(
                { contents },
                { signal }
            );
            return { type: 'stream', result };
        }

        const result = await genModel.generateContent(
            { contents },
            { signal }
        );
        return { type: 'response', result: result.response };
    } catch (error) {
        console.error('SDK Error:', error.message);
        throw error;
    }
}

// ── FETCH GENERATION ───────────────────────────────────────────────────────────

async function generateWithFetch(contents, model, config, apiKey, signal) {
    const url = `https://generativelanguage.googleapis.com/v1/models/${model}:${config.stream ? 'streamGenerateContent' : 'generateContent'}`;
    const finalUrl = config.stream ? `${url}?alt=sse` : url;

    try {
        const response = await fetch(finalUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-goog-api-key': apiKey
            },
            body: JSON.stringify({
                contents,
                generationConfig: {
                    temperature: config.temperature,
                    maxOutputTokens: config.maxOutputTokens,
                    topP: config.topP || 0.95,
                    topK: config.topK || 40
                }
            }),
            signal: signal
        });

        if (!response.ok) {
            const errorText = await response.text();
            let errorData;
            try { errorData = JSON.parse(errorText); } catch (e) { errorData = { error: errorText }; }

            const error = new Error(
                errorData.error?.message ||
                errorData.error ||
                `HTTP ${response.status}`
            );
            error.status = response.status;
            error.code = response.status;
            error.response = response;
            throw error;
        }

        return { type: 'response', result: response };
    } catch (error) {
        console.error('Fetch Error:', error.message);
        throw error;
    }
}

// ── STREAM RESPONSE HANDLER ────────────────────────────────────────────────────

async function handleStreamResponse(result, res) {
    let fullText = '';
    let lastUsageMetadata = null;

    if (result.type === 'stream') {
        // SDK streaming
        for await (const chunk of result.result.stream) {
            const text = chunk.text();
            if (text) {
                fullText += text;
                const payload = {
                    candidates: [{
                        content: {
                            parts: [{ text: fullText }]
                        }
                    }]
                };
                res.write(`data: ${JSON.stringify(payload)}\n\n`);
            }
            if (chunk.usageMetadata) {
                lastUsageMetadata = chunk.usageMetadata;
            }
        }
    } else {
        // Fetch streaming
        const reader = result.result.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const jsonStr = line.slice(6).trim();
                    if (!jsonStr || jsonStr === '[DONE]') continue;

                    try {
                        const data = JSON.parse(jsonStr);
                        const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
                        if (text) {
                            fullText += text;
                            const payload = {
                                candidates: [{
                                    content: {
                                        parts: [{ text: fullText }]
                                    }
                                }]
                            };
                            res.write(`data: ${JSON.stringify(payload)}\n\n`);
                        }
                        if (data.usageMetadata) {
                            lastUsageMetadata = data.usageMetadata;
                        }
                    } catch (e) {
                        // Skip invalid JSON
                    }
                }
            }
        }
    }

    // Send final usage metadata
    if (lastUsageMetadata) {
        const metaPayload = { usageMetadata: lastUsageMetadata };
        res.write(`data: ${JSON.stringify(metaPayload)}\n\n`);
    }

    // Handle empty response
    if (!fullText) {
        fullText = "[The AI returned an empty response. This can happen due to a safety filter or an ambiguous prompt. Please try rephrasing.]";
        const payload = {
            candidates: [{
                content: {
                    parts: [{ text: fullText }]
                }
            }]
        };
        res.write(`data: ${JSON.stringify(payload)}\n\n`);
    }

    res.write('data: [DONE]\n\n');
    res.end();
    return fullText;
}

// ── HYBRID GENERATOR ──────────────────────────────────────────────────────────

async function generateHybrid(contents, model, config, res, webSearch = false, mastermind = false) {
    const modelsToTry = [model, ...FALLBACK_MODELS.filter(m => m !== model)];
    let lastError = null;
    let lastMethod = '';

    const methodStats = {
        sdk: { attempts: 0, successes: 0, errors: 0 },
        fetch: { attempts: 0, successes: 0, errors: 0 }
    };

    // Determine method priority
    let methods = [];
    if (USE_FETCH_FIRST) {
        methods = ['fetch', 'sdk'];
    } else if (USE_SDK_FIRST) {
        methods = ['sdk', 'fetch'];
    } else {
        methods = ['sdk', 'fetch'];
    }

    for (const currentModel of modelsToTry) {
        console.log(`🔄 Trying model: ${currentModel}`);

        for (let method of methods) {
            if (method === 'sdk') {
                for (let i = 0; i < ALL_KEYS.length; i++) {
                    const apiKey = ALL_KEYS[i];
                    if (!apiKey) continue;

                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

                    try {
                        console.log(`🔑 SDK Method | Key ${i + 1}/${ALL_KEYS.length} | Model: ${currentModel}`);
                        methodStats.sdk.attempts++;

                        const result = await generateWithSDK(
                            contents,
                            currentModel,
                            { ...config, stream: true },
                            apiKey,
                            controller.signal
                        );

                        clearTimeout(timeoutId);
                        methodStats.sdk.successes++;
                        lastMethod = 'sdk';

                        await handleStreamResponse(result, res);
                        console.log(`✅ SDK success with key ${i + 1}`);
                        return { success: true, method: 'sdk', model: currentModel };

                    } catch (error) {
                        clearTimeout(timeoutId);
                        methodStats.sdk.errors++;
                        const isQuota = isQuotaError(error);
                        const isNetwork = isNetworkError(error);

                        console.warn(`⚠️ SDK failed (key ${i + 1}): ${error.message}`);

                        if (isQuota || isNetwork) {
                            lastError = error;
                            continue;
                        }

                        if (SDK_FALLBACK) {
                            console.log(`🔄 SDK error (not quota), falling back to fetch`);
                            break;
                        }

                        throw error;
                    }
                }
            }

            if (method === 'fetch' || (method === 'sdk' && SDK_FALLBACK)) {
                for (let i = 0; i < ALL_KEYS.length; i++) {
                    const apiKey = ALL_KEYS[i];
                    if (!apiKey) continue;

                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

                    try {
                        console.log(`🌐 Fetch Method | Key ${i + 1}/${ALL_KEYS.length} | Model: ${currentModel}`);
                        methodStats.fetch.attempts++;

                        const result = await generateWithFetch(
                            contents,
                            currentModel,
                            { ...config, stream: true },
                            apiKey,
                            controller.signal
                        );

                        clearTimeout(timeoutId);
                        methodStats.fetch.successes++;
                        lastMethod = 'fetch';

                        await handleStreamResponse(result, res);
                        console.log(`✅ Fetch success with key ${i + 1}`);
                        return { success: true, method: 'fetch', model: currentModel };

                    } catch (error) {
                        clearTimeout(timeoutId);
                        methodStats.fetch.errors++;
                        const isQuota = isQuotaError(error);
                        const isNetwork = isNetworkError(error);

                        console.warn(`⚠️ Fetch failed (key ${i + 1}): ${error.message}`);

                        if (isQuota || isNetwork) {
                            lastError = error;
                            continue;
                        }

                        if (method === 'fetch' && USE_SDK_FIRST) {
                            console.log(`🔄 Fetch error, trying SDK...`);
                            break;
                        }

                        throw error;
                    }
                }
            }
        }
    }

    // All methods failed
    let retryDelay = 60;
    if (lastError) {
        const details = lastError.errorDetails || lastError.details || [];
        for (const d of details) {
            if (d['@type']?.includes('RetryInfo') && d.retryDelay) {
                const s = typeof d.retryDelay === 'string' ? parseInt(d.retryDelay, 10) : d.retryDelay.seconds || 0;
                if (!isNaN(s) && s > 0) { retryDelay = s; break; }
            }
        }
        if (retryDelay === 60 && lastError.headers) {
            const ra = lastError.headers['retry-after'] || lastError.headers['Retry-After'];
            if (ra) { const p = parseInt(ra, 10); if (!isNaN(p) && p > 0) retryDelay = p; }
        }
    }

    console.log('📊 Method Stats:', {
        sdk: `${methodStats.sdk.successes}/${methodStats.sdk.attempts} (${Math.round(methodStats.sdk.successes / (methodStats.sdk.attempts || 1) * 100)}%)`,
        fetch: `${methodStats.fetch.successes}/${methodStats.fetch.attempts} (${Math.round(methodStats.fetch.successes / (methodStats.fetch.attempts || 1) * 100)}%)`
    });

    return {
        success: false,
        error: 'All methods and models have failed. Please try again later.',
        retryDelay,
        stats: methodStats,
        lastError: lastError?.message || 'Unknown error'
    };
}

// ── HEALTH CHECK ──────────────────────────────────────────────────────────────

async function checkHealth() {
    const health = {
        timestamp: new Date().toISOString(),
        status: 'healthy',
        methods: {},
        keys: ALL_KEYS.length,
        models: FALLBACK_MODELS,
        config: {
            useSdkFirst: USE_SDK_FIRST,
            sdkFallback: SDK_FALLBACK,
            useFetchFirst: USE_FETCH_FIRST
        }
    };

    // Test SDK
    if (USE_SDK_FIRST) {
        try {
            const testKey = ALL_KEYS[0];
            if (testKey) {
                const genAI = new GoogleGenerativeAI(testKey);
                const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash-lite' });
                await model.generateContent({
                    contents: [{ role: 'user', parts: [{ text: 'test' }] }],
                    generationConfig: { maxOutputTokens: 5 }
                });
                health.methods.sdk = { status: 'healthy', tested: true };
            } else {
                health.methods.sdk = { status: 'no_key', tested: false };
            }
        } catch (error) {
            health.methods.sdk = { status: 'unhealthy', error: error.message };
            health.status = 'degraded';
        }
    } else {
        health.methods.sdk = { status: 'disabled', tested: false };
    }

    // Test Fetch
    try {
        const testKey = ALL_KEYS[0];
        if (testKey) {
            const response = await fetch(
                `https://generativelanguage.googleapis.com/v1/models`,
                {
                    headers: { 'x-goog-api-key': testKey },
                    signal: AbortSignal.timeout(5000)
                }
            );
            health.methods.fetch = {
                status: response.ok ? 'healthy' : 'unhealthy',
                tested: true,
                statusCode: response.status
            };
            if (!response.ok) health.status = 'degraded';
        } else {
            health.methods.fetch = { status: 'no_key', tested: false };
        }
    } catch (error) {
        health.methods.fetch = { status: 'unhealthy', error: error.message };
        health.status = 'degraded';
    }

    return health;
}

// ── MAIN HANDLER ──────────────────────────────────────────────────────────────

module.exports = async function handler(req, res) {
    const origin = req.headers.origin || '';
    const corsHeaders = getCorsHeaders(origin);

    Object.entries(corsHeaders).forEach(([key, value]) => {
        res.setHeader(key, value);
    });
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-Requested-With');

    // ── OPTIONS ──────────────────────────────────────────────────────────────

    if (req.method === 'OPTIONS') {
        return res.status(200).end();
    }

    // ── GET - Health Check ──────────────────────────────────────────────────

    if (req.method === 'GET') {
        const health = await checkHealth();
        return res.status(200).json({
            status: 'ok',
            ...health,
            version: '2.0-hybrid'
        });
    }

    // ── POST ─────────────────────────────────────────────────────────────────

    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    try {
        const {
            contents,
            model = 'gemini-2.5-flash',
            temperature = 0.8,
            maxTokens = 2048,
            webSearch = false,
            mastermind = false
        } = req.body;

        // ── Validate Request ──────────────────────────────────────────────

        if (!contents || !Array.isArray(contents) || contents.length === 0) {
            return res.status(400).json({ error: 'Missing or invalid "contents"' });
        }

        for (const msg of contents) {
            if (!msg.role || !msg.parts || !Array.isArray(msg.parts) || msg.parts.length === 0) {
                return res.status(400).json({ error: 'Each message must have "role" and non-empty "parts"' });
            }
            if (!msg.parts[0].text) {
                return res.status(400).json({ error: 'Each message part must have "text" property' });
            }
        }

        if (ALL_KEYS.length === 0) {
            return res.status(500).json({ error: 'No API keys configured on server' });
        }

        // ── Real-Time Context Injection ────────────────────────────────────

        let workingContents = [...contents];
        const now = new Date();
        const utcStr = now.toUTCString();
        const watStr = now.toLocaleString('en-US', {
            timeZone: 'Africa/Lagos',
            hour12: true,
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });

        const liveContext = `[LIVE SYSTEM CLOCK - injected by server]
UTC: ${utcStr}
West Africa Time (WAT, UTC+1, Nigeria): ${watStr}
You MUST use this timestamp to answer any question about current time or date in any timezone.
Convert using UTC offsets: e.g. EST = UTC-5, GMT = UTC+0, IST = UTC+5:30, JST = UTC+9, etc.
NEVER claim you cannot access real-time time data — the clock above IS real-time.`;

        if (workingContents.length > 0 && workingContents[0].role === 'user') {
            workingContents[0] = {
                ...workingContents[0],
                parts: [{ text: workingContents[0].parts[0].text + '\n\n' + liveContext }]
            };
        }

        // ── Web Search ──────────────────────────────────────────────────────

        if (webSearch) {
            const userMsgs = workingContents.slice(1).filter(m => m.role === 'user');
            const lastUserMsg = userMsgs[userMsgs.length - 1];
            let query = lastUserMsg?.parts?.[0]?.text || '';
            query = query.replace(/📄 File:[sS]*?---/g, '').trim();
            query = (query.split('\n').find(l => l.trim().length > 3) || query).substring(0, 200);

            if (query) {
                const searchResults = await performWebSearch(query);
                if (searchResults) {
                    const injectIdx = workingContents.findLastIndex(m => m.role === 'user');
                    const searchContext = {
                        role: 'user',
                        parts: [{ text: `[LIVE WEB SEARCH - ${utcStr}]
Query: "${query.substring(0, 150)}"

${searchResults}

[END SEARCH RESULTS]
Use these fresh results for your answer. State the retrieval time. Cite sources with their URLs.` }]
                    };
                    workingContents.splice(injectIdx, 0, searchContext);
                    console.log('🔍 Web search injected for:', query.substring(0, 60));

                    res.write(`data: ${JSON.stringify({
                        searchStatus: {
                            succeeded: true,
                            query: query.substring(0, 150),
                            resultCount: searchResults.split('\n\n').length
                        }
                    })}\n\n`);
                } else {
                    const injectIdx = workingContents.findLastIndex(m => m.role === 'user');
                    workingContents.splice(injectIdx, 0, {
                        role: 'user',
                        parts: [{ text: '[NOTE] Live web search API is not configured. For time/date questions use the LIVE SYSTEM CLOCK above. For other real-time queries, be transparent about your training cutoff but always use the injected clock for time calculations.' }]
                    });
                    console.log('🔍 Web search: CSE not configured, injected fallback note');

                    res.write(`data: ${JSON.stringify({
                        searchStatus: {
                            succeeded: false,
                            error: 'CSE not configured'
                        }
                    })}\n\n`);
                }
            }
        }

        // ── Mastermind Mode ────────────────────────────────────────────────

        if (mastermind) {
            workingContents = [...workingContents];
            const thinkingInstruction = {
                role: 'user',
                parts: [{ text: `[MASTERMIND MODE - CONDENSED REASONING]

DO NOT output thinking tags. Instead:
1. Reason BRIEFLY (1-2 sentences max) about the core issue
2. Consider 1 alternative perspective
3. State your final answer with confidence

Focus on QUALITY over length. Avoid verbose explanations. This prevents token overflow while maintaining accuracy.` }]
            };
            workingContents = [thinkingInstruction, ...workingContents];
        }

        // ── Setup Streaming ─────────────────────────────────────────────────

        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('X-Accel-Buffering', 'no');

        // ── Generate with Hybrid ───────────────────────────────────────────

        const config = {
            temperature: temperature,
            maxOutputTokens: mastermind ? Math.max(maxTokens, 3000) : maxTokens,
            stream: true,
            topP: 0.95,
            topK: 40
        };

        const result = await generateHybrid(
            workingContents,
            model,
            config,
            res,
            webSearch,
            mastermind
        );

        if (!result || !result.success) {
            if (!res.headersSent) {
                const errorMsg = result?.error || 'Generation failed';
                res.write(`data: ${JSON.stringify({ error: errorMsg })}\n\n`);
                res.write('data: [DONE]\n\n');
                res.end();
            }
            return;
        }

        console.log(`✅ Hybrid generation succeeded with ${result.method} method on ${result.model}`);

    } catch (error) {
        console.error('🔥 Backend error:', error);

        if (res.headersSent) {
            try {
                res.write(`data: ${JSON.stringify({ error: error.message || 'Internal server error' })}\n\n`);
                res.write('data: [DONE]\n\n');
                res.end();
            } catch (e) {
                // Ignore write errors
            }
            return;
        }

        res.status(500).json({
            error: error.message || 'Internal server error',
            timestamp: new Date().toISOString()
        });
    }
};

// ── EXPORTS FOR TESTING ──────────────────────────────────────────────────────

module.exports.generateWithSDK = generateWithSDK;
module.exports.generateWithFetch = generateWithFetch;
module.exports.checkHealth = checkHealth;
module.exports.isQuotaError = isQuotaError;
module.exports.isNetworkError = isNetworkError;
module.exports.performWebSearch = performWebSearch;
module.exports.handleStreamResponse = handleStreamResponse;
module.exports.generateHybrid = generateHybrid;
