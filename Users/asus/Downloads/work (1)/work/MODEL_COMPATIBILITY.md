# Model Compatibility Matrix

## Overview
This document provides a comprehensive compatibility matrix for all LLM models in the evaluation system.

## Model Status

| Model ID | Name | Provider | API Key | Status | Max Tokens | Notes |
|----------|------|----------|---------|--------|------------|-------|
| `mistral-7b-instruct` | Mistral 7B Instruct | OpenRouter | OPENROUTER_API_KEY | ✅ Working | 4000 | Free tier, reliable |
| `llama-3.1-8b-instant` | Llama 3.1 8B Instant | Groq | GROQ_API_KEY | ✅ Working | 8192 | Fastest response |
| `llama-3.3-70b-versatile` | Llama 3.3 70B Versatile | Groq | GROQ_API_KEY | ✅ Working | 8192 | High quality |
| `gemma-7b-it` | Gemma 7B IT | Groq | GROQ_API_KEY | ✅ Fixed | 8192 | Replaced gemma2-9b-it |
| `mixtral-8x22b` | Mixtral 8x22B | Groq | GROQ_API_KEY | ✅ Fixed | 65536 | Replaced mixtral-8x7b-32768 |
| `llama-3.1-70b` | Llama 3.1 70B | Groq | GROQ_API_KEY | ✅ Fixed | 8192 | Replaced llama-3.2-3b |
| `gemini-1.5-flash` | Gemini 1.5 Flash | Google | GEMINI_API_KEY | ✅ Fixed | 8192 | Rate limited |
| `gemini-2.0-flash` | Gemini 2.0 Flash | Google | GEMINI_API_KEY | ✅ Fixed | 8192 | Experimental, rate limited |
| `gemini-1.5-pro` | Gemini 1.5 Pro | Google | GEMINI_API_KEY | ✅ Fixed | 8192 | High quality, rate limited |
| `gemini-pro` | Gemini Pro | Google | GEMINI_API_KEY | ✅ Fixed | 8192 | Alias for gemini-1.5-pro |

## API Key Mapping

### OpenRouter
- **Key Variable**: `OPENROUTER_API_KEY`
- **Models**: `mistral-7b-instruct`
- **Endpoint**: `https://openrouter.ai/api/v1/chat/completions`
- **Headers**: 
  - `Authorization: Bearer ${OPENROUTER_API_KEY}`
  - `HTTP-Referer: http://localhost:3000`
  - `X-Title: Educational AI Testing Platform`

### Groq
- **Key Variable**: `GROQ_API_KEY`
- **Models**: `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `gemma-7b-it`, `mixtral-8x22b-instruct`, `llama-3.1-70b-versatile`
- **Endpoint**: `https://api.groq.com/openai/v1/chat/completions`
- **Headers**: 
  - `Authorization: Bearer ${GROQ_API_KEY}`
  - `Content-Type: application/json`

### Google Gemini
- **Key Variable**: `GEMINI_API_KEY`
- **Models**: `gemini-1.5-flash`, `gemini-2.0-flash-exp`, `gemini-1.5-pro`
- **SDK**: `@google/generative-ai`
- **Model ID Mapping**:
  - `gemini-pro` → `gemini-1.5-pro`
  - `gemini-2.0-flash-exp` → `gemini-2.0-flash-exp`

## Common Issues and Fixes

### Issue: Model Decommissioned
**Error**: "Model has been decommissioned"
**Solution**: Use updated model IDs:
- `gemma2-9b-it` → `gemma-7b-it`
- `mixtral-8x7b-32768` → `mixtral-8x22b-instruct`
- `llama-3.2-3b-preview` → `llama-3.1-70b-versatile`

### Issue: Quota Exceeded
**Error**: "Quota exceeded" or "429"
**Solution**: 
- Wait and retry
- Check API key limits
- Use alternative models

### Issue: Model Not Found
**Error**: "Model not available" or "404"
**Solution**:
- Verify model ID spelling
- Check provider documentation for current model names
- Use model ID mapping in code

### Issue: Empty Response
**Error**: "Empty response"
**Solution**:
- Check prompt length
- Verify API key permissions
- Check model availability status

## Evaluation Metrics Support

All models support:
- ✅ ROUGE-1, ROUGE-2, ROUGE-L
- ✅ BLEU
- ✅ BERTScore (simplified)
- ✅ Exact Match
- ✅ SummaC
- ✅ Response Length Metrics
- ✅ LLM-as-Judge (G-Eval)

## Performance Characteristics

| Model | Avg Response Time | Quality Score | Best For |
|-------|------------------|---------------|----------|
| Mistral 7B | ~2-5s | 4/5 | General Q&A |
| Llama 3.1 8B | ~0.5-1s | 4/5 | Fast responses |
| Llama 3.3 70B | ~1-3s | 5/5 | Complex reasoning |
| Gemma 7B | ~1-2s | 4/5 | Educational content |
| Mixtral 8x22B | ~2-4s | 5/5 | Diverse tasks |
| Llama 3.1 70B | ~2-4s | 5/5 | High quality |
| Gemini 1.5 Flash | ~1-3s | 5/5 | Multimodal |
| Gemini 2.0 Flash | ~1-2s | 5/5 | Advanced reasoning |
| Gemini 1.5 Pro | ~3-6s | 5/5 | Best quality |

## Token Limits

- **Default**: 4000 tokens
- **Groq Models**: Up to 8192 tokens (some up to 65536)
- **Gemini Models**: Up to 8192 tokens
- **OpenRouter**: Varies by model

## Rate Limits

- **Groq**: ~30 requests/minute (free tier)
- **Gemini**: Varies by model and tier
- **OpenRouter**: Varies by model

## Validation Checklist

Before using a model:
- [ ] API key is set in `.env`
- [ ] Model ID matches provider documentation
- [ ] Provider endpoint is accessible
- [ ] Token limits are appropriate for use case
- [ ] Rate limits are understood

