# LLM Abstraction Layer Setup Guide

## Overview

The LLM abstraction layer allows seamless switching between local LLMs (for development/testing) and API-based LLMs (for production) without any code changes.

## Quick Setup

### 1. For Local LLM Usage (Recommended for Development)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull a model (e.g., Llama 3.1 8B)
ollama pull llama3.1:8b

# 3. Set environment variables
export LLM_PROVIDER=local
export LOCAL_LLM_MODEL=llama3.1:8b
```

### 2. For OpenAI API Usage (Production)

```bash
# Set environment variables
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_api_key_here
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `local` | Provider type: `local` or `openai` |
| `LOCAL_LLM_URL` | `http://localhost:11434` | Ollama server URL |
| `LOCAL_LLM_MODEL` | `llama3.1:8b` | Default local model |
| `LOCAL_LLM_TIMEOUT` | `300` | Request timeout in seconds |
| `OPENAI_API_KEY` | - | OpenAI API key (required for OpenAI provider) |

## Supported Models

### Local Models (via Ollama)
- `llama3.1:8b` - Fast, good quality (recommended)
- `llama3.1:70b` - Higher quality, slower
- `mistral:7b` - Alternative option
- `codellama:7b` - Code-focused

### OpenAI Models
- `gpt-4o-mini` - Fast, cost-effective (default)
- `gpt-4o` - Higher quality
- `gpt-4-turbo` - Previous generation

## Testing the Setup

```python
# Test local provider
from app.providers.llm.factory import LLMProviderFactory

provider = LLMProviderFactory.create_provider(
    provider_type="local",
    local_llm_model="llama3.1:8b"
)

print(f"Provider: {provider.provider_name}")
print(f"Models: {await provider.get_available_models()}")
```

## Cost Comparison

| Provider | Cost per 1M tokens | Best for |
|----------|-------------------|----------|
| Local (Ollama) | **FREE** | Development, testing, privacy |
| OpenAI GPT-4o-mini | ~$0.50 | Production, high volume |
| OpenAI GPT-4o | ~$15.00 | Production, highest quality |

## Architecture Benefits

✅ **Zero refactoring** when switching providers
✅ **Configuration-driven** provider selection
✅ **Testable** with mock providers
✅ **Cost-effective** development with local models
✅ **Production-ready** with API providers

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama service
ollama serve
```

### Model Not Found
```bash
# List available models
ollama list

# Pull missing model
ollama pull llama3.1:8b
```

### Performance Tips
- Use smaller models (7B-8B) for development
- Use 70B+ models only when quality is critical
- Set appropriate timeouts for your hardware
- Consider GPU acceleration for local models