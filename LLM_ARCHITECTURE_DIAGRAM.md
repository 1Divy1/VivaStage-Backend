# LLM Abstraction Layer - Visual Architecture

## Before (Direct OpenAI Usage)
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ReelService   │    │   LLMEngine     │    │   OpenAI API    │
│                 │────│                 │────│                 │
│ process_reel()  │    │ llm_inference() │    │ gpt-4o-mini     │
│                 │    │                 │    │ ($$$)          │
└─────────────────┘    └─────────────────┘    └─────────────────┘

Problem: Hard-coded to OpenAI only, costs money for every test
```

## After (With Abstraction Layer)
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ReelService   │    │   LLMEngine     │    │ LLMProvider     │
│                 │────│                 │────│  (Interface)    │
│ process_reel()  │    │ llm_inference() │    │                 │
│ (unchanged!)    │    │ (unchanged!)    │    └─────────┬───────┘
└─────────────────┘    └─────────────────┘              │
                                                        │
                       ┌────────────────────────────────┴────────────────────────────────┐
                       │                                                                  │
                       ▼                                                                  ▼
            ┌─────────────────────┐                                          ┌─────────────────────┐
            │  OpenAIProvider     │                                          │  LocalLLMProvider   │
            │                     │                                          │                     │
            │ ┌─────────────────┐ │                                          │ ┌─────────────────┐ │
            │ │   OpenAI API    │ │                                          │ │ Ollama Server   │ │
            │ │   gpt-4o-mini   │ │                                          │ │ llama3.1:8b     │ │
            │ │   ($$$)        │ │                                          │ │ (FREE!)         │ │
            │ └─────────────────┘ │                                          │ └─────────────────┘ │
            └─────────────────────┘                                          └─────────────────────┘
```

## Configuration Switch
```
Environment Variable Controls Which Provider:

LLM_PROVIDER=local     →  Uses Local LLM (Free)
LLM_PROVIDER=openai    →  Uses OpenAI API (Paid)

┌─────────────────────┐
│    .env file        │
│                     │
│ LLM_PROVIDER=local  │  ←── Change this one line
│                     │      to switch providers!
└─────────────────────┘
```

## Data Flow Example
```
1. User Request (YouTube URL)
   ↓
2. ReelService.process_reel()
   ↓
3. LLMEngine.extract_highlights()
   ↓
4. LLMProvider.generate_structured_response()
   ↓
5. [Provider Specific Logic]

   LOCAL PROVIDER:                    OPENAI PROVIDER:
   ┌─────────────────┐               ┌─────────────────┐
   │ Format prompt   │               │ Format prompt   │
   │ for Ollama      │               │ for OpenAI      │
   │       ↓         │               │       ↓         │
   │ HTTP request to │               │ API call to     │
   │ localhost:11434 │               │ api.openai.com  │
   │       ↓         │               │       ↓         │
   │ Parse JSON      │               │ Extract parsed  │
   │ response        │               │ response        │
   └─────────────────┘               └─────────────────┘

6. Return HighlightMoments (same format from both!)
   ↓
7. ReelService continues with video processing...
```

## Key Benefits Visual
```
┌──────────────────────────────────────────────────────────────────┐
│                    SAME CODE EVERYWHERE                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ReelService  │  LLMEngine  │  Controllers  │  Dependencies     │
│      ✓        │      ✓      │       ✓       │       ✓           │
│   No changes  │  No changes │   No changes  │   Minor changes   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────┐                              ┌─────────────────┐
│   DEVELOPMENT   │                              │   PRODUCTION    │
│                 │                              │                 │
│ LLM_PROVIDER=   │  ←── One line change ──→    │ LLM_PROVIDER=   │
│     local       │                              │     openai      │
│                 │                              │                 │
│ Cost: FREE      │                              │ Cost: Per token │
│ Speed: Variable │                              │ Speed: Fast     │
│ Privacy: 100%   │                              │ Privacy: Cloud  │
└─────────────────┘                              └─────────────────┘
```

## File Structure Visual
```
app/
├── providers/
│   └── llm/
│       ├── base.py           ←── The "contract" all providers follow
│       ├── factory.py        ←── The "hiring manager" for providers
│       ├── openai_provider.py ←── Talks to OpenAI API
│       └── local_provider.py  ←── Talks to local Ollama
│
├── engines/
│   └── llm_engine.py         ←── Uses providers (no direct API calls)
│
├── core/
│   └── config.py             ←── Environment variable settings
│
└── dependencies/
    └── reel_dependencies.py  ←── Injects the right provider
```

This is the "plugin architecture" - your main code doesn't care which plugin (provider) it's using, as long as it follows the contract!