# External Services Configuration Guide

This document lists all external services used by the AI Chartered Accountant system, their pricing, and importance levels.

## Quick Start

1. Copy `.env.example` to `.env`
2. Sign up for at least one **CRITICAL** LLM service
3. (Optional) Add free tier services for better performance

---

## Service Categories

| Legend | Meaning |
|--------|---------|
| 🟢 **FREE** | Completely free or generous free tier |
| 🟡 **FREEMIUM** | Free tier with paid upgrades |
| 🔴 **PAID** | Requires payment |
| ⚡ **CRITICAL** | System won't work without it |
| ⭐ **RECOMMENDED** | Significantly improves performance |
| 📦 **OPTIONAL** | Nice to have, not required |

---

## 1. LLM Providers (at least one CRITICAL)

| Service | Tier | Importance | Free Tier | Paid Price | Sign Up |
|---------|------|------------|-----------|------------|---------|
| **Google Gemini** | 🟢 FREE | ⚡ CRITICAL | 60 req/min | $0.075/1M tokens | [Get API Key](https://makersuite.google.com/app/apikey) |
| **Groq** | 🟢 FREE | ⚡ CRITICAL | 30 req/min | - | [Get API Key](https://console.groq.com/keys) |
| **Ollama** | 🟢 FREE | ⭐ RECOMMENDED | Unlimited (local) | - | [Download](https://ollama.ai/) |
| **OpenRouter** | 🟡 FREEMIUM | 📦 OPTIONAL | $1 free credit | Pay-per-use | [Get API Key](https://openrouter.ai/keys) |
| **OpenAI** | 🔴 PAID | 📦 OPTIONAL | - | $2.50/1M tokens | [Get API Key](https://platform.openai.com/api-keys) |
| **HuggingFace** | 🟢 FREE | 📦 OPTIONAL | Rate limited | - | [Get Token](https://huggingface.co/settings/tokens) |

### Environment Variables
```env
GOOGLE_API_KEY=your-key
GROQ_API_KEY=your-key
OPENROUTER_API_KEY=your-key
OPENAI_API_KEY=your-key
HUGGINGFACEHUB_API_TOKEN=your-token
```

---

## 2. Embeddings (auto mode handles this)

| Service | Tier | Importance | Free Tier | Paid Price | Sign Up |
|---------|------|------------|-----------|------------|---------|
| **SentenceTransformers** | 🟢 FREE | ⭐ RECOMMENDED | Unlimited (local) | - | Auto-downloaded |
| **Jina AI** | 🟢 FREE | ⭐ RECOMMENDED | 1M tokens/month | $0.02/1M | [Get API Key](https://jina.ai/) |
| **HuggingFace API** | 🟢 FREE | 📦 OPTIONAL | Rate limited | - | Same as above |
| **Voyage AI** | 🟡 FREEMIUM | 📦 OPTIONAL | 50M tokens (new accts) | $0.02/1M | [Get API Key](https://www.voyageai.com/) |
| **OpenAI Embeddings** | 🔴 PAID | 📦 OPTIONAL | - | $0.02/1M | Same as LLM |
| **Cohere Embed** | 🟡 FREEMIUM | 📦 OPTIONAL | Trial credits | $0.10/1M | [Get API Key](https://cohere.com/) |
| **AWS Bedrock Titan** | 🔴 PAID | 📦 OPTIONAL | - | $0.11/1M | [AWS Console](https://console.aws.amazon.com/) |

### Environment Variables
```env
EMBEDDING_PROVIDER=auto
JINA_API_KEY=your-key
VOYAGE_API_KEY=your-key
COHERE_API_KEY=your-key
```

---

## 3. Re-Ranking (auto mode handles this)

| Service | Tier | Importance | Free Tier | Paid Price | Sign Up |
|---------|------|------------|-----------|------------|---------|
| **Cross-Encoder** | 🟢 FREE | ⭐ RECOMMENDED | Unlimited (local) | - | Auto-downloaded |
| **Jina Reranker** | 🟢 FREE | ⭐ RECOMMENDED | 10M tokens/month | $0.02/1M | [Get API Key](https://jina.ai/) |
| **Cohere Rerank** | 🟡 FREEMIUM | 📦 OPTIONAL | 1k calls/month | ~$1/1k calls | [Get API Key](https://cohere.com/) |

### Environment Variables
```env
# Uses JINA_API_KEY from embeddings section
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
COHERE_RERANK_MODEL=rerank-v3.5
```

---

## 4. Vector Database

| Service | Tier | Importance | Free Tier | Paid Price | Sign Up |
|---------|------|------------|-----------|------------|---------|
| **ChromaDB** | 🟢 FREE | ⭐ RECOMMENDED | Unlimited (local) | - | Auto-installed |
| **Qdrant Cloud** | 🟡 FREEMIUM | ⭐ RECOMMENDED | 1GB free | From $25/mo | [Sign Up](https://cloud.qdrant.io/) |

### Environment Variables
```env
VECTOR_DB_TYPE=qdrant
VECTOR_DB_QDRANT_HOST=https://your-cluster.cloud.qdrant.io
VECTOR_DB_QDRANT_API_KEY=your-key
QDRANT_COLLECTION=AI-CA-768
```

---

## 5. Caching (Redis)

| Service | Tier | Importance | Free Tier | Paid Price | Sign Up |
|---------|------|------------|-----------|------------|---------|
| **Local Redis** | 🟢 FREE | 📦 OPTIONAL | Unlimited | - | [Docker](https://hub.docker.com/_/redis) |
| **Upstash Redis** | 🟡 FREEMIUM | 📦 OPTIONAL | 10k cmds/day | From $0.20/100k | [Sign Up](https://upstash.com/) |

### Environment Variables
```env
REDIS_URL=https://your-instance.upstash.io
REDIS_TOKEN=your-token
```

---

## 6. Code Execution Sandbox

| Service | Tier | Importance | Free Tier | Paid Price | Sign Up |
|---------|------|------------|-----------|------------|---------|
| **Local Sandbox** | 🟢 FREE | ⭐ RECOMMENDED | Unlimited | - | Built-in |
| **E2B Sandbox** | 🟡 FREEMIUM | 📦 OPTIONAL | 100 hrs/month | $0.14/hr | [Sign Up](https://e2b.dev/) |

### Environment Variables
```env
ENABLE_E2B=true
E2B_API_KEY=your-key
```
**Requirement**: `pip install e2b-code-interpreter` (for Remote Mode)

---

## 7. AWS Services (Production Only)

| Service | Tier | Importance | Free Tier | Paid Price | Sign Up |
|---------|------|------------|-----------|------------|---------|
| **AWS S3** | 🟡 FREEMIUM | 📦 OPTIONAL | 5GB/12 months | $0.023/GB | [AWS Console](https://console.aws.amazon.com/) |
| **AWS ElastiCache** | 🔴 PAID | 📦 OPTIONAL | - | From $0.017/hr | [AWS Console](https://console.aws.amazon.com/) |

### Environment Variables
```env
DEPLOYMENT_ENV=aws
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_DEFAULT_REGION=ap-south-1
DEPLOYMENT_AWS_S3_BUCKET=your-bucket
```

---

## 8. Observability (Optional)

| Service | Tier | Importance | Free Tier | Paid Price | Sign Up |
|---------|------|------------|-----------|------------|---------|
| **Sentry** | 🟡 FREEMIUM | 📦 OPTIONAL | 5k events/month | From $26/mo | [Sign Up](https://sentry.io/) |
| **OpenObserve** | 🟢 FREE | 📦 OPTIONAL | Self-hosted | - | [GitHub](https://github.com/openobserve/openobserve) |

### Environment Variables
```env
SENTRY_DSN=your-dsn
OPENOBSERVE_URL=your-url
OPENOBSERVE_AUTH_KEY=your-key
```

---

## Minimum Viable Configuration

For the system to work, you need **at minimum**:

```env
# Just ONE of these LLM providers:
GOOGLE_API_KEY=your-google-api-key
# OR
GROQ_API_KEY=your-groq-api-key

# Everything else will use local/free fallbacks automatically!
```

---

## Recommended Production Configuration

For best performance in production:

```env
# LLM (multiple for redundancy)
GOOGLE_API_KEY=your-key
GROQ_API_KEY=your-key

# Embeddings & Reranking (FREE cloud)
JINA_API_KEY=your-key

# Vector DB (cloud)
VECTOR_DB_TYPE=qdrant
VECTOR_DB_QDRANT_HOST=your-cluster-url
VECTOR_DB_QDRANT_API_KEY=your-key

# Caching (cloud)
REDIS_URL=your-upstash-url
REDIS_TOKEN=your-token

# Deployment
DEPLOYMENT_ENV=aws
```

---

## Cost Estimation (Monthly)

| Usage Level | Estimated Cost | Services Used |
|-------------|----------------|---------------|
| **Development** | $0 | Local + Free tiers |
| **Low Volume** (<1k queries/day) | $0-5 | Free tiers only |
| **Medium Volume** (1k-10k queries/day) | $20-50 | Qdrant + Upstash |
| **High Volume** (>10k queries/day) | $100+ | Full cloud stack |

---

## Quick Links Summary

| Category | Primary Service | Link |
|----------|-----------------|------|
| LLM | Google Gemini | https://makersuite.google.com/app/apikey |
| LLM | Groq | https://console.groq.com/keys |
| Embeddings | Jina AI | https://jina.ai/ |
| Vector DB | Qdrant Cloud | https://cloud.qdrant.io/ |
| Cache | Upstash | https://upstash.com/ |
| Sandbox | E2B | https://e2b.dev/ |
| AWS | Console | https://console.aws.amazon.com/ |
