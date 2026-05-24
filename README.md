# Mobile CI RAG Analyzer

> RAG-augmented root cause analysis for iOS & Android mobile CI/CD pipelines.

Mobile CI pipelines fail in recurring, predictable patterns. This system automatically intercepts Jenkins build failures, retrieves semantically similar historical fixes from a Qdrant vector database, and generates structured root cause analysis using a locally-run LLM — entirely on-premise, no cloud API required.

---

## How It Works

```
Jenkins build failure
         │
         ▼
  ┌─────────────┐
  │  Log Parser │  ← extracts error type, platform, stage
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ RAG Router  │  ← selects relevant knowledge collections
  └──────┬──────┘
         │
    ┌────┴──────────────────────────────┐
    │                                   │
    ▼                                   ▼
Historical Fixes                 Platform Knowledge
Dependency RAG                   Conflict Resolution
    │                                   │
    └──────────────┬────────────────────┘
                   │  retrieved context
                   ▼
          ┌─────────────────┐
          │   Ollama LLM    │  ← llama3.1:8b, runs locally
          └────────┬────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │  Structured RCA       │
       │  • main_category      │
       │  • root_cause         │
       │  • explanation        │
       │  • fix suggestion     │
       │  • confidence score   │
       └───────────┬───────────┘
                   │
                   ▼
          n8n Webhook
          ├── Slack notification
          └── Bitbucket PR comment
```

---

## Features

- **4 specialized RAG collections** — each tuned for a different failure class
- **Semantic routing** — error type detection picks the right knowledge base before querying
- **100-entry anonymized dataset** — real iOS/Android CI failure patterns with ground truth RCA
- **Fully local** — Qdrant + Ollama, no data leaves your infrastructure
- **Feedback loop** — low-confidence results go to a pending queue; human-approved fixes are promoted back into the vector store
- **Jenkins-native** — single `post { failure { ... } }` block integration
- **n8n automation** — routes results to Slack channels and Bitbucket PR comments

---

## Architecture

| Layer | Component | Technology |
|---|---|---|
| Vector Store | 4 Qdrant collections | [Qdrant](https://qdrant.tech) |
| Embeddings | `nomic-embed-text` | [Ollama](https://ollama.com) |
| LLM | `llama3.1:8b` | [Ollama](https://ollama.com) |
| Orchestration | Python 3.11+ | — |
| Workflow | Webhook → Slack / Bitbucket | [n8n](https://n8n.io) |
| CI System | Jenkins Pipelines | Jenkinsfile |

---

## RAG Collections

| Collection | What it stores | Triggered by |
|---|---|---|
| `historical_fixes` | Past failure → fix pairs from real CI logs | All failure types |
| `dependency_knowledge` | CocoaPods, SPM, Gradle dependency patterns | `dependency_error`, `spm_*`, `cocoapods_*` |
| `platform_knowledge` | Xcode, Gradle, Fastlane platform errors | `xcode_build_error`, `gradle_*`, `export_failed` |
| `conflict_resolution` | Git merge conflict patterns for mobile files | `merge_conflict`, `git_merge_conflict` |

---

## Dataset

`agentops/data/dataset.jsonl` contains **100 anonymized CI failure records** collected from real iOS and Android mobile pipelines.

### Coverage

| Platform | Failure Category | Examples |
|---|---|---|
| iOS | Resource / Network | Git clone timeout, SPM DNS failure, agent disconnect |
| iOS | Build Artifact | Xcode build error, SwiftFormat failure, archive failed |
| iOS | Dependency | CocoaPods conflict, SPM resolution failure |
| iOS | Signing | Distribution cert missing, provisioning profile error |
| iOS | Merge Conflict | `.pbxproj` conflict, `Podfile.lock` conflict |
| Android | Build Artifact | Gradle daemon crash, Kotlin compile error, KAPT error |
| Android | Resource / Network | Gradle cache lock, JVM crash, curl timeout |
| Android | Merge Conflict | `build.gradle` conflict, `AndroidManifest.xml` conflict |
| Android | Dependency | Unresolved reference, missing interface implementation |

### Record Schema

```json
{
  "id": "example-ios-001",
  "metadata": {
    "platform": "ios",
    "error_type": "spm_dns_resolution_failure",
    "failed_stage": "Xcode Build",
    "status": "FAILURE"
  },
  "log_chunk": "...",
  "error_message": "...",
  "ground_truth_rca": "...",
  "fix": { "type": "...", "description": "...", "code": "..." },
  "rag_query": "..."
}
```

---

## Quick Start

### 1. Prerequisites

```bash
# Qdrant — local vector database
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Ollama — local LLM runtime
brew install ollama
ollama serve

# Pull required models
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 2. Install

```bash
git clone https://github.com/omerfarukfildisi/mobile-ci-rag-analyzer.git
cd mobile-ci-rag-analyzer

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

### 3. Configure

```bash
cp .env.example .env
```

```env
# Qdrant
QDRANT_URL=http://localhost:6333

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# n8n (optional)
N8N_WEBHOOK_URL=http://localhost:5678/webhook/agentops

# RAG tuning
RAG_MIN_SCORE=0.60
```

### 4. Seed the vector database

```python
from agentops.rag.historical_fix import HistoricalFixRAG
from agentops.rag.dependency import DependencyRAG
from agentops.rag.platform_knowledge import PlatformKnowledgeRAG
from agentops.rag.conflict_resolution import ConflictResolutionRAG

dataset = "agentops/data/dataset.jsonl"

for rag_cls in [HistoricalFixRAG, DependencyRAG, PlatformKnowledgeRAG, ConflictResolutionRAG]:
    rag = rag_cls()
    rag.ensure_collection()
    rag.seed_defaults_if_empty(dataset)
```

Or via CLI:
```bash
python -m agentops.cli db-status
```

### 5. Run demo analyses

```bash
# .pbxproj merge conflict
python -m agentops.cli demo pbxproj_conflict

# SPM DNS resolution failure
python -m agentops.cli demo spm_dns_exit74

# CocoaPods version conflict
python -m agentops.cli demo cocoapods_error

# iOS signing error
python -m agentops.cli demo signing_error
```

---

## Jenkins Integration

Add to your `Jenkinsfile`:

```groovy
pipeline {
    agent any
    stages {
        // ... your existing stages
    }
    post {
        failure {
            sh """
                source /path/to/.venv/bin/activate
                python -m agentops.cli analyze \
                  --run-id "${BUILD_TAG}" \
                  --platform "${PLATFORM}" \
                  --app "${APP_NAME}" \
                  --environment "${ENVIRONMENT}" \
                  --reason "Stage failed: ${STAGE_NAME}" \
                  --log-file /tmp/agentops_raw_log.txt
            """
        }
    }
}
```

The analyzer will:
1. Parse the failure log
2. Route to the relevant RAG collection
3. Retrieve top-k similar historical fixes
4. Generate structured RCA with the LLM
5. Post results to Slack and comment on the Bitbucket PR

---

## CLI Reference

```bash
# Analyze a real failure log
python -m agentops.cli analyze \
  --run-id jenkins-my-app-PR-42-123 \
  --platform iOS \
  --app MyApp \
  --environment staging \
  --reason "Stage failed: Xcode Build" \
  --log-file /path/to/build.log

# Run a demo with a built-in sample log
python -m agentops.cli demo [log_key]

# Check Qdrant collection sizes
python -m agentops.cli db-status

# List low-confidence pending analyses
python -m agentops.cli pending

# Promote a pending analysis to historical_fixes
python -m agentops.cli promote <run_id> --pr-title "Fix: SPM DNS timeout"
```

---

## Evaluation

Evaluation results comparing model versions (8B / 9B) and RAG modes (no RAG / RAG routing / RAG all) are stored in `eval_results/`.

```bash
# Run evaluation against the dataset
python -m agentops.evaluation_runner \
  --dataset agentops/data/dataset.jsonl \
  --mode rag_routing \
  --model llama3.1:8b:8b \
  --output eval_results/my_run.json
```

### Evaluated configurations

| Mode | Description |
|---|---|
| `no_rag` | LLM only, no retrieval |
| `rag_routing` | Semantic routing → targeted collection query |
| `rag_all` | Query all 4 collections, merge results |

---

## Project Structure

```
mobile-ci-rag-analyzer/
├── agentops/
│   ├── __init__.py
│   ├── analyzer.py            # SimpleAnalyzer + OllamaAnalyzer
│   ├── cli.py                 # CLI entrypoint
│   ├── evaluation_runner.py   # Batch evaluation
│   ├── feedback_service.py    # Pending queue + promote logic
│   ├── log_models.py          # CiLog, AnalysisResult models
│   ├── notifier.py            # Console + n8n webhook notifier
│   ├── ontology.py            # Error taxonomy definitions
│   ├── data/
│   │   └── dataset.jsonl      # 100-entry anonymized dataset
│   ├── knowledge/
│   │   ├── error_knowledge.py
│   │   └── error_ontology.json
│   └── rag/
│       ├── qdrant_client.py       # Qdrant connection (env-based)
│       ├── router.py              # RAG module selector
│       ├── embedder.py            # nomic-embed-text via Ollama
│       ├── historical_fix.py      # Historical fix RAG
│       ├── dependency.py          # Dependency error RAG
│       ├── platform_knowledge.py  # Platform-specific RAG
│       ├── conflict_resolution.py # Merge conflict RAG
│       └── pending_store.py       # Low-confidence queue
├── n8n_workflow.json          # n8n workflow export
├── requirements.txt
├── setup.py
└── .env.example
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server address |
| `QDRANT_API_KEY` | _(empty)_ | API key for Qdrant Cloud (optional) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `llama3.1:8b` | LLM model for analysis |
| `N8N_WEBHOOK_URL` | _(empty)_ | n8n webhook endpoint |
| `BITBUCKET_HOST` | _(empty)_ | Bitbucket server host |
| `BITBUCKET_PR_WORKSPACE` | _(empty)_ | Bitbucket workspace slug |
| `BITBUCKET_PR_REPO` | _(empty)_ | Bitbucket repo slug |
| `RAG_MIN_SCORE` | `0.60` | Minimum cosine similarity threshold |

---

## License

MIT
