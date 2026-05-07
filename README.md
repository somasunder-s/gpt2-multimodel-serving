# gpt2-multimodel-serving

> **Personal experiment** — an exploration of how to host two domain-specific fine-tuned GPT-2 models behind a single GPU-backed endpoint, and how worker/thread/concurrency choices affect throughput.

A **multi-model GPT-2 inference server** built with FastAPI and deployed via Docker + Gunicorn. Serves two fine-tuned GPT-2 models (`domain_a`, `domain_b`) behind a single `/generate/` endpoint, with the model chosen per-request. The "domains" are intentionally generic — drop in your own fine-tuned weights for whatever extraction or generation task you're working on.

Includes a **GPU benchmarking suite** that load-tests the deployment across worker / thread / concurrency configurations on NVIDIA L4 instances and writes results to CSV for analysis.

## What it does

```
                        client request
                  { prompts: [...], model: "domain_a" | "domain_b" }
                                    │
                                    ▼
                          FastAPI / Gunicorn
                          (10 workers · 32 threads · uvicorn workers)
                                    │
                            select model by name
                            │                │
                            ▼                ▼
                       domain_a         domain_b
                        GPT-2            GPT-2
                       (CUDA)           (CUDA)
                            │                │
                            └──── batch encode ────┘
                                    │
                                    ▼
                            generated text → JSON
```

Both models are loaded once at process start and pinned to GPU. Each request specifies which model to route to, so a single deployment serves both domains without cold-starting.

## Why this design

- **Single endpoint, multi-model** — operational simplicity over deploying two services. Routing is one dict lookup; memory cost is the sum of model weights, which fit on a single L4 (24 GB).
- **Gunicorn + uvicorn workers** — process-level parallelism (workers) for CPU-bound preprocessing, async (uvicorn) inside each worker for I/O. The Dockerfile defaults to `-w 10 --threads 32`, picked because the sweep below showed throughput plateaus around concurrency 7–9 regardless of worker/thread count, and 10×32 has the most headroom before errors appear.
- **Two separate models** instead of one model with a routing prefix — each domain was fine-tuned on its own labeled data with different convergence behavior, so squashing them into a single conditioned model would have meant either retraining together or picking one's hyperparameters. Routing-by-name keeps the inference code dumb and lets the training side iterate independently.
- **Docker** — reproducible runtime, deployable to any GPU host (GCP, AWS, on-prem) without environment drift.

## Benchmarking results

Sweep across worker × thread × client-concurrency on **NVIDIA L4** (16 vCPU, 54 GB RAM, linux-deb), 250 requests per run, 512-token max generation. Full data in `data/gpu_vm_exp_summary.csv` (61 runs); summary below:

| Config (w / t / concurrency) | Success | Errors (503) | p90 latency | Throughput |
|---|---|---|---|---|
| **10 / 16 / 9** | 248/250 | 2 | 7.30 s | **1.28 req/s** ← peak |
| 10 / 16 / 8 | 248/250 | 2 | 6.56 s | 1.26 req/s |
| 8 / 32 / 7 | 249/250 | 1 | 5.82 s | 1.25 req/s |
| 8 / 32 / 6 (zero-error) | 250/250 | 0 | 5.05 s | 1.24 req/s |
| 8 / 32 / 4 (low concurrency) | 250/250 | 0 | 3.58 s | 1.17 req/s |

**Findings:**
- Throughput **plateaus at ~1.25 req/s** — limited by GPU compute on 512-token generation, not by web-tier scheduling. More workers/threads can't overcome it.
- The **error-free zone** is concurrency ≤ 7. Above that, the GPU's effective queue saturates and the server returns 503 (overload-shed). Worker/thread choice mainly affects *how gracefully* the system degrades past saturation.
- Picking 10×32 in production trades a small p90 latency increase for the most generous error-free band (≈ concurrency 8 still nearly clean).

`data/gpu_summariser.py` aggregates raw timing logs into the summary CSV.

## Tech stack

- **Inference:** PyTorch + Hugging Face Transformers (`GPT2LMHeadModel`, `GPT2Tokenizer`), CUDA
- **API:** FastAPI · Pydantic
- **Serving:** Gunicorn + Uvicorn workers
- **Container:** Docker (`python:3.9-slim` base)
- **Benchmarking:** custom client (`client_calls/`) using `requests` + `concurrent.futures`

## Project layout

```
app/
└── app.py                         FastAPI app: model loading, /generate endpoint
client_calls/
├── load_test.py, load_test_sweep.py  load-test clients (concurrent POSTs)
└── smoke_test.py                      single-request smoke test
data/
├── gpu_vm_exp_summary.csv           aggregated benchmark results
└── gpu_summariser.py                raw-runs → summary CSV
model/                                fine-tuned GPT-2 weights go here (gitignored)
Dockerfile                            container build
requirements.txt                      pinned deps
```

## Running

### Locally

```bash
# 1. Drop fine-tuned weights into:
#      model/domain_a_model/
#      model/domain_b_model/

pip install -r requirements.txt
gunicorn app.app:app -w 10 -k uvicorn.workers.UvicornWorker --threads 32 --bind 0.0.0.0:8000
# → POST http://localhost:8000/generate/
```

### Docker

```bash
docker build -t gpt2-serving .
docker run --gpus all -p 8000:8000 -v $(pwd)/model:/app/model gpt2-serving
```

### Example request

```bash
curl -X POST http://localhost:8000/generate/ \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["Whatever input shape your fine-tuned model expects."],
    "model": "domain_a",
    "max_length": 512
  }'
```

## Notes

- Sample inputs in `client_calls/` are placeholder strings — they're load-test fixtures and don't mean anything to the model.
- Fine-tuned weights are not in the repo. Drop your own into `model/domain_a_model/` and `model/domain_b_model/` to use it.
- This is the **serving** half of a larger workflow — training the two GPT-2 models was a separate exercise and isn't included here.
