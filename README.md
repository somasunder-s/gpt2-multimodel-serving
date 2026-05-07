# gpt2-multimodel-serving

A **multi-model GPT-2 inference server** built with FastAPI and deployed via Docker + Gunicorn. Serves two domain-specific fine-tuned GPT-2 models (resume entity extraction for *internship* and *education* sections) behind a single `/generate/` endpoint, with the model chosen per-request.

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
                      internship       education
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
- **Two separate models** instead of one model with a routing prefix — each section (internship / education) was fine-tuned on its own labeled data and the two models had different convergence behavior. Routing-by-name keeps the inference code dumb and lets the training side iterate independently.
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
├── ngrok_call.py, ngrok_call2.py   load-test clients (concurrent POSTs)
└── res_api.py                       single-request smoke test
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
    "prompts": ["Jane Doe — QA Engineer …"],
    "model": "domain_a",
    "max_length": 512
  }'
```

## Notes

- Sample inputs in `client_calls/` use a synthetic resume (`Jane Doe`); they're load-test fixtures, not real candidate data.
- Fine-tuned weights are not in the repo. Drop your own into `model/<name>_agent_tagged_model/` to use it.
- This is the **serving** half of a larger workflow — the training half lived in a separate notebook outside this repo.
