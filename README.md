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

- **Single endpoint, multi-model** — operational simplicity over deploying two services. Routing is one dict lookup; memory cost is the sum of model weights, which fit on a single L4.
- **Gunicorn + uvicorn workers** — process-level parallelism (workers) for CPU-bound preprocessing, async (uvicorn) inside each worker for I/O. 10×32 was the sweet spot found via the benchmarks (see below).
- **Docker** — reproducible runtime, deployable to any GPU host (GCP, AWS, on-prem) without environment drift.

## Benchmarking

`data/gpu_vm_exp_summary.csv` contains throughput / latency measurements across:

- worker count (`w`) × thread count (`t`) × client concurrency (`con`)
- request volumes up to 250 concurrent
- on NVIDIA L4 instances (linux-deb, 16 vCPU, 54 GB RAM)

Metrics captured per run: mean / median / p95 / p99 latency, request rate, error rate (HTTP 503 from the server when overloaded).

`data/gpu_summariser.py` aggregates the raw runs into the summary CSV.

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
