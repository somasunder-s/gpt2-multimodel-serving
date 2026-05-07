"""Single-request smoke test against the /generate endpoint."""

import requests

URL = "http://localhost:8000/generate/"

payload = {
    "prompts": ["Sample input text for smoke-testing the /generate endpoint. Replace with whatever shape of input your fine-tuned model expects."],
    "max_length": 512,
    "model": "domain_a",  # or "domain_b" for the other model
}

response = requests.post(URL, json=payload)

if response.status_code == 200:
    generated_texts = response.json().get("generated_texts", [])
    for i, text in enumerate(generated_texts):
        print(f"[{i}] {text}")
else:
    print(f"Error: {response.status_code}, {response.text}")
