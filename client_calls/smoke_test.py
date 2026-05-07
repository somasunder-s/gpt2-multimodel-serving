import requests

# Define the API endpoint
url = "http://localhost:8000/generate/"

# prompt_ =
# Define the payload with the prompt, max_length, and model type
payload = {
    "prompt": "Sample input text for smoke-testing the /generate endpoint. Replace with whatever shape of input your fine-tuned model expects.",
    "max_length": 512,
    "model": "domain_a"  # or "domain_b" for the other model
}

# Make the POST request
response = requests.post(url, json=payload)

# Check the response
if response.status_code == 200:
    # Parse the JSON response
    generated_text = response.json().get("generated_text")
    print("Generated Text:", generated_text)
else:
    print(f"Error: {response.status_code}, {response.text}")
