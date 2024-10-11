from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = FastAPI()

# Load models and tokenizers
models = {
    "domain_a": {
        "model": GPT2LMHeadModel.from_pretrained("model/domain_a_model"),
        "tokenizer": GPT2Tokenizer.from_pretrained("model/domain_a_model"),
    },
    "domain_b": {
        "model": GPT2LMHeadModel.from_pretrained("model/domain_b_model"),
        "tokenizer": GPT2Tokenizer.from_pretrained("model/domain_b_model"),
    }
}

# Set device and move models to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model_name, model_info in models.items():
    model_info["model"].to(device)
    model_info["model"].eval()  # Set model to evaluation mode
    print(f"Model '{model_name}' loaded on device {device}")

# Log allocated GPU memory
print("Allocated GPU memory:", torch.cuda.memory_allocated() / 1024 ** 2, "MB")


class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 512
    model: str  # Specify which model to use


@app.post("/generate/")
async def generate_text(request: TextGenerationRequest):
    """Generate text based on the provided prompt using the specified model."""

    # Validate the model parameter
    if request.model not in models:
        raise HTTPException(status_code=400, detail="Invalid model specified. Choose 'domain_a' or 'domain_b'.")

    # Retrieve the selected model and tokenizer
    selected_model = models[request.model]["model"]
    selected_tokenizer = models[request.model]["tokenizer"]

    # Tokenize the input prompt
    inputs = selected_tokenizer.encode(request.prompt, return_tensors="pt").to(device)

    # Create attention mask
    attention_mask = torch.ones(inputs.shape, device=device)

    with torch.no_grad():
        # Generate text
        outputs = selected_model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=request.max_length,
            num_return_sequences=1,
            pad_token_id=selected_tokenizer.eos_token_id
        )

    # Decode the generated text
    generated_text = selected_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}


