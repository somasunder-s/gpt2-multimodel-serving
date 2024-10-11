from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from typing import List

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
    prompts: List[str]  # Accept a list of prompts
    max_length: int = 512
    model: str  # Specify which model to use

@app.post("/generate/")
async def generate_text(request: TextGenerationRequest):
    """Generate text based on the provided prompts using the specified model."""

    # Validate the model parameter
    if request.model not in models:
        raise HTTPException(status_code=400, detail="Invalid model specified. Choose 'domain_a' or 'domain_b'.")

    # Retrieve the selected model and tokenizer
    selected_model = models[request.model]["model"]
    selected_tokenizer = models[request.model]["tokenizer"]

    # Tokenize the input prompts
    inputs = selected_tokenizer.batch_encode_plus(request.prompts, return_tensors="pt", padding=True, truncation=True, max_length=request.max_length).to(device)

    # Create attention mask
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        # Generate text for the entire batch
        outputs = selected_model.generate(
            inputs['input_ids'],
            attention_mask=attention_mask,
            max_length=request.max_length,
            num_return_sequences=1,
            pad_token_id=selected_tokenizer.eos_token_id
        )

    # Decode the generated text for each prompt in the batch
    generated_texts = [selected_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return {"generated_texts": generated_texts}  # Return a list of generated texts
