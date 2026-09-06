# Python
import torch

# Third party
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Local
from utils import BASE_MODEL_ID

CHECKPOINT_DIR = "./microservices_model_qlora/checkpoint-7500"


def load_base_model():
    """Load the non-fine-tuned model with 4-bit quantization.

    Returns:
        tuple: Base model and tokenizer instances.
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    return model, tokenizer


def generate_checkpoint_preview() -> None:
    """Load a checkpoint and generate a code continuation for validation.

    Returns:
        None: This function does not return a value.
    """
    base_model, tokenizer = load_base_model()
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
    model.eval()

    prompt = """# Microservices Architecture
# User management microservice using FastAPI and RabbitMQ.

from fastapi import FastAPI
import pika

app = FastAPI()
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print("\nGenerating code continuation...\n")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("--- GENERATED OUTPUT ---")
    print(result)


if __name__ == "__main__":
    generate_checkpoint_preview()