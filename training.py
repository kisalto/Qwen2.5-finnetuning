# Python
import torch

# Third party
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

# Local
from utils import BASE_MODEL_ID, DATASET_PATH, MICROSERVICES_MODEL_DIR

torch.__version__ = "2.6.0"


def build_quantization_config() -> BitsAndBytesConfig:
    """Create the 4-bit quantization configuration used by QLoRA.

    Returns:
        BitsAndBytesConfig: Parameters for model quantization.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config() -> LoraConfig:
    """Create the LoRA adapter configuration for fine-tuning.

    Returns:
        LoraConfig: The PEFT configuration used in training.
    """
    return LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_training_args() -> SFTConfig:
    """Create the main training arguments for the SFT trainer.

    Returns:
        SFTConfig: Training configuration for the model.
    """
    return SFTConfig(
        output_dir=MICROSERVICES_MODEL_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        bf16=True,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        dataset_text_field="text",
        max_length=4096,
        packing=False,
        dataset_num_proc=6,
    )


def train_model() -> None:
    """Load the base model, train it with QLoRA and save the final adapter.

    Returns:
        None: This function does not return a value.
    """
    quantization_config = build_quantization_config()
    print("Loading the model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = build_lora_config()
    training_args = build_training_args()

    print("Loading the dataset...")
    dataset = load_dataset("json", data_files={"train": DATASET_PATH})

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        args=training_args,
    )

    print("Starting training...")
    last_checkpoint = get_last_checkpoint(MICROSERVICES_MODEL_DIR)

    if last_checkpoint is not None:
        print(f"Checkpoint found. Resuming from: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No checkpoint found. Starting from scratch...")
        trainer.train()

    trainer.model.save_pretrained(MICROSERVICES_MODEL_DIR)
    tokenizer.save_pretrained(MICROSERVICES_MODEL_DIR)
    print("Training finished and saved successfully!")


if __name__ == "__main__":
    train_model()