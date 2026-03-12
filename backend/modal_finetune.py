"""
Fine-tune Gemma 2 on Modal for SeaTac Airport Operations
Run: modal run modal_finetune.py

Requirements:
- Modal account: modal setup
- HuggingFace token: modal secret create huggingface-secret HF_TOKEN=hf_...
- Training data: seatac_llama_training.json
"""

import modal
import json

# Create Modal app
app = modal.App("seatac-gemma-finetune")

# Create persistent volume for models
volume = modal.Volume.from_name("seatac-models", create_if_missing=True)

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2.0",
        "scipy",
        "torch==2.1.0",
        "transformers==4.45.0",
        "datasets==2.15.0",
        "accelerate==0.35.0",
        "peft==0.7.0",
        "bitsandbytes==0.41.3",
        "trl==0.7.4",
        "sentencepiece",
        "protobuf"
    )
)

@app.function(
    image=image,
    gpu="A100",  # 40GB GPU
    timeout=3600 * 6,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def finetune_gemma(training_data_json: str):
    """Fine-tune Gemma 2 on SeaTac operations data"""
    
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    import torch
    
    print("=" * 70)
    print("💎 Fine-Tuning Gemma 2 for SeaTac Operations")
    print("=" * 70)
    
    # Load training data from parameter
    print("\n📊 Loading training data...")
    training_data = json.loads(training_data_json)
    print(f"✅ Loaded {len(training_data)} training examples")
    
    # Load base model
    model_name = "google/gemma-2-2b-it"
    print(f"\n🔧 Loading base model: {model_name}")
    
    import os
    hf_token = os.environ.get("HF_TOKEN")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        low_cpu_mem_usage=True,
        use_cache=False
    )
    
    print(f"✅ Model loaded: {model_name}")
    
    # Enable gradient checkpointing
    print("\n🔧 Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()
    
    # Configure LoRA
    print("🔧 Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("✅ LoRA configured")
    
    # Format training data
    print("\n📝 Formatting training data...")
    
    def format_prompt(example):
        instruction = example['instruction']
        output = example['output']
        
        prompt = f"""<bos><start_of_turn>user
Generate SQL for SeaTac airport operations:

{instruction}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn><eos>"""
        
        return {"text": prompt}
    
    formatted_data = [format_prompt(ex) for ex in training_data]
    dataset = Dataset.from_list(formatted_data)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=1024, 
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, remove_columns=["text"])
    
    print(f"✅ Prepared {len(tokenized_dataset)} training samples")
    
    # Training configuration
    print("\n🏋️  Starting training...")
    training_args = TrainingArguments(
        output_dir="/models/seatac-gemma-checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=1,
        warmup_steps=50,
        logging_dir="/models/logs",
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch_fused"
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    print("\n🔥 Training started...")
    print("Estimated time: 1-2 hours on A100 GPU")
    print("=" * 70)
    
    trainer.train()
    
    print("\n✅ Training complete!")
    
    # Save model
    print("\n💾 Saving fine-tuned model...")
    final_model_path = "/models/seatac-gemma-finetuned"
    
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    volume.commit()
    
    print(f"✅ Model saved to: {final_model_path}")
    print("\n🎉 Fine-tuning complete!")
    
    return final_model_path


@app.local_entrypoint()
def main():
    """Run fine-tuning job"""
    import json
    
    print("🚀 Starting Modal fine-tuning job...")
    
    # Load local training data
    with open('seatac_llama_training.json', 'r') as f:
        training_data = json.load(f)
    
    print(f"📤 Uploading {len(training_data)} examples to Modal...")
    
    model_path = finetune_gemma.remote(json.dumps(training_data))
    
    print(f"\n✅ Fine-tuned model saved at: {model_path}")
    print("\n📋 Next steps:")
    print("   1. Deploy inference server: modal deploy modal_serve.py")
    print("   2. Get API endpoint URL")
    print("   3. Update backend to use the endpoint")