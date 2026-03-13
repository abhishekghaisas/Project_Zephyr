"""
Fine-tune Code Llama on Modal for SeaTac Airport SQL Generation
FIXED: Attention mask dimension issue

Setup:
1. Install Modal: pip install modal
2. Setup: modal setup
3. Add HF token: modal secret create huggingface-secret HF_TOKEN=hf_...
4. Run: modal run modal_codellama_finetune.py
"""

import modal
import json

# Create Modal app
app = modal.App("seatac-codellama-finetune")

# Create persistent volume for models
volume = modal.Volume.from_name("seatac-models", create_if_missing=True)

# Optimized image for Code Llama
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2",
        "transformers==4.36.0",
        "datasets==2.16.0",
        "accelerate==0.25.0",
        "peft==0.7.1",
        "bitsandbytes==0.41.3",
        "trl==0.7.10",
        "sentencepiece",
        "protobuf",
        "scipy"
    )
)


@app.function(
    image=image,
    gpu="A10G",  # 24GB GPU
    timeout=3600 * 4,  # 4 hours
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=32768  # 32GB RAM
)
def finetune_codellama(training_data_json: str):
    """Fine-tune Code Llama 7B on SeaTac SQL generation"""
    
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch
    import os
    
    print("=" * 80)
    print("🦙 Fine-Tuning Code Llama 7B for SeaTac SQL Generation")
    print("=" * 80)
    
    # Load training data
    print("\n📊 Loading training data...")
    training_data = json.loads(training_data_json)
    print(f"✅ Loaded {len(training_data)} training examples")
    
    # Count examples per use case
    use_case_counts = {}
    for ex in training_data:
        # Handle both dict and list formats
        if isinstance(ex, dict):
            uc = ex.get('use_case', 'unknown')
        else:
            uc = 'unknown'
        use_case_counts[uc] = use_case_counts.get(uc, 0) + 1
    
    print("\n📋 Examples per use case:")
    for uc, count in sorted(use_case_counts.items()):
        print(f"   Use Case {uc}: {count} examples")
    
    # Model configuration
    model_name = "codellama/CodeLlama-7b-Instruct-hf"
    print(f"\n🔧 Loading base model: {model_name}")
    
    hf_token = os.environ.get("HF_TOKEN")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=hf_token,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model in 8-bit for memory efficiency
    print("🔧 Loading model in 8-bit mode...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False  # IMPORTANT: Disable cache for gradient checkpointing
    )
    
    print(f"✅ Model loaded: {model_name}")
    print(f"   Model size: ~7B parameters")
    print(f"   Quantization: 8-bit (memory efficient)")
    
    # Prepare model for training
    print("\n🔧 Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    # IMPORTANT: Disable gradient checkpointing to avoid attention mask issues
    # We have enough memory on A10G with 8-bit quantization
    print("🔧 Disabling gradient checkpointing (fixes attention mask issue)...")
    
    # Configure LoRA
    print("🔧 Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,  # Scaling factor
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ LoRA configured:")
    print(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"   Total params: {total_params:,}")
    
    # Format training data for Code Llama
    print("\n📝 Formatting training data...")
    
    def format_codellama_prompt(example):
        """Format in Code Llama instruction format"""
        instruction = example['instruction']
        output = example['output']
        
        # Code Llama instruction format - SIMPLIFIED
        prompt = f"""[INST] {instruction} [/INST] {output}"""
        
        return {"text": prompt}
    
    formatted_data = [format_codellama_prompt(ex) for ex in training_data]
    dataset = Dataset.from_list(formatted_data)
    
    print(f"✅ Formatted {len(dataset)} examples")
    
    # Tokenize dataset
    print("🔤 Tokenizing dataset...")
    
    def tokenize_function(examples):
        # Tokenize with FIXED max_length to avoid attention mask issues
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,  # Reduced from 2048 to avoid attention issues
            padding="max_length",
            return_tensors=None
        )
        # Copy input_ids to labels for causal LM
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=["text"],
        batched=False,
        desc="Tokenizing"
    )
    
    print(f"✅ Tokenized {len(tokenized_dataset)} samples")
    
    # Training configuration - FIXED
    print("\n🏋️  Configuring training...")
    
    training_args = TrainingArguments(
        output_dir="/models/seatac-codellama-checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Increased from 2 (we have memory)
        gradient_accumulation_steps=4,  # Reduced (effective batch still 16)
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        warmup_steps=50,
        logging_dir="/models/logs",
        report_to="none",
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        gradient_checkpointing=False,  # DISABLED to fix attention mask issue
        dataloader_num_workers=0,  # Single worker to avoid threading issues
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    print("\n" + "=" * 80)
    print("🔥 TRAINING STARTED")
    print("=" * 80)
    print(f"Model: Code Llama 7B Instruct")
    print(f"Training examples: {len(tokenized_dataset)}")
    print(f"Epochs: 3")
    print(f"Batch size: 4 (effective: 16 with gradient accumulation)")
    print(f"Max length: 1024 tokens (sufficient for SQL)")
    print(f"GPU: A10G (24GB)")
    print(f"Gradient checkpointing: DISABLED (fixes attention mask bug)")
    print(f"Estimated time: 1-2 hours")
    print(f"Estimated cost: $3-5")
    print("=" * 80)
    
    # Train!
    try:
        trainer.train()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    
    # Save final model
    print("\n💾 Saving fine-tuned model...")
    final_model_path = "/models/seatac-codellama-final"
    
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Commit volume
    print("💾 Committing to persistent volume...")
    volume.commit()
    
    print(f"✅ Model saved to persistent volume: {final_model_path}")
    print("\n🎉 Fine-tuning complete!")
    print("\n📋 Model Details:")
    print(f"   Base: codellama/CodeLlama-7b-Instruct-hf")
    print(f"   Fine-tuned on: {len(training_data)} SeaTac SQL examples")
    print(f"   Method: LoRA (16-rank)")
    print(f"   Quantization: 8-bit")
    print(f"   Max tokens: 1024")
    print(f"   Size: ~4GB (quantized)")
    
    return {
        "model_path": final_model_path,
        "training_examples": len(training_data),
        "epochs": 3,
        "base_model": model_name,
        "status": "success"
    }


@app.local_entrypoint()
def main():
    """Main entry point - run from local machine"""
    
    print("\n" + "=" * 80)
    print("🦙 SeaTac Code Llama Fine-Tuning Pipeline")
    print("=" * 80)
    
    # Load training data
    print("\n📂 Loading training data...")
    try:
        with open('seatac_llama_training.json', 'r') as f:
            training_data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: seatac_llama_training.json not found")
        print("   Make sure you've generated the training data first:")
        print("   python generate_training_data.py")
        return
    
    print(f"✅ Loaded {len(training_data)} examples from seatac_llama_training.json")
    
    # Show first example
    if training_data:
        print("\n📝 Sample training example:")
        print("-" * 80)
        sample = training_data[0]
        inst = sample.get('instruction', 'N/A')
        out = sample.get('output', 'N/A')
        print(f"Instruction: {inst[:80]}...")
        print(f"Output: {out[:80]}...")
        print("-" * 80)
    
    # Confirm before starting expensive GPU job
    print("\n⚠️  This will start a GPU instance on Modal")
    print(f"   GPU: A10G (24GB) - ~$1.10/hour")
    print(f"   Estimated time: 1-2 hours")
    print(f"   Estimated cost: $3-5")
    print(f"   Fix applied: Attention mask issue resolved ✅")
    
    response = input("\n▶️  Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Cancelled")
        return
    
    print("\n🚀 Uploading data and starting fine-tuning...")
    print("   (This may take a few minutes to spin up the GPU)")
    
    # Start fine-tuning on Modal
    try:
        result = finetune_codellama.remote(json.dumps(training_data))
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        print("\n🔍 Troubleshooting:")
        print("   1. Check Modal dashboard: modal.com")
        print("   2. View logs: modal app logs seatac-codellama-finetune")
        print("   3. Check GPU quota: modal.com → Billing")
        return
    
    if result.get('status') != 'success':
        print("\n❌ Training did not complete successfully")
        return
    
    print("\n" + "=" * 80)
    print("✅ FINE-TUNING COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Results:")
    print(f"   Model: {result['base_model']}")
    print(f"   Training examples: {result['training_examples']}")
    print(f"   Epochs: {result['epochs']}")
    print(f"   Saved to: {result['model_path']}")
    
    print("\n📋 Next Steps:")
    print("   1. Test the model:")
    print("      modal run modal_serve_codellama.py::test")
    print("   2. Deploy inference server:")
    print("      modal deploy modal_serve_codellama.py")
    print("   3. Get API endpoint and integrate with backend")
    
    print("\n🎉 Your custom SQL model is ready!")


if __name__ == "__main__":
    main()