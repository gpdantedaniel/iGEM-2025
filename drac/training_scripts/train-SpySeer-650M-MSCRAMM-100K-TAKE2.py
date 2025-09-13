import os
import torch
from transformers import EsmTokenizer, EsmForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_from_disk, load_dataset

os.environ['HF_HUB_OFFLINE'] = '1'

def compute_metrics(eval_pred):
    # Convert Hugging Face outputs to PyTorch tensors
    logits, labels = eval_pred
    logits = torch.tensor(logits)  # Shape: (batch_size, seq_len, vocab_size=33)
    labels = torch.tensor(labels)  # Shape: (batch_size, seq_len)
    
    # 1. Identify masked tokens (where labels != -100)
    mask = labels != -100  # Boolean mask for prediction positions
    num_masked_tokens = mask.sum().item()  # Count masked tokens
    
    # Handle edge case: no masked tokens
    if num_masked_tokens == 0:
        return {'pppl_global': float('nan')}

    # 2. Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
    
    # 3. Create safe labels: replace -100 with 0 (to avoid out-of-bounds errors)
    safe_labels = torch.where(labels == -100, 0, labels).long()
    
    # 4. Gather true token log probabilities
    token_log_probs = torch.gather(  # Gather along vocabulary dimension
        input=log_probs, 
        dim=-1, 
        index=safe_labels.unsqueeze(-1)  # Add extra dim for gathering
    ).squeeze(-1)  # Shape: (batch_size, seq_len)
    
    # 5. Compute NLL only for masked tokens
    nll = -token_log_probs[mask]  # Shape: (num_masked_tokens,)
    
    # 6. Calculate pseudo-perplexity
    avg_nll = nll.mean()  # Average NLL over masked tokens
    pppl_global = torch.exp(avg_nll).item()  # exp(avg_nll)
    
    return {'pppl_global': pppl_global}

def training_loop(dataset_path, base_model, output_model, batch_size=1):
    # Step 1. Load the model and its tokenizer
    tokenizer = EsmTokenizer.from_pretrained(base_model, local_files_only=True)
    model = EsmForMaskedLM.from_pretrained(base_model, local_files_only=True)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.TOKEN_CLS, 
        inference_mode=False, 
        r=2,
        lora_alpha=16, 
        target_modules=["query", "key", "value"],
        lora_dropout=0.2,
        bias="none"
    ))

    # Step 2. Load our pre-tokenized dataset
    tokenized_dataset = load_dataset(dataset_path)

    # Step 3. Prepare our training loop
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=f'{output_model}',
        num_train_epochs=5,
        save_strategy='epoch',
        save_total_limit=5,
        logging_strategy='steps',
        logging_steps=10,
        eval_strategy='epoch',
        learning_rate=4e-7,
        weight_decay=0.01,
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=batch_size,
        report_to='tensorboard',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['eval'],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    # Step 4. Train!
    trainer.train()

if __name__ == '__main__':
    training_loop(
        dataset_path='dantedgp/mscramm-100K-tokenized',
        base_model='facebook/esm2_t33_650M_UR50D',
        output_model='SpySeer-650M-MSCRAMM-100K-TAKE2',
        batch_size=12
    )