from transformers import EsmTokenizer, EsmForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import os

# ========== Variables ==========

os.environ['HF_TOKEN'] = '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'
SPYCATCHER = 'VDTLSGLSSEQGPSGDMTIEEDSATHIKFSKRDEDGKELAGATMELRDSSGKTISTWISDGQVKDFYLYPGKYTFVETAAPDGYEVATAITFTVNEQGQVTVNGKATKGDAHI'

# ========== Metrics Computation ==========

def compute_metrics(eval_pred):
    """Calculates pseudo-perplexity on the eval set"""
    results = {}
    # Convert Hugging Face outputs to PyTorch tensors
    logits, labels = eval_pred
    logits = torch.tensor(logits)  # Shape: (batch_size, seq_len, vocab_size=33)
    labels = torch.tensor(labels)  # Shape: (batch_size, seq_len)
    
    # 1. Identify masked tokens (where labels != -100)
    mask = labels != -100  # Boolean mask for prediction positions
    num_masked_tokens = mask.sum().item()  # Count masked tokens
    
    # Handle edge case: no masked tokens
    if num_masked_tokens == 0:
        results['pppl_global'] = float('nan')
        return results

    # 2. Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
    
    # 3. Create safe labels: replace -100 with 0 (to avoid out-of-bounds errors)
    safe_labels = torch.where(labels == -100, 0, labels).long()
    
    # 4. Gather true token log probabilities
    token_log_probs = torch.gather(input=log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    
    # 5. Compute NLL only for masked tokens
    nll = -token_log_probs[mask]  # Shape: (num_masked_tokens,)
    
    # 6. Calculate pseudo-perplexity
    avg_nll = nll.mean()  # Average NLL over masked tokens
    pppl_global = torch.exp(avg_nll).item()  # exp(avg_nll)

    results['pppl_global'] = pppl_global
    return results

def train(dataset, base_model, output_model):
    print('Loadig model, tokenizer and dataset')
    # Step 1. Load our model, its tokenizer and the dataset
    tokenizer = EsmTokenizer.from_pretrained(base_model)
    model = EsmForMaskedLM.from_pretrained(base_model)
    tokenized_dataset = load_dataset(dataset)

    # Just for testing
    tokenized_dataset = tokenized_dataset['train'].train_test_split(train_size=100, test_size=10, shuffle=True) 

    # Step 2. Prepare the training utilities
    print('Preparing training arguments and trainer')
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=output_model,
        num_train_epochs=5,
        save_total_limit=5,
        save_strategy='epoch',
        eval_strategy='epoch',
        logging_strategy='steps',
        logging_steps=100,
        learning_rate=4e-4,
        weight_decay=0.01,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        push_to_hub=True,
        hub_model_id=output_model,
        report_to='tensorboard',
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    # Step 3. Train!
    trainer.train()

if __name__ == '__main__':
    dataset = "dantedgp/mscramm-100K-tokenized"
    base_model = "facebook/esm2_t33_650M_UR50D"
    output_model = "SpySeer-ESM2-650M-MSCRAMM-100K-test"
    
    train(dataset, base_model, output_model)


        



        

        

        

