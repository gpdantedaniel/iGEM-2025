import torch
from transformers import EsmTokenizer, EsmForMaskedLM

def calculate_pppl(model, tokenizer, sequence):
    """Calculates the pseudo-perplexity score of a sequence"""
    token_ids = tokenizer.encode(sequence, return_tensors='pt')
    input_length = token_ids.size(1)
    log_likelihood = 0.0

    for i in range(1, input_length - 1):
        # Create a copy of the token IDs
        masked_token_ids = token_ids.clone()
        # Mask a token that we will try to predict back
        masked_token_ids[0, i] = tokenizer.mask_token_id

        with torch.no_grad():
            output = model(masked_token_ids)
            logit_prob = torch.nn.functional.log_softmax(output.logits, dim=-1)
        
        log_likelihood += logit_prob[0, i, token_ids[0, i]]

    # Calculate the average log likelihood per token
    avg_log_likelihood = log_likelihood / (input_length - 2)

    # Compute and return the pseudo-perplexity
    pppl = torch.exp(-avg_log_likelihood)
    return pppl.item()

SPYCATCHER = 'VDTLSGLSSEQGPSGDMTIEEDSATHIKFSKRDEDGKELAGATMELRDSSGKTISTWISDGQVKDFYLYPGKYTFVETAAPDGYEVATAITFTVNEQGQVTVNGKATKGDAHI'
SPYTAG = 'RGVPHIVMVDAYKRYK' # SIXTEEN RESIDUES PEOPLE (Robots people, robots)

# Calculate PPPL for ESM2-650M
model_name = 'facebook/esm1v_t33_650M_UR90S_1'
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmForMaskedLM.from_pretrained(model_name)

print('ESM2-650M SpyCatcher:',calculate_pppl(model, tokenizer, SPYCATCHER))
print('ESM2-650M SpyTag:',calculate_pppl(model, tokenizer, SPYTAG))

# Calculate PPPL for optimizer
model_name = 'optimizer'
model = EsmForMaskedLM.from_pretrained(model_name)

print('Optimizer SpyCatcher:', calculate_pppl(model, tokenizer, SPYCATCHER))
print('Optimizer SpyTag:', calculate_pppl(model, tokenizer, SPYTAG))