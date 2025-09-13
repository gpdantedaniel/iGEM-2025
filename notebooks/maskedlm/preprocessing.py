from Bio import SeqIO
from datasets import Dataset, DatasetDict
from transformers import EsmTokenizer

model_name = 'facebook/esm2_t6_8M_UR50D'
tokenizer = EsmTokenizer.from_pretrained(model_name)

def fasta_to_ds(fasta_path):
    """
    Reads a fasta file and transcribes its
    contents into a HuggingFace Dataframe
    """
    data = {'id': [], 'description': [], 'sequence': [], 'length': []}

    with open(fasta_path) as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            seq_str = str(rec.seq)
            data['id'].append(rec.id)
            data['description'].append(rec.description)
            data['sequence'].append(seq_str)
            data['length'].append(len(seq_str))

    return Dataset.from_dict(data)

def preprocess_function(examples):
    """Returns the tokenization of a batch of examples"""
    return tokenizer(examples['sequence'], truncation=True, max_length=1024)

def preprocess_dataset(filename):
    """
    Preprocesses a FASTA file into an ESM2 token dataset
    divided into splits and pushes it to the HuggingFace Hub
    """
    # Dump FASTA contents into dataset
    dataset = fasta_to_ds(filename + '.fasta')
    # Tokenize the dataset's sequences
    dataset = dataset.map(function=preprocess_function, batched=True, remove_columns=dataset.column_names)

    # Split the dataset 
    train_eval = dataset.train_test_split(test_size=0.2, shuffle=True)
    # Format into a dataset divided into splits
    dataset = DatasetDict({'train': train_eval['train'], 'eval': train_eval['test']})

    # Format name and save to disk and Hub
    dataset_name = filename + '-tokenized'
    dataset.save_to_disk(dataset_name)
    dataset.push_to_hub(dataset_name, private=False)


if __name__ == '__main__':
    # The names of the FASTA files to be processed into tokenized datasets
    fasta_datasets = ['fnbp-spy', 'fnbp-30K', 'mscramm-227k']

    for filename in fasta_datasets:
        preprocess_dataset(filename)    