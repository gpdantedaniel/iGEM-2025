import pandas as pd
from Bio import SeqIO
import pandas as pd
import torch
from torch import nn
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.stats import pearsonr

device = 'cuda'

def parse_fasta(fasta_file):
    """Parse FASTA files into dataframes """
    data = {'id': [], 'description': [], 'sequence': [], 'length': []}

    for record in SeqIO.parse(fasta_file, "fasta"):
        data['id'].append(record.id)
        data['description'].append(record.description)
        data['sequence'].append(record.seq)
        data['length'].append(len(record.seq))

    return pd.DataFrame.from_dict(data)

def get_sequence(PDB, sequence_maps):
    """Returns a sequence from a PDF:sequence map dictionary"""
    for key in sequence_maps.keys():
        if key in PDB:
            return sequence_maps[key]
    return None

def insert_residue(sequence, residue, position):
    """Insert a residue at a given position in a sequence and return it"""
    sequence = list(sequence)
    sequence[position - 1] = residue
    return ''.join(sequence)

def read_S669():
    """
    Reads and returns the S669 dataset using the TSV file 
    and the FASTA file that both complement each other. 
    """

    df_tsv = pd.read_csv('S669.tsv', sep='\t')
    df_fasta = parse_fasta('S669.fasta')

    sequence_maps = {
        sequence_maps[row[-1]['id']]: row[-1]['sequence']
        for row in df_fasta.iterrows()
    }

    # Assign the wild-type sequences based on PDB ID
    df_tsv['wt_sequence'] = df_tsv.apply(lambda row: get_sequence(row['PDB'], sequence_maps), axis=1)
    df_tsv['length'] = df_fasta['length']

    # Construct the wild-type and mutant-type sequences
    df_tsv['wt_sequence'] = df_tsv.apply(lambda row: insert_residue(row['wt_sequence'], row['WT'], row['POS']), axis=1)
    df_tsv['mt_sequence'] = df_tsv.apply(lambda row: insert_residue(row['wt_sequence'], row['MT'], row['POS']), axis=1)

    # Retain only the essential columns
    df_tsv = df_tsv[['wt_sequence', 'mt_sequence', 'DDG']]

class DDGPredictor650M(nn.Module):
    """
    The achitecture class that describes the fully 
    connected regressor model that consumes embeddings.
    """
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, wild, mut):
        diff = mut - wild
        return self.head(diff).squeeze()

def embed_difference(model_name, df_s669):
    """
    Generates embeddings for wild-type and mutant-type
    sequences and returns their difference
    """
    # Load the model for encoding the embeddings
    model = SentenceTransformer(model_name)

    wild_embeddings = model.encode(
        df_s669['wt_sequence'].tolist(),
        convert_to_tensor=True,
        normalize_embeddings=False
    ).to(device)

    mutant_embeddings = model.encode(
        df_s669['mt_sequence'].tolist(),
        convert_to_tensor=True,
        normalize_embeddings=False
    ).to(device)

    return wild_embeddings, mutant_embeddings


def evaluate(model_checkpoint):
    """
    The main evaluation method that tests a model of a 
    given checkpoint against the S669 dataset
    """
    # Load the model onto GPU with model weights
    regressor = DDGPredictor650M().to('cuda')
    regressor.load_state_dict(torch.load(model_checkpoint, weights_only=True))
    regressor.eval()

    # Load the S669 evaluation dataset
    df_s669 = read_S669()

    wt_embeddings, mt_embeddings = embed_difference(
        model_name='facebook/esm2_t33_650M_UR50D',
        df_s669=df_s669
    )

    # Store the evaluation values
    rmses, maes, pccs = [], [], []

    # Calculate performance for both forward and reverse mutations
    for direction in ['forward', 'reverse']:
        match direction:
            case 'forward': 
                predicted_ddg = pd.Series([
                    regressor.forward(wild=wt_emb, mut=mt_emb).item()
                    for wt_emb, mt_emb in zip(wt_embeddings, mt_embeddings)
                ])
            case 'reverse':
                predicted_ddg = pd.Series([
                    regressor.forward(wild=mt_emb, mut=wt_emb).item()
                    for wt_emb, mt_emb in zip(wt_embeddings, mt_embeddings)
                ])

        # Calculate RMSE
        squared_errors = (df_s669['DDG'] - predicted_ddg) ** 2
        rmse = np.sqrt(squared_errors.mean())
        rmses.append(rmse)

        # Calculate MAE
        absolute_errors = np.abs(df_s669['DDG'] - predicted_ddg)
        mae = absolute_errors.mean()
        maes.append(mae)

        # Calculate PCC
        pcc, _ = pearsonr(df_s669['DDG'], predicted_ddg)
        pccs.append(pcc)

        # Print results to the terminal
        print(f'{direction} | rmse: {round(rmse, 4)} | mae: {round(mae, 4)} | pcc: {round(pcc, 4)}')
    
    # Calculate average values for forward and reverse predictions and print results
    total_rmse, total_mae, total_pcc = sum(rmses)/2, sum(maes)/2, sum(pccs)/2
    print(f'total   | rmse: {round(total_rmse, 4)} | mae: {round(total_mae, 4)} | pcc: {round(total_pcc, 4)}')

if __name__ == '__main__':
    evaluate('regressor/checkpoint-epoch-100.pt')
