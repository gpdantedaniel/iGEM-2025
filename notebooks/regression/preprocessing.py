from sentence_transformers import SentenceTransformer
import pandas as pd
import torch

spysmith_df = pd.read_csv('spysmith.csv')
# Start with the largest value to help determine the most optimal batch size
spysmith_df = spysmith_df.sort_values(by='wild_seq', key=lambda s: s.str.len(), ascending=False)
# The target values will remain the same across datasets
ddg = torch.tensor(spysmith_df["ddg"].values, dtype=torch.float32)

def generate_embeddings(model_name, output_filename, batch_size):
    """
    Generates embeddings for native and mutant sequences
    and stores them along with DDG values in a '.pt' file.
    """
    # Load the model for encoding the embeddings
    model = SentenceTransformer(model_name)

    wild_embeddings = model.encode(
        spysmith_df['wild_seq'].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=batch_size,  
        normalize_embeddings=False
    )

    mutant_embeddings = model.encode(
        spysmith_df['mutant_seq'].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=batch_size,  
        normalize_embeddings=False
    )

    data = {
        "wild_embeddings": wild_embeddings, 
        "mutant_embeddings": mutant_embeddings,
        "ddg": ddg
    }

    torch.save(data, f'{output_filename}.pt')

if __name__ == '__main__':
    generate_embeddings(
        model_name='facebook/esm2_t33_650M_UR50D', 
        output_filename='embeddings',
        batch_size=4
    )


