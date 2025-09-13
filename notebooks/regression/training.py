import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader, Dataset
from tqdm import tqdm
import gc

device = 'cuda'

class ProteinDDGDataset(Dataset):
    """
    A Dataset class that contains wild-type sequence embeddings,
    mutant-type sequence embeddings and corresponding DDG values.
    """
    def __init__(self, wild_embeddings, mutant_embeddings, ddg):
        self.wild = torch.tensor(wild_embeddings, dtype=torch.float32)
        self.mut = torch.tensor(mutant_embeddings, dtype=torch.float32)
        self.ddg = torch.tensor(ddg, dtype=torch.float32)

    def __len__(self):
        return len(self.ddg)
    
    def __getitem__(self, idx):
        return self.wild[idx], self.mut[idx], self.ddg[idx]
    
def create_loader(data, subset, batch_size, shuffle=False):
    """
    Creates and returns a DataLoader of a
    subset from a larger dataset
    """
    split = ProteinDDGDataset(
        data['wild_embeddings'][subset.indices],
        data['mutant_embeddings'][subset.indices],
        data['ddg'][subset.indices]
    )

    return DataLoader(split, batch_size, shuffle)

def train_model(model_class, train_loader, val_loader, model_name, epochs=100):
    """
    The main method that trains the regressor model. Upon completion,
    it generates a folder model_name that holds the weights of the last
    five epoch checkpoints along with Tensorboard-generated training data
    """
    model = model_class().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    outputs_path = f'{model_name}'
    writer = SummaryWriter(outputs_path)

    for epoch in tqdm(range(epochs)):
        # Training
        model.train()
        train_loss = 0.0
        all_preds_train = []    # Store predictions
        all_targets_train = []  # Store ground truth

        for wild, mut, ddg in train_loader:
            wild, mut, ddg = wild.to(device), mut.to(device), ddg.to(device)
            optimizer.zero_grad()
            outputs = model(wild, mut)
            loss = criterion(outputs, ddg)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * wild.size(0)
            
            # Store batch predictions and targets
            all_preds_train.append(outputs.detach().cpu())
            all_targets_train.append(ddg.detach().cpu())

        # Concatenate all batches
        all_preds_train = torch.cat(all_preds_train).flatten()
        all_targets_train = torch.cat(all_targets_train).flatten()
        
        # Calculate training metrics
        train_loss /= len(train_loader.dataset)
        train_rmse = torch.sqrt(torch.tensor(train_loss))
        train_pearson = torch.corrcoef(torch.stack([all_preds_train, all_targets_train]))[0, 1].item()

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds_val = []  # Store predictions
        all_targets_val = []  # Store ground truth
        
        with torch.no_grad():
            for wild, mut, ddg in val_loader:
                wild, mut, ddg = wild.to(device), mut.to(device), ddg.to(device)
                outputs = model(wild, mut)
                loss = criterion(outputs, ddg)
                val_loss += loss.item() * wild.size(0)
                
                # Store batch predictions and targets
                all_preds_val.append(outputs.detach().cpu())
                all_targets_val.append(ddg.detach().cpu())

        # Concatenate all batches
        all_preds_val = torch.cat(all_preds_val).flatten()
        all_targets_val = torch.cat(all_targets_val).flatten()
        
        # Calculate validation metrics
        val_loss /= len(val_loader.dataset)
        val_rmse = torch.sqrt(F.mse_loss(all_preds_val, all_targets_val)).item()
        val_pearson = torch.corrcoef(torch.stack([all_preds_val, all_targets_val]))[0, 1].item()

        # Log all metrics to TensorBoard
        writer.add_scalars('MSE', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('RMSE', {'train': train_rmse, 'val': val_rmse}, epoch)
        writer.add_scalars('Pearson', {'train': train_pearson, 'val': val_pearson}, epoch)
        
        # Log histograms of parameters
        if epoch % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch)

        # Print statistics
        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | '
              f'Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | '
              f'Train Pearson: {train_pearson:.4f} | Val Pearson: {val_pearson:.4f}')
        
        # Save the last five models
        if epoch in list(range(epochs -5, epochs)):
            torch.save(model.state_dict(), outputs_path + f'/checkpoint-epoch-{epoch+1}.pt')

    writer.close()
    del model
    torch.cuda.empty_cache()

def training_loop(dataset_path, model_class, model_name, batch_size, epochs=100):
    """
    The main training loop method that handles
    data loading, splitting and model training.
    """
    # Load the training data
    data = torch.load(dataset_path)
    
    dataset = ProteinDDGDataset(
        wild_embeddings=data['wild_embeddings'],
        mutant_embeddings=data['mutant_embeddings'],
        ddg=data['ddg']
    )

    # Split dataset in a 9:1 ratio
    train_size = int(0.90 * len(dataset))
    test_size = len(dataset) - train_size

    # Generate reproducible split
    generator = torch.Generator().manual_seed(42)

    # Generate the subsets of the splits
    train_subset, eval_subset = random_split(
        dataset=dataset, 
        lengths=[train_size, test_size],
        generator=generator
    )

    # Create the DataLoaders for the splits
    train_loader = create_loader(data, train_subset, batch_size, shuffle=True)
    eval_loader = create_loader(data, eval_subset, batch_size, shuffle=False)

    # Log dataset and subset metrics
    print(f"Total samples: {len(dataset)}")
    print(f"Train: {len(train_subset)} ({len(train_subset)/len(dataset):.1%})")
    print(f"Validation: {len(eval_subset)} ({len(eval_subset)/len(dataset):.1%})")

    del dataset
    gc.collect()
    torch.cuda.empty_cache()

    # And at last, train the model!
    train_model(model_class, train_loader, eval_loader, model_name, epochs)

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

if __name__ == '__main__':
    training_loop(
        dataset_path='embeddings.pt',
        model_class=DDGPredictor650M,
        model_name='regressor',
        batch_size=2,
        epochs=100
    )