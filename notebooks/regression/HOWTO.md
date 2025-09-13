# How to train for regression

## 1. Building the dataset

The building_dataset.ipynb notebook is used to concatenate and filter three datasets used for predicting DDG from mutations: ThermoMuteDB, FireProtDB and Tsuboyama's Megascale Dataset. The script reads each of these and combines them, removing any duplicates and filtering for appropriate length, into a single dataset for training. It is then saved locally as a CSV file.

## 2. Preprocessing the dataset

The preprocessing.iypnb notebook can then be used to read the sequences from the previously assembled dataset to generate embeddings from ESM2-8M, ESM2-35M, ESM2-150M and ESM2-650M. These embeddings are stored along with the DDG values in .pt files locally for training later.

## 3. Training the model

With the preprocessed data and the training.ipynb notebook in hand, four regressor models are initialized to accept embeddings from each fo the four ESM2 variants used to generate embeddings. These are then trained sequentially on the locally loaded embeddings, and their final weights along with their training results, written in tensorboard-readable files, are stored locally.

## 4. Evaluating the model

The resulting regressor models can then be tested using the S669 datasets whose samples are converted into embeddings within the evaluation.ipynb notebook. Results of interest that are computed into the MSE, MAE and PCC. 


