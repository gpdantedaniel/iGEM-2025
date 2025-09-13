# How-to: Optimizing protein fitness

## Dataset curation

Dataset curation for our fitness-optimizing model is performed using MMseqs2, a sequence alignment software. First, download the
fasta files fnbpb-spy.fasta, fnbpb-30k.fasta, and mscramm-227k.fasta. Then, download MMseqs2 and SeqKit (a tool for manipulating 
FASTA files). Lastly, run the Bash script MMseqs2_script.sh in your terminal using the commands:

```
chmod +x MMseqs2_script.sh
./MMseqs2_script.sh
```

Expected output: fasta file mscramm-100k.fasta

## Preprocessing the dataset

Once you have downloaded and curated your dataset, the next step is to transform it into a tokenized dataset that that can be readily used by an ESM-2 model for training. To do so, run the preprocessing.py file. This should also take care of splitting the data into a training split and an evaluation split. Run the script as follows from your terminal:

```
python preprocessing.py
```

Expected output: a folder called 'mscramm-100k-tokenized'

## Training the model

With your training data tokenized and ready to be consumed by a model, the next step is then to train ESM2-650M on your MSCRAMM sequences. To do so, run the training.py script to load the model with LoRA quantization and train it on your data for five epochs. Because the training script is configured to run with batches of 12 samples, this will require fairly heavy computational resources. A recommended setup includes 4 NVIDIA H100 GPUs. To run the script, use:

```
python training.py
```

Expected output: a folder called 'optimizer' that includes the fine-tuned model weights and training performance results stored as tensorboard-readable files 

## Evaluating the model

Once the model has been trained, it can be used to calcualte its PPPL score against the SpyTag and SpyCatcher sequences. Running the evaluation.py file runs in order ESM2-650M and the newly trained optimizer along with a PPPL calculation function to output scores for each sequence and for each model. To do so, run the evaluation.py script using:

```
python evaluation.py
```

