
import pandas as pd
import requests, json
from tqdm import tqdm

base_url = 'https://rest.uniprot.org/uniprotkb/search?fields=sequence&query=accession_id:{accession_id}&format=json'

def get_sequence(accession_id):
    """
    Get a sequence from UNIPROT based on its accession ID
    """
    url = base_url.format(accession_id=accession_id)
    response = json.loads(requests.get(url).text)
    sequence = response['results'][0]['sequence']['value']
    return sequence

def construct_mutant(sequence, position, mutation):
    """ 
    Constructs the mutant sequence using the
    original sequence, the mutant and its position.
    """
    try:
        sequence = list(sequence)
        sequence[position - 1] = mutation
        sequence = ''.join(sequence)
        return sequence
    
    except Exception as e: 
        print(f'error for {''.join(sequence)} {position} {mutation}', e)
        return None
    
def parse_megascale(mut_type):
    """
    Parses mutations for Tsuboyama's dataset.
    Assumes that there are no "wt" values.
    """
    if mut_type.startswith('ins'):
        # Handles insertion (example: insA13)
        wild_aa = ""
        location = int(mut_type[4:])
        mutant_aa = mut_type[3]
    elif mut_type.startswith('del'):
        # Handles deletion (example: delD1)
        wild_aa = mut_type[3]
        location = int(mut_type[4:])
        mutant_aa = ""
    else:
        # Handles substitution (example: A11B)
        wild_aa = mut_type[0]
        location = int(mut_type[1:-1])
        mutant_aa = mut_type[1:-1]

    return wild_aa, location, mutant_aa

def construct_wild(mutant_sequence, mutation_code):
    """
    Reconstructs the wild-type sequence from
    a mutant sequence and the mutation code.
    """
    try:
        sequence = list(mutant_sequence)
        for mutation in mutation_code.split(':'):
            wild_aa, location, mutant_aa = parse_megascale(mutation)
            index = location - 1 

            if mutant_aa == "":
                # Handle a deletion
                sequence[index:index] = wild_aa
            else:
                # Handle substitution or insertion
                sequence[location - 1] = wild_aa
                
        sequence = ''.join(sequence)
        return sequence
    except Exception as e: 
        print(f'error for {''.join(mutant_sequence)} {mutation_code}', e)
        return None
    
def parse_mutation(mutation_code):
    """
    Parses a mutation's location, wild-type
    and mutant-type amino acids
    """
    location = int(''.join([char for char in mutation_code if char.isdigit()]))
    wild_aa = mutation_code.split(str(location))[0]
    mutant_aa = mutation_code.split(str(location))[-1]

    if len(mutant_aa) > 1:
        raise Exception('mutant_aa longer than 1 char:', mutant_aa)

    return wild_aa, location, mutant_aa

def generate_mutant(sequence, mutation_code):
    """
    Encodes the wild-type and the mutant
    residues within the protein sequence.
    """
    try:
        sequence = list(sequence)
        mutation_code = mutation_code.replace(';', ',').strip(',')

        for code in [code.strip() for code in mutation_code.split(',')]:
            # Apply each mutation code to the sequence
            wtaa, location, mtaa = parse_mutation(code)
            sequence[location - 1] = mtaa
            
        sequence = ''.join(sequence)
        return sequence
    
    except Exception as e: 
        print(f'error for {''.join(sequence)} {mutation_code}', e)
        return None


def build_dataset():
    """
    Builds the training dataset for regressor models by combing and 
    preprocessing FireProtDB, Tsuboyama's Megascale Dataset and ThermoMuteDB
    """
    # STEP 1: Processing FireProtDB

    fireprotdb_df = pd.read_csv('fireprotdb.csv')
    # Filter out empty DDG values
    fireprotdb_df = fireprotdb_df.dropna(subset=['ddG'])
    # Keep the essential columns
    fireprotdb_df = fireprotdb_df[['position','mutation','sequence', 'ddG']]
    # Rename the sequence column to wild_seq
    fireprotdb_df = fireprotdb_df.rename(columns={'sequence': 'wild_seq', 'ddG': 'ddg'})

    # Construct the mutant sequences using information about mutations and wild sequences
    fireprotdb_df['mutant_seq'] = fireprotdb_df.apply(lambda row: construct_mutant(row['wild_seq'], row['position'], row['mutation']), axis=1, result_type='expand')
    # Drop the intermediary position and mutation columns
    fireprotdb_df = fireprotdb_df.drop(columns=['position', 'mutation'])

    # STEP 2. Processing Tsuboyama's megascale dataset

    tsuboyama_df = pd.read_csv('Tsuboyama2023_Dataset2_Dataset3_20230416.csv')
    # Remove unreliable DDG samples and wild-type mutations
    tsuboyama_df = tsuboyama_df[(tsuboyama_df['ddG_ML'] != '-') & (tsuboyama_df['mut_type'] != 'wt')]
    # Rename aa_seq to mutant_seq
    tsuboyama_df = tsuboyama_df.rename(columns={'aa_seq': 'mutant_seq', 'ddG_ML': 'ddg'})

    tsuboyama_df['wild_seq'] = tsuboyama_df.apply(lambda row: construct_wild(row['mutant_seq'], row['mut_type']), axis=1, result_type='expand')
    # Only keep the relevant columns
    tsuboyama_df = tsuboyama_df[['wild_seq', 'mutant_seq', 'ddg']]

    # STEP 3: Processing ThermoMuteDB
    thermomute_df = pd.read_json('thermomutdb.json')
    # Filter between 250K and 350K between pH 6 and pH 8
    thermomute_df = thermomute_df[(thermomute_df['temperature'] > 250) & (thermomute_df['temperature'] < 350) & (thermomute_df['ph'] > 6) & (thermomute_df['ph'] < 8)]
    # Remove where accesion ID is "-""
    thermomute_df = thermomute_df[thermomute_df['uniprot'] != '-']
    # The accession ID p01892 was changed to P04439
    thermomute_df.loc[thermomute_df['uniprot'] == 'P01892', 'uniprot'] = 'P04439'
    # Drop null ddg values
    thermomute_df = thermomute_df.dropna(subset=['ddg'])  

    ids_sequences = {}

    for accession_id in tqdm(set(thermomute_df['uniprot'])):
        try:
            sequence = get_sequence(accession_id)
            ids_sequences[accession_id] = sequence
        except Exception as e:
            print(f'Exception for {accession_id}:', e)

    # Assign sequences to every row
    thermomute_df['wild_seq'] = thermomute_df['uniprot'].apply(lambda accession_id: ids_sequences[accession_id])
    # Generate mutant sequences using information from mutation codes and wild sequences
    thermomute_df['mutant_seq'] = thermomute_df.apply(lambda row: generate_mutant(row['wild_seq'], row['mutation_code']), axis=1, result_type='expand')

    # Save the mutations, the encoded sequences and ddg
    thermomute_df = thermomute_df[['wild_seq', 'mutant_seq', 'ddg']]
    # Drop empty values
    thermomute_df = thermomute_df.dropna()
    # Flip the DDG values
    thermomute_df['ddg'] = thermomute_df['ddg'].apply(lambda ddg: -ddg)

    # STEP 4: Joining all the datasets together

    # Concatenate all of our datasets together
    spysmith_df = pd.concat([fireprotdb_df, tsuboyama_df, thermomute_df])
    # Convert all the DDG values to float
    spysmith_df['ddg'] = spysmith_df['ddg'].apply(lambda ddg: float(ddg))
    # Keep all sequences that are less than or equal to 1022 amino acids
    spysmith_df = spysmith_df[spysmith_df['wild_seq'].apply(lambda s: len(s) <= 1022)]
    # Keep sequences whose DDG values lie in [-10, 10]
    spysmith_df = spysmith_df[(spysmith_df['ddg'] < 10) & (spysmith_df['ddg'] > -10)]
    # Drop any duplicates
    spysmith_df = spysmith_df.drop_duplicates()

    # And save at last to a spysmith.csv file
    spysmith_df.to_csv('spysmith.csv', index=False)

if __name__ == '__main__':
    build_dataset()