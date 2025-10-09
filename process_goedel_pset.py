from datasets import load_dataset
import json
import os
from tqdm import tqdm
import argparse


def process_goedel_pset(dataset_name, output_dir, num_files=10000):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the dataset
    dataset = load_dataset(dataset_name)['train']
    dataset = dataset.shuffle(seed=42)

    # Split the dataset into num_files parts
    chunk_size = len(dataset) // num_files
    for i in tqdm(range(num_files)):
        chunk = dataset[i*chunk_size : (i+1)*chunk_size] if i < num_files - 1 else dataset[i*chunk_size :]
        to_save = []

        for j in range(len(chunk['formal_statement'])):
            to_save.append({'problem_id': chunk['problem_id'][j], 'lean4_code': chunk['formal_statement'][j]})

        output_file = os.path.join(output_dir, f'goedel_pset_part_{i+1}.jsonl')

        with open(output_file, 'w') as f:
            for entry in to_save:
                f.write(json.dumps(entry) + '\n')


    print(f'Split {len(dataset)} entries into {num_files} files in {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Goedel Pset dataset and split into multiple JSON files.')
    parser.add_argument('--dataset_name', type=str, default='Goedel-LM/Goedel-Pset-v1', help='Input dataset name or path')
    parser.add_argument('--output_dir', type=str, default='goedel_pset_split', help='Output directory to save JSON files')
    parser.add_argument('--num_files', type=int, default=10000, help='Number of JSON files to split the dataset into')
    args = parser.parse_args()

    process_goedel_pset(args.dataset_name, args.output_dir, args.num_files)