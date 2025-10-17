import json
from pathlib import Path
from transformers import AutoTokenizer
import pandas as pd

def calculate_statistics():
    model_id = "Goedel-LM/Goedel-Prover-V2-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    processed_data_path = Path('processed_data')
    if not processed_data_path.is_dir():
        print(f"Directory not found: {processed_data_path}")
        return

    lengths = {
        'context': [],
        'prev_attempt': [],
        'target': []
    }

    for jsonl_file in processed_data_path.glob('*.jsonl'):
        print(f"Processing {jsonl_file.name}...")
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                for key in lengths.keys():
                    if key in data and data[key]:
                        tokenized_text = tokenizer(data[key])['input_ids']
                        lengths[key].append(len(tokenized_text))
    
    print("\n\nStatistics for token lengths:")
    for key, values in lengths.items():
        if values:
            print(f"\n--- {key} ---")
            series = pd.Series(values)
            print(series.describe())
        else:
            print(f"\n--- {key} ---")
            print("No data found.")

if __name__ == "__main__":
    calculate_statistics()
