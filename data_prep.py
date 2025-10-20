import json

import re
# loop through all directories. Get to_inference_codes.json and set context/target. Get to_inference_codes_corrk.json for all k and set context target with message history:
from pathlib import Path
from transformers import AutoTokenizer

model_id = "Goedel-LM/Goedel-Prover-V2-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def process_data(path):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found {path}")

    corr_pattern = re.compile(r"^to_inference_codes_corr\d+.json$")

    data = []
    for entry in path.iterdir():
        if entry.is_file():
            if corr_pattern.match(entry.name):
                data.extend(load_correction_attempt(entry))
            elif "to_inference_codes.json" in entry.name:
                data.extend(load_original_attempt(entry))

    return data

def load_original_attempt(path):
    data = json.load(open(path))
    if data:
        return [{'context': d['model_input'], 'target': d['model_output'], 'prev_attempt': "", 'type': 'initial_attempt'} for d in data]
    else:
        return []

def load_correction_attempt(path):
    data = json.load(open(path))
    if data:

        return [{'context': tokenizer.apply_chat_template([ d['messages_history_list'][0], data[0]['messages_history_list'][-1]], tokenize=False, add_generation_prompt=False), 'target': d['model_output'], 'prev_attempt': tokenizer.apply_chat_template([ d['messages_history_list'][-2]], tokenize=False, add_generation_prompt=False), 'type': 'correction'} for d in data]
    else:
        return []



# test_data = process_data('results/run_goedel_pset/goedel_pset_part_100')

if __name__ == "__main__":
    # loop through all directories in run_goedel_pset and save data as separate jsonl file
    base_path = Path('results/run_goedel_pset')
    output_base_path = Path('processed_data')
    output_base_path.mkdir(exist_ok=True)

    for pset_dir in base_path.iterdir():
        if pset_dir.is_dir():
            print(f"Processing directory: {pset_dir.name}")
            processed_data = process_data(pset_dir)

            output_file_path = output_base_path / f"{pset_dir.name}.jsonl"
            with open(output_file_path, 'w') as f:
                for entry in processed_data:
                    f.write(json.dumps(entry) + '\n')
            print(f"Saved processed data to {output_file_path}")



# print ('context \n\n')
# print (test_data[-1]['context'])
# print ('prev \n\n')
# print (test_data[-1]['prev_attempt'])
# print ('target \n\n')
# print (test_data[-1]['target'])

