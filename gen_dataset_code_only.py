# Convert processed dataset into code only targets/previous contexts

import json
from pathlib import Path
import re


def extract_code(inputs):
    import_head = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"
    pattern = r'```lean4\n(.*?)\n```'
    matches = re.findall(pattern, inputs, re.DOTALL)
    if matches:
        return import_head + matches[-1]
    pattern = r'```lean4\n(.*?)```'
    matches = re.findall(pattern, inputs, re.DOTALL)
    if matches:
        return import_head + matches[-1]
    pattern = r'```lean\n(.*?)```'
    matches = re.findall(pattern, inputs, re.DOTALL)
    if matches:
        return import_head + matches[-1]
    return "None"


def process_data(path):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found {path}")

    output_base_path = Path('processed_data_code_only')
    output_base_path.mkdir(exist_ok=True)

    for entry in path.iterdir():
        output_file_path = output_base_path / f"{entry.name}.jsonl"
        data = []
        if entry.is_file():
            # load jsonl file, and replace prev_attempt and target fields
            with open(entry, 'r') as f:
                for line in f:
                    d = json.loads(line)
                    d['target'] = extract_code(d['target'])
                    if d['prev_attempt']:
                        d['prev_attempt'] = extract_code(d['prev_attempt'])
                        print(f'target: {d['target']}\n\n')
                        print(f'prev_attempt: {d['prev_attempt']}')
                    else:
                        d['prev_attempt'] = ''

                    data.append(d)

            with open(output_file_path, 'w') as f:
                for d in data:
                    f.write(json.dumps(d) + '\n')



# main

if __name__ == "__main__":
    process_data('processed_data')