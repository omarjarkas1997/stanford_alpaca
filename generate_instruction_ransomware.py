import json
import os
import tqdm
import random
from rouge_score import rouge_scorer

# Path to the seed tasks JSONL file
seed_tasks_path = "./seed_tasks.json"
output_dir="test_1"
num_instructions_to_generate=100
num_prompt_instructions=3

# Load the tasks from the file and parse each line as a JSON object
with open(seed_tasks_path, "r") as file:
    seed_tasks = [json.loads(line) for line in file]

# Pretty print the seed tasks in a human-readable JSON format
print("Seed Tasks (Pretty Printed):")
print(json.dumps(seed_tasks[0], indent=4))

# Extract the seed instruction data (instruction, input, output) from seed tasks
seed_instruction_data = [
    {
        "instruction": task["instruction"],
        "input": task["instances"][0]["input"],
        "output": task["instances"][0]["output"]
    }
    for task in seed_tasks
]

# Pretty print the seed instruction data in a human-readable JSON format
print("\nSeed Instruction Data (Pretty Printed):")
print(json.dumps(seed_instruction_data[0], indent=4))


os.makedirs(output_dir, exist_ok=True)
request_idx = 0
# load the LM-generated instructions
machine_instruction_data = []


# similarities = {}
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


# first we tokenize all the seed instructions and generated machine instructions
all_instructions = [d["instruction"] for d in seed_instruction_data] + [
    d["instruction"] for d in machine_instruction_data
]
all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

print(all_instruction_tokens)

prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)

print("Seed Tasks (Pretty Printed):")
print(json.dumps(seed_tasks[0], indent=4))