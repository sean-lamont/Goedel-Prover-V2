import os
import time
import json
import ctypes
import resource
import tempfile
import traceback
import threading
import subprocess
import multiprocessing as mp
from pprint import pprint

import random

import numpy as np

def split_list_randomly(lst, k):
    random.shuffle(lst)  # Shuffle the list randomly
    return list(map(list, np.array_split(lst, k)))  # Split into k approximately equal parts


from prover.lean.ast_parser import lean4_parser
from prover.workers import ProcessScheduler
from prover.utils import AttrDict

HOME_DIR = os.path.expanduser('~')

DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'

DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

DEFAULT_IMPORTS = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


statement_sample = "\n/-- Show that $\frac{9x^2\\sin^2 x + 4}{x\\sin x} \\geq 12$ for $0 < x < \\pi$.-/\ntheorem aime_1983_p9 (x : ℝ) (h₀ : 0 < x ∧ x < Real.pi) :\n  12 ≤ (9 * (x ^ 2 * Real.sin x ^ 2) + 4) / (x * Real.sin x) :="

proof_code_sample_1 = " by\n  /-\n  To find the minimum value of $\frac{9x^2\\sin^2 x + 4}{x\\sin x}$ for $0 < x < \\pi$, we need to show that it is at least 12. We start by noting that the expression can be rewritten using the division property of inequalities. We then use the fact that \\$sin x$ and $x$ are positive in the given range to establish the necessary inequalities. Finally, we apply these results to conclude that the minimum value is indeed 12.\n  -/\n  -- We start by ensuring that the product x * sin x is positive in the given range.\n  have h₁ : 0 < x * Real.sin x := by\n    apply mul_pos\n    -- x is positive in the range (0, π).\n    exact h₀.1\n    -- sin x is positive in the range (0, π).\n    exact Real.sin_pos_of_pos_of_lt_pi h₀.1 h₀.2\n  -- Using the division property of inequalities, we rewrite the expression.\n  rw [le_div_iff h₁]\n  /- tactic state:\n    x : ℝ\n    h₀ : 0 < x ∧ x < π\n    h₁ : 0 < x * x.sin\n    ⊢ 12 * (x * x.sin) ≤ 9 * (x ^ 2 * x.sin ^ 2) + 4\n  -/\n  -- This is equivalent to showing that 9x^2 sin^2 x - 12x sin x + 4 ≥ 0, and the left hand side can be rewritten as a perfect square (3x sin x - 2)^2.\n  -- We use the fact that (3x sin x - 2)^2 is non-negative to establish this.\n  nlinarith [sq_nonneg (3 * x * Real.sin x - 2)]\n"

proof_code_sample_2 = " by sorry"

# proof_code_list_sample = [proof_code_sample] * 1
# proof_code_list_sample = [statement_sample + proof_code_sample_1, statement_sample + proof_code_sample_2] * 2

proof_code_list_sample = [{"name": "test_problem", "code": statement_sample + proof_code_sample_1}] + [{"name": "test_problem", "code": statement_sample + proof_code_sample_2}] * 2

problem_list_sample = [proof_code_list_sample] * 2 #each item in problem_list_sample is a proof_code_list which I want a single process to do

starttime = time.time()
# proof_code_list needs to have problem_id for each problem

def verify_lean4_proofs(proof_code_list, imports = DEFAULT_IMPORTS, lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE): # cpu?
    # result queue: a queue that all processes write output to it

    # Function to send a JSON command to the Lean REPL
    def send_command(proc,command,env=None):
        if env==None:
            json_cmd = json.dumps({"cmd": command})     
        else:  
            json_cmd = json.dumps({"cmd": command, "env": env})  # Convert command to JSON
        proc.stdin.write(json_cmd + "\n\n")  # Send command (double newline needed)
        proc.stdin.flush()  # Ensure it is sent
        # print(f"command {command} already sent")


    start_time = time.time()
    # Start Lean 4 REPL as a subprocess
    repl_proc = subprocess.Popen(
        [lake_path, "exe", "repl"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True, # Handles input/output as text
        cwd=lean_workspace
    )


    print(f"Initializing Lean REPL: (PID: {repl_proc.pid})", flush = True)
    
    # IMPORTS = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


    send_command(repl_proc,imports)

    for proof_code_dict in proof_code_list:
        proof_code = proof_code_dict["code"]
        # print("Sending command")
        send_command(repl_proc, proof_code, env=0)
        # send_command(repl_proc,proof_code)

    # Read output
    output, errors = repl_proc.communicate()

    lean_results = []

    # Split output by blank lines to get individual JSON objects
    json_blocks = output.strip().split("\n\n")

    for i, block in enumerate(json_blocks):
        if i==0: # the output of import
            continue
        block = block.strip()
        if block:  # Skip empty blocks
            code = imports + proof_code_list[i-1]["code"]
            problem_id = proof_code_list[i-1]["name"]
            try:
                result = json.loads(block)
                parsed_result = {
                    "sorries": result.get("sorries", []),
                    "tactics": result.get("tactics", []),
                    "errors": [m for m in result.get("messages", []) if m.get("severity") == "error"],
                    "warnings": [m for m in result.get("messages", []) if m.get("severity") == "warning"],
                    "infos": [m for m in result.get("messages", []) if m.get("severity") == "info"],
                    # "verified_code": code,
                    # "problem_id": problem_id
                    "system_errors": None
                }
                parsed_result["pass"] = not parsed_result["errors"]
                parsed_result["complete"] = (
                    parsed_result["pass"]
                    and not parsed_result["sorries"]
                    and not any(
                        "declaration uses 'sorry'" in warning["data"] or "failed" in warning["data"]
                        for warning in parsed_result["warnings"]
                    )
                )

            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}\nBlock: {block}")
                parsed_result = {
                    "pass": False,
                    "complete": False,
                    # "verified_code": code,
                    # "problem_id": problem_id,
                    "system_errors": f"JSON decoding error: {e}\nBlock: {block}"
                }
            
            lean_results.append({"name": problem_id, "code": code, "compilation_result": parsed_result})
    repl_proc.wait()
    # Close the process
    repl_proc.terminate()

    print(f"PID: {repl_proc.pid} verify time:", time.time() - start_time, flush = True)


    # # Store (index, results) in the queue
    # result_queue.put(lean_results)

    return lean_results

# def parallel_verification(problem_list, imports = DEFAULT_IMPORTS, lake_path =DEFAULT_LAKE_PATH, lean_workspace =DEFAULT_LEAN_WORKSPACE):
#     """Launch multiple Lean REPL processes, each handling a different problem on a separate CPU core."""
#     all_results = []

#     processes = []
#     # num_cores = min(len(problem_list), os.cpu_count())  # Limit to available CPU cores
#     result_queue = mp.Queue()

#     for proof_code_list in problem_list:
#         p = mp.Process(target=verify_lean4_proofs, args=(result_queue, proof_code_list, imports, lake_path, lean_workspace))
#         processes.append(p)
#         p.start()
    
#     # Wait for all processes to complete
#     for p in processes:
#         p.join()

#     print("All verification processes completed.")

#     while not result_queue.empty():
#         lean_results = result_queue.get()
#         all_results += lean_results
        
#     return all_results


def parallel_verification(problem_list, imports=DEFAULT_IMPORTS, lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE):
    """Launch multiple Lean REPL processes using multiprocessing.Pool."""
    
    # len(problem_list) should not be too large compare to num available cpus

    # num_cores = min(len(problem_list), mp.cpu_count())  # Limit to available CPU cores
    
    all_results = []

    num_proc = len(problem_list)

    # Prepare arguments for starmap
    args = [(proof_code_list, imports, lake_path, lean_workspace) for proof_code_list in problem_list]

    # Use multiprocessing Pool to manage worker processes
    with mp.Pool(processes = num_proc) as pool:
        results = pool.starmap(verify_lean4_proofs, args)  # Distribute tasks across cores

    # Combine all results
    for result in results:
        all_results += result  # Merge lists

    print("All verification processes completed.", flush = True)
    return all_results

# print(verify_lean4_proofs(statement_sample,proof_code_list_sample))


# print(verify_lean4_proofs(proof_code_list_sample))
if __name__ == '__main__':
    print(os.cpu_count())


    print(parallel_verification(problem_list_sample))

    print("time:",time.time()-starttime)

    # output_dir="/scratch/gpfs/st3812/aiformath/Deepseek/eval_results/minif2f/Goedel-Prover-SFT_long_form_thought_data_5k_Epoch2_LR1e-5"

    # input_file = output_dir + "/to_inference_codes.json"


    # compile_output_path=output_dir+ "/code_compilation_repl2.json"


    # with open(input_file, 'r') as json_file:
    #     codes = json.load(json_file)

    #     codes = codes[:800]
    #     # remove imports

    #     imports = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n"

    #     for code in codes:
    #         code["code"] = code["code"][len(imports):]

    #     to_compile_problems_list= split_list_randomly(codes, 16)

    #     to_compile_problems_list = [to_compile_problems_list[0]] * 16

    #     outputs_list = parallel_verification(to_compile_problems_list)


    # with open(compile_output_path, 'w') as json_file:
    #     json.dump(outputs_list, json_file, indent=4)

    # print("time:",time.time()-starttime)