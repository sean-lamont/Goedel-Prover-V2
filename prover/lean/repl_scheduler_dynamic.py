import os
import sys
import time
import json
import ctypes
import tempfile
import traceback
import threading
import pexpect
import subprocess
import multiprocessing as mp
from pprint import pprint
import random
import numpy as np

from prover.lean.ast_parser import lean4_parser
from prover.workers import ProcessScheduler
from prover.utils import AttrDict

# Helper function to split a list randomly into k parts.
def split_list_randomly(lst, k):
    random.shuffle(lst)
    return list(map(list, np.array_split(lst, k)))

# Sample proof code
# statement_sample = (
#     "\n/-- Show that $\\frac{9x^2\\sin^2 x + 4}{x\\sin x} \\geq 12$ for $0 < x < \\pi$.-/\n"
#     "theorem aime_1983_p9 (x : \u211d) (h\u2080 : 0 < x \u2227 x < Real.pi) :\n"
#     "  12 \u2264 (9 * (x ^ 2 * Real.sin x ^ 2) + 4) / (x * Real.sin x) :="
# )

statement_sample = "\n/-- Show that $\frac{9x^2\\sin^2 x + 4}{x\\sin x} \\geq 12$ for $0 < x < \\pi$.-/\ntheorem aime_1983_p9 (x : ℝ) (h₀ : 0 < x ∧ x < Real.pi) :\n  12 ≤ (9 * (x ^ 2 * Real.sin x ^ 2) + 4) / (x * Real.sin x) :="


# proof_code_sample_1 = (
#     " by\n  /-\n  To find the minimum value of $\n"
#     "  ... (proof details) ...\n"
#     "  -/\n  have h\u2081 : 0 < x * Real.sin x := by\n    apply mul_pos\n"
#     "    exact h\u2080.1\n    exact Real.sin_pos_of_pos_of_lt_pi h\u2080.1 h\u2080.2\n"
#     "  rw [le_div_iff h\u2081]\n  nlinarith [sq_nonneg (3 * x * Real.sin x - 2)]\n"
# )

proof_code_sample_1 = " by\n  /-\n  To find the minimum value of $\frac{9x^2\\sin^2 x + 4}{x\\sin x}$ for $0 < x < \\pi$, we need to show that it is at least 12. We start by noting that the expression can be rewritten using the division property of inequalities. We then use the fact that \\$sin x$ and $x$ are positive in the given range to establish the necessary inequalities. Finally, we apply these results to conclude that the minimum value is indeed 12.\n  -/\n  -- We start by ensuring that the product x * sin x is positive in the given range.\n  have h₁ : 0 < x * Real.sin x := by\n    apply mul_pos\n    -- x is positive in the range (0, π).\n    exact h₀.1\n    -- sin x is positive in the range (0, π).\n    exact Real.sin_pos_of_pos_of_lt_pi h₀.1 h₀.2\n  -- Using the division property of inequalities, we rewrite the expression.\n  rw [le_div_iff h₁]\n  /- tactic state:\n    x : ℝ\n    h₀ : 0 < x ∧ x < π\n    h₁ : 0 < x * x.sin\n    ⊢ 12 * (x * x.sin) ≤ 9 * (x ^ 2 * x.sin ^ 2) + 4\n  -/\n  -- This is equivalent to showing that 9x^2 sin^2 x - 12x sin x + 4 ≥ 0, and the left hand side can be rewritten as a perfect square (3x sin x - 2)^2.\n  -- We use the fact that (3x sin x - 2)^2 is non-negative to establish this.\n  nlinarith [sq_nonneg (3 * x * Real.sin x - 2)]\n"

proof_code_sample_2 = " by sorry"

proof_code_sample_3 = " hhh"


sample_task1 = {"name": "test_problem", "code": statement_sample + proof_code_sample_1}
sample_task2 = {"name": "sorry_problem", "code": statement_sample + proof_code_sample_2}
sample_task3 = {"name": "wrong_problem", "code": statement_sample + proof_code_sample_3}
sample_batch = [sample_task1, sample_task2, sample_task3] * 1 

def initiate_child(scheduler):
    child = pexpect.spawn(
        "/bin/bash",
        cwd=scheduler.lean_workspace,
        encoding='utf-8',
        maxread=1,
        echo=False
    )
    child.sendline("stty -icanon")
    child.sendline(f"cd {scheduler.lean_workspace}")
    child.sendline(f"{scheduler.lake_path} exe repl")
    response = send_command_and_wait(child, scheduler.default_imports, timeout=scheduler.import_timeout, scheduler=scheduler)
    return child, response

def send_command_and_wait(child, command, env=None, timeout=None, scheduler=None):
    json_cmd = json.dumps({"cmd": command}) if env is None else json.dumps({"cmd": command, "env": env})
    child.sendline(json_cmd)
    child.sendline("")

    code = scheduler.default_imports + command
    try:
        child.expect(["\r\n\r\n", "\n\n"], timeout=timeout or scheduler.proof_timeout)
        block = child.before.strip()
        try:
            result = json.loads(block)
            parsed_result = {
                "sorries": result.get("sorries", []),
                "tactics": result.get("tactics", []),
                "errors": [m for m in result.get("messages", []) if m.get("severity") == "error"],
                "warnings": [m for m in result.get("messages", []) if m.get("severity") == "warning"],
                "infos": [m for m in result.get("messages", []) if m.get("severity") == "info"],
                "system_errors": None
            }
            parsed_result["pass"] = not parsed_result["errors"]
            parsed_result["complete"] = (
                parsed_result["pass"] and not parsed_result["sorries"] and not any(
                    "declaration uses 'sorry'" in warning["data"] or "failed" in warning["data"]
                    for warning in parsed_result["warnings"]
                )
            )
        except json.JSONDecodeError as e:
            parsed_result = {"pass": False, "complete": False, "system_errors": f"JSONDECODE ERROR: {e}"}
        response = {"code": command, "compilation_result": parsed_result}
    except pexpect.TIMEOUT as e:
        response = {"code": command, "compilation_result": {"pass": False, "complete": False, "system_errors": f"TIMEOUT ERROR: {e}"}}
    except pexpect.EOF as e:
        response = {"code": command, "compilation_result": {"pass": False, "complete": False, "system_errors": f"EOF ERROR: {e}"}}
    except Exception as e:
        response = {"code": command, "compilation_result": {"pass": False, "complete": False, "system_errors": f"UNEXPECTED ERROR: {e}"}}
    return response

def worker(worker_id, task_queue, result_list, total_restarts, lock, scheduler):
    child, _ = initiate_child(scheduler)
    print(f"Worker {worker_id} started Lean REPL.", flush=True)
    start_time = time.time()
    while True:
        task = task_queue.get()
        if task is None:
            break

        proof_code = task["code"]
        proof_name = task["name"]
        proof_score = task.get("score", None) # added for tree search
        batch_id = task.get("batch_id", None)

        if len(proof_code) == 0:
            response = {"code": proof_code, "compilation_result": {"pass": False, "complete": False, "system_errors": None}}
            response["name"] = proof_name
            response["batch_id"] = batch_id
            response["score"] = proof_score
            response["verify_time"] = round(time.time() - start_time, 2)
            start_time = time.time()
            with lock:
                result_list.append(response)
        else:
            response = send_command_and_wait(child, proof_code, env=0, scheduler=scheduler)
            response["name"] = proof_name
            response["batch_id"] = batch_id
            response["score"] = proof_score
            response["verify_time"] = round(time.time() - start_time, 2)
            start_time = time.time()
            with lock:
                result_list.append(response)

            if response["compilation_result"]["system_errors"] is not None:
                with total_restarts.get_lock():
                    total_restarts.value += 1
                try:
                    child.close()
                except Exception:
                    child.terminate(force=True)
                # if not task_queue.empty():
                #     child, _ = initiate_child(scheduler)
                # else:
                #     print(f"Worker {worker_id}: No more tasks. Not restarting REPL.", flush=True)
                #     break
                child, _ = initiate_child(scheduler)


    try:
        child.close()
    except Exception:
        child.terminate(force=True)
    print(f"Worker {worker_id} terminated Lean REPL.", flush=True)

class DynamicScheduler:
    def __init__(
        self,
        num_workers=4,
        lake_path=f'{os.path.expanduser("~")}/.elan/bin/lake',
        lean_workspace='/scratch/gpfs/st3812/aiformath/Deepseek/mathlib4/',
        default_imports=(
            "import Mathlib\n"
            "import Aesop\n\n"
            "set_option maxHeartbeats 500000\n\n"
            "open BigOperators Real Nat Topology Rat\n\n"
        ),
        import_timeout=300,
        proof_timeout=30,
    ):
        self.num_workers = num_workers
        self.lake_path = lake_path
        self.lean_workspace = lean_workspace
        self.default_imports = default_imports
        self.import_timeout = import_timeout
        self.proof_timeout = proof_timeout

        self.task_queue = mp.Queue()
        manager = mp.Manager()
        self.result_list = manager.list()
        self.lock = manager.Lock()
        self.total_restarts = mp.Value('i', 0)
        self.workers = []

    def start_workers(self):
        for i in range(self.num_workers):
            p = mp.Process(target=worker, args=(i, self.task_queue, self.result_list, self.total_restarts, self.lock, self))
            p.start()
            self.workers.append(p)
        print(f"Started {self.num_workers} worker(s).", flush=True)

    def submit_batch(self, tasks, batch_id):
        for task in tasks:
            task["batch_id"] = batch_id
            self.task_queue.put(task)
        return len(tasks)

    def wait_for_batch(self, batch_id, batch_size, poll_interval=1):
        while True:
            with self.lock:
                count = sum(1 for result in self.result_list if result.get("batch_id") == batch_id)
            if count >= batch_size:
                break
            time.sleep(poll_interval)

        with self.lock:
            batch_results = [r for r in self.result_list if r.get("batch_id") == batch_id]
            remaining_results = [r for r in self.result_list if r.get("batch_id") != batch_id]
            self.result_list[:] = remaining_results

        return batch_results

    def shutdown(self):
        for _ in range(self.num_workers):
            self.task_queue.put(None)
        for p in self.workers:
            p.join()
        print("All workers have been shut down.", flush=True)

if __name__ == '__main__':

    # scheduler = DynamicScheduler(
    #     num_workers=4,
    #     lake_path="/custom/path/to/lake",
    #     lean_workspace="/my/custom/lean_workspace",
    #     import_timeout=500,
    #     proof_timeout=60
    # )

    scheduler = DynamicScheduler(num_workers=10)
    scheduler.start_workers()

    batch_id = "batch_1"
    num_tasks = scheduler.submit_batch(sample_batch, batch_id)
    print(f"Submitted {num_tasks} tasks in {batch_id}.")

    batch_results = scheduler.wait_for_batch(batch_id, num_tasks)
    print(f"Batch {batch_id} completed with results:")
    pprint(batch_results)

    batch_id2 = "batch_2"
    num_tasks2 = scheduler.submit_batch(sample_batch, batch_id2)
    print(f"Submitted {num_tasks2} tasks in {batch_id2}.")
    batch_results2 = scheduler.wait_for_batch(batch_id2, num_tasks2)
    print(f"Batch {batch_id2} completed with results:")
    pprint(batch_results2)

    scheduler.shutdown()


