import argparse
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from core.utils import count_functions, func_arg_template, write_jsonl
import subprocess
import os
import json
import signal
from time import sleep, time
from multiprocessing import Pool
from tqdm import tqdm
import gc
import shutil
import pickle
import random
random.seed(123)

ANSWER_PREFIX = '<<< Return value from main_solution: '

def _find_answer_line(trace: str) -> bool:
    if not trace:
        return False
    # scan from the bottom to be robust
    for line in reversed(trace.strip().splitlines()):
        if line.strip().startswith(ANSWER_PREFIX):
            return True
    return False

def process_one_item(item):
    # --- normalize & guard fields ---
    try:
        input_val = item.get('input', '')
        input_str = input_val if isinstance(input_val, str) else json.dumps(input_val, ensure_ascii=False)
        qid       = item['id']
        ios_id    = item['ioid']
        code      = item['refcode']
    except KeyError as e:
        return f"ERROR: missing_key:{e}"

    # prep filename and snoop decorator
    main_filename = f"main_{qid}_{ios_id}.py"
    nf = count_functions(code)
    code = code.replace(
        "def main_solution",
        f"\n@snoop(depth={nf})\ndef main_solution"
    )

    # build the call args: repr keeps quoting safe
    args = f"{input_str!r}"
    exec_code = func_arg_template.safe_substitute(refcode=code, args=args)

    os.makedirs(run_path, exist_ok=True)
    file_path = os.path.join(run_path, main_filename)
    with open(file_path, 'w', errors="ignore") as f:
        f.write(exec_code)

    runtime_limit = 5  # seconds
    grace = 0.5        # seconds after TERM before KILL

    try:
        # start in a new process group so we can kill children
        proc = subprocess.Popen(
            [used_python_path, main_filename],
            stdin=subprocess.DEVNULL,   # <-- don’t feed code to stdin
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=run_path,
            text=True,
            start_new_session=True
        )

        try:
            stdout, stderr = proc.communicate(timeout=runtime_limit)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            # give it a moment to exit cleanly
            try:
                stdout, stderr = proc.communicate(timeout=grace)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = proc.communicate()
            return "timeout"
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = proc.communicate(timeout=grace)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = proc.communicate()
            return "execution_error"
        finally:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=0.2)
            except Exception:
                pass

        # snoop logs generally go to stderr; preserve entire stream
        # (stdout can still contain prints; concatenate if you prefer)
        return stderr if stderr else stdout

    except Exception:
        return "subprocess_exception"
    finally:
        # best-effort cleanup of file
        try:
            os.remove(file_path)
        except OSError:
            pass

def process_item(item):
    # keep original fields if available
    item['qid']     = item.get('id')
    item['ios_id']  = item.get('ioid')
    item['io_pred'] = item.get('io_pred', '')

    if isinstance(item['io_pred'], str) and "inversion" in item['io_pred']:
        return {"trace": ""}

    trace = process_one_item(item)

    if trace in {"timeout", "execution_error", "subprocess_exception", "input_type_error"}:
        return {"trace": "ERROR: " + trace}

    if not _find_answer_line(trace):
        return {"trace": "ERROR: missing_answer_prefix", **{k: item.get(k) for k in ("id","ioid")}}

    result = item.copy()
    result["trace"] = trace
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data/processed/")
    parser.add_argument('--algorithm', type=str, required=True, help="Algorithm name to filter data")
    parser.add_argument('--input_file', type=str, default="codeio_1k_msg.jsonl")
    parser.add_argument('--output_file', type=str, default="codeio_1k_msg_executed.jsonl")
    parser.add_argument('--python_path', type=str, default="python")
    parser.add_argument('--run_path', type=str, default="./temp/temp/")
    parser.add_argument('--procs', type=int, default=min(8, os.cpu_count() or 8))  # <-- lower default
    args = parser.parse_args()

    if os.path.exists(args.run_path):
        try:
            shutil.rmtree(args.run_path)
        except Exception as e:
            print(f"Error: {e}")
    os.makedirs(args.run_path, exist_ok=True)

    used_python_path = args.python_path
    run_path = args.run_path

    ofn = os.path.join(args.data_dir, args.algorithm, args.output_file)
    os.makedirs(os.path.dirname(ofn), exist_ok=True)

    dt = []
    src = os.path.join(args.data_dir, args.algorithm, args.input_file)
    with open(src, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                dt.append(json.loads(line))

    adt = []
    goodcount = 0
    totalcount = 0

    # truncate output at start; we will append thereafter
    open(ofn, "w").close()

    with Pool(processes=args.procs, maxtasksperchild=10) as pool:
        for result in tqdm(pool.imap(process_item, dt), total=len(dt)):
            totalcount += 1
            adt.append(result)
            tr = result.get('trace', '')
            if _find_answer_line(tr) and not tr.startswith("ERROR:"):
                goodcount += 1

            if len(adt) >= 1000:
                write_jsonl(adt, ofn, "a")  # append chunk
                adt = []
                print(f"{goodcount}/{totalcount}")

    if adt:
        write_jsonl(adt, ofn, "a")       # <-- APPEND, not "w"
        print(f"Final - {goodcount}/{totalcount}")

    try:
        shutil.rmtree(args.run_path)
    except Exception as e:
        print(f"Error: {e}")
