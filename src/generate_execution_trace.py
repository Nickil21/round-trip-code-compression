import argparse
from utils import *
import subprocess
import os
import json
import signal
from time import sleep

from multiprocessing import Pool
from tqdm import tqdm
import gc
import shutil
import pickle
import random
random.seed(123)

def process_one_item(item):
    # avoid shadowing built-in
    input_str = item['input']  
    qid       = item['id']
    ios_id    = item['ioid']
    code      = item['refcode']

    # prep filename and snoop decorator
    main_filename = f"main_{qid}_{ios_id}.py"
    nf = count_functions(code)
    code = code.replace(
        "def main_solution",
        f"\n@snoop(depth={nf})\ndef main_solution"
    )

    # build the call args: param='the string…'
    args = f"{input_str!r}"

    print(f"Args for {main_filename}: {args}")

    # substitute into your runner template
    exec_code = func_arg_template.safe_substitute(
        refcode=code,
        args=args
    )

    with open(run_path + main_filename, 'w', errors="ignore") as f:
        f.write(exec_code)
    
    runtime_limit = 5  # seconds

    try:
        # Start the subprocess in a new session (process group)
        proc = subprocess.Popen(
            [used_python_path, main_filename],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=run_path,
            text=True,
            start_new_session=True
        )

        try:
            # Communicate with the subprocess
            stdout, stderr = proc.communicate(input=exec_code, timeout=runtime_limit)
        except subprocess.TimeoutExpired:
            # Timeout expired; kill the process group
            os.killpg(proc.pid, signal.SIGTERM)  # Send SIGTERM to the process group
            stdout, stderr = proc.communicate()
            return "timeout"
        except Exception as e:
            # Other exception occurred; kill the process group
            os.killpg(proc.pid, signal.SIGTERM)
            stdout, stderr = proc.communicate()
            return "execution_error"
        finally:
            # Ensure the subprocess is terminated
            proc.kill()
            proc.wait()
 
        return stderr

    except Exception as e:
        # Handle any exceptions that might occur while setting up the subprocess
        return "subprocess_exception"

def process_item(item):
    item['qid']    = item['id']
    item['ios_id'] = item['ioid']
    item['io_pred'] = item['io_pred']
    
    if "inversion" in item['io_pred']:
        result = {"trace": ""}
    else:
        result = dict()
        trace = ""
        trace = process_one_item(item)
        if trace in ["timeout", "execution_error", "subprocess_exception", "input_type_error"]:
            result = {
                "trace": "ERROR: " + trace,
            }
        elif "<<< Return value from main_solution:" not in trace:
            result = {
                "trace": "ERROR: " + trace,
            }
        else:
            result = item.copy()
            result.update({'trace': trace})
    gc.collect()
    try:
        os.remove(run_path + f"main_{item['qid']}_{item['ios_id']}.py")
    except OSError:
        pass
    sleep(0.05)
        
    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="processed_datasets/")
    parser.add_argument('--algorithm', type=str, help="Algorithm name to filter data")
    parser.add_argument('--input_file', type=str, default="codeio_1k_msg.jsonl")
    parser.add_argument('--output_file', type=str, default="codeio_1k_msg_executed.jsonl")
    parser.add_argument('--python_path', type=str, default="python")
    parser.add_argument('--run_path', type=str, default="./temp/temp/")
    args = parser.parse_args()

    if os.path.exists(args.run_path):
        try:
            shutil.rmtree(args.run_path)
        except Exception as e:
            print(f"Error: {e}")
    os.makedirs(args.run_path, exist_ok=True)
    
    used_python_path = args.python_path
    run_path = args.run_path

    ofn = args.data_dir + args.algorithm + os.sep + args.output_file

    dt = []
    with open(args.data_dir + args.algorithm + os.sep + args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dt.append(json.loads(line))

    adt = []
    goodcount=0
    totalcount=0
    with Pool(processes=64, maxtasksperchild=10) as pool:
        for result in tqdm(pool.imap(process_item, dt), total=len(dt)):
            totalcount += 1
            adt.append(result)
            trace_lines = result['trace'].strip().split('\n')
            if trace_lines[-1].strip().startswith('<<< Return value from main_solution:') and not result['trace'].startswith("ERROR:"):
                goodcount += 1
            if len(adt) >= 1000:
                write_jsonl(adt, ofn, "a")
                adt = []
                print(f"{goodcount}/{totalcount}")

    if len(adt) > 0:
        write_jsonl(adt, ofn, "w")
        print(f"Final - {goodcount}/{totalcount}")

    try:
        shutil.rmtree(args.run_path)
    except Exception as e:
        print(f"Error: {e}")
        