try:
    from snoop.formatting import DefaultFormatter
except ImportError:
    DefaultFormatter = object  # snoop not available (e.g. inside container)
import os
import json
import re
import ast
import glob
from typing import Optional
try:
    from pympler import asizeof
except ImportError:
    import sys as _sys
    class _asizeof_stub:          # shallow fallback when pympler is absent
        @staticmethod
        def asizeof(obj): return _sys.getsizeof(obj)
    asizeof = _asizeof_stub()
from string import Template


# ---------- Offline HF helpers ----------
def hub_root_from_cache_dir(cache_dir: str) -> str:
    """Normalize an HF cache directory to the .../hub root."""
    if not cache_dir:
        return os.path.expanduser("~/.cache/huggingface/hub")
    if cache_dir.endswith("/hub"):
        return cache_dir
    # If models--* dirs exist directly in cache_dir, it is already the hub root
    # (e.g. HF_HOME was set to point at the cache root rather than its parent).
    if glob.glob(os.path.join(cache_dir, "models--*")):
        return cache_dir
    return os.path.join(cache_dir, "hub")

def resolve_local_snapshot_dir(repo_id: str, hub_root: str) -> Optional[str]:
    """
    Map a repo like 'Qwen/Qwen3-32B' to its local model directory.

    Checks two layouts in order:
      1. Standard HF cache:  hub_root/models--Org--Model/snapshots/<hash>/
      2. Flat layout:        hub_root/../ModelName/   (e.g. manually downloaded models)
         A flat directory is accepted only when it contains a config.json.
    """
    # ── Layout 1: standard HF cache ──────────────────────────────────────────
    safe_repo = repo_id.replace("/", "--")
    base = os.path.join(hub_root, f"models--{safe_repo}", "snapshots")
    if os.path.isdir(base):
        try:
            snaps = [os.path.join(base, d) for d in os.listdir(base)]
            snaps = [d for d in snaps if os.path.isdir(d)]
            snaps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            # Only return a snapshot that has a config.json (not just tokenizer files).
            for snap in snaps:
                if os.path.exists(os.path.join(snap, "config.json")):
                    return snap
        except Exception:
            pass

    # ── Layout 2: flat directory next to hub_root ─────────────────────────────
    # e.g. hub_root = /lus/.../hf-cache/models/hub
    #      flat dir = /lus/.../hf-cache/models/Qwen2.5-7B-Instruct
    model_name = repo_id.split("/")[-1]
    flat_path  = os.path.join(os.path.dirname(hub_root), model_name)
    if os.path.isdir(flat_path) and os.path.exists(os.path.join(flat_path, "config.json")):
        return flat_path

    return None

def format_messages(messages):
    """Format a list of chat messages into a single prompt string."""
    return '\n'.join([f"{m['role'].capitalize()}: {m['content']}" for m in messages]) + "\nAssistant:"
# ----------------------------------------


def get_freq_dict(ioid: int, algo: str, base_path="data/raw") -> dict:
    """
    Loads frequency dictionary from a .json file.
    """
    prefix = f"{ioid:03d}_"
    if algo == "ae":
        search_path = os.path.join(base_path, algo, "output", f"{prefix}*_freq.json")
    elif algo == "huffman":
        search_path = os.path.join(base_path, algo, "output", f"{prefix}*_codebook.json")
    matches = glob.glob(search_path)
    if not matches:
        print(f"[WARN] No freq file found for ID {ioid} in {algo}")
        return {}

    file_path = matches[0]
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return {}


def load_jsonl_yield(path):
    with open(path) as f:
        for row, line in enumerate(f):
            try:
                line = json.loads(line)
                yield line
            except Exception:
                pass


def build_messages(prompt, response = None, system_message = None):
    messages = []
    if system_message is not None:
        messages.append({"role":"system","content":system_message})
    messages.append({"role":"user","content":prompt})
    if response is not None:
        messages.append({"role":"assistant","content":response})
    return messages
    

class MyFormatter(DefaultFormatter):
    def __init__(self, prefix, columns, color):
        super().__init__(prefix, columns, color)
        
    def format_start(self, event):
        if event.frame_info.is_ipython_cell:
            return []
        if event.comprehension_type:
            return [u'{type}:'.format(type=event.comprehension_type)]
        else:
            if event.event == 'enter':
                description = 'Enter with block in'
            else:
                assert event.event == 'call'
                if event.frame_info.is_generator:
                    if event.is_yield_value:
                        description = 'Re-enter generator'
                    else:
                        description = 'Start generator'
                else:
                    description = 'Call to'
            return [
                u'{c.cyan}>>> {description} {name}'.format(
                    name=event.code_qualname(),
                    c=self.c,
                    description=description,
                )]

func_arg_template = Template("""from pathlib import Path
import snoop
from snoop.formatting import DefaultFormatter
snoop.tracer.internal_directories += tuple(
    map(str, Path(snoop.tracer.internal_directories[0]).parent.glob("*"))
)
class MyFormatter(DefaultFormatter):
    def __init__(self, prefix, columns, color):
        super().__init__(prefix, columns, color)
        
    def format_start(self, event):
        if event.frame_info.is_ipython_cell:
            return []
        if event.comprehension_type:
            return [u'{type}:'.format(type=event.comprehension_type)]
        else:
            if event.event == 'enter':
                description = 'Enter with block in'
            else:
                assert event.event == 'call'
                if event.frame_info.is_generator:
                    if event.is_yield_value:
                        description = 'Re-enter generator'
                    else:
                        description = 'Start generator'
                else:
                    description = 'Call to'
            return [
                u'{c.cyan}>>> {description} {name}'.format(
                    name=event.code_qualname(),
                    c=self.c,
                    description=description,
                )]
snoop.install(columns='', replace_watch_extras=(), formatter_class=MyFormatter)

$refcode

if __name__ == "__main__":
    main_solution($args)""")

def strict_check_size(obj):
    if asizeof.asizeof(obj) >= 1024: 
        return False

    if isinstance(obj, dict):
        if len(obj) >= 20:  
            return False
        for k, v in obj.items():
            if not strict_check_size(k) or not strict_check_size(v):
                return False

    elif isinstance(obj, (list, tuple, set)):
        if len(obj) >= 20:  
            return False
        for item in obj:
            if not strict_check_size(item):
                return False

    elif isinstance(obj, str):
        if len(obj) >= 100: 
            return False
    else:
        if asizeof.asizeof(obj) >= 128:  
            return False

    return True

def deduplicate_io_pairs(io_pairs):
    seen = set()
    unique = []
    for item in io_pairs:
        # Convert unhashable objects to a hashable string representation
        try:
            key = json.dumps(item[0], sort_keys=True)
        except TypeError:
            key = str(item[0])  # fallback for unserializable objects

        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique

def count_functions(code):
    # Use regex to find all function definitions
    functions = re.findall(r'\bdef\s+\w+\s*\(.*?\)', code)
    return len(functions)

def read_jsonl(jsonl_file_path):
    s = []
    with open(jsonl_file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        linex = line.strip()
        if linex == "":
            continue
        s.append(json.loads(linex))
    return s

def write_jsonl(data, jsonl_file_path, mode="w"):
    # data is a list, each of the item is json-serilizable
    assert isinstance(data, list)
    if len(data) == 0:
        return
    with open(jsonl_file_path, mode) as f:
        for item in data:
            try:
                f.write(json.dumps(item, ensure_ascii=False)+"\n")
            except Exception as e:
                print(item)