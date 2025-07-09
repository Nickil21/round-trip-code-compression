import os
import json
import random
import string
import uuid
import argparse
from collections import Counter
from datetime import datetime

# ----------------------------
# DATA GENERATOR
# ----------------------------
class SyntheticDataGenerator:
    def __init__(self, seed=42):
        random.seed(seed)

    def generate_string_data(self, count=100):
        entries = []
        for _ in range(10):
            char = random.choice(string.ascii_uppercase)
            length = random.randint(5, 30)
            entries.append((char * length, "char_repeat"))
        for size in range(2, 7):
            pattern = ''.join(string.ascii_uppercase[:size])
            repeat = random.randint(3, 10)
            entries.append(((pattern * repeat)[:random.randint(size * 3, size * 10)], "alternating_pattern"))
        for size in range(3, 9):
            block = ''.join(random.choices(string.ascii_uppercase, k=size))
            repeat = random.randint(2, 8)
            entries.append((block * repeat, "block_repeat"))
        for _ in range(10):
            base = ''.join(random.choices(string.ascii_uppercase, k=3))
            nested = base + base[::-1] + base
            entries.append((nested * random.randint(2, 5), "nested_repeat"))
        for _ in range(5):
            half = ''.join(random.choices(string.ascii_uppercase, k=5))
            entries.append((half + half[::-1], "palindrome"))
        for _ in range(5):
            s = ''.join(random.choices(string.ascii_uppercase, k=11))
            mid = random.randint(0, len(s) - 1)
            pal = s[:mid] + s[mid] + s[:mid][::-1]
            entries.append((pal, "near_palindrome"))
        pangram = "THE_QUICK_BROWN_FOX_JUMPS_OVER_THE_LAZY_DOG"
        entries.append((pangram, "pangram"))
        entries.append((pangram * 2, "pangram"))
        for _ in range(8):
            entries.append((' '.join(random.sample(pangram.split('_'), 5)), "pangram_mixed"))
        row = "QWERTYUIOP"
        entries.append((row, "keyboard"))
        entries.append((row[::-1], "keyboard"))
        for _ in range(8):
            chunk = row[random.randint(0, len(row) - 5):random.randint(5, len(row))]
            entries.append((chunk * random.randint(2, 6), "keyboard_repeat"))
        for _ in range(10):
            base = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            s = ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(5, 15)))
            for _ in range(3):
                s = s[:random.randint(0, len(s))] + base + s[random.randint(0, len(s)):]
            entries.append((s, "pseudo_random"))
        sentences = [
            "compression is the transformation of data to reduce its size",
            "lossless algorithms preserve every bit of the original data",
            "entropy measures unpredictability in information theory",
            "run length encoding compresses runs of repeated symbols",
            "huffman coding builds optimal prefix trees based on frequencies"
        ]
        for s in sentences:
            entries.append((s, "natural_language"))
            entries.append((s + " " + s, "natural_language_repeat"))
        for _ in range(10):
            s = ''.join(chr(random.randint(32, 126)) for _ in range(100))
            motif = ''.join(random.choices(string.ascii_lowercase, k=5))
            idx = random.randint(0, 95)
            s = s[:idx] + motif + s[idx:]
            entries.append((s, "random_motif"))
        return entries[:count]

    def generate_log_data(self, count=100):
        entries = []
        levels = ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
        endpoints = ["/api/v1/user", "/api/v1/order", "/status", "/metrics", "/auth/login", "/home"]
        methods = ["GET", "POST", "PUT", "DELETE"]
        modules = ["auth", "server", "db", "cache", "worker"]
        for _ in range(count):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            level = random.choice(levels)
            method = random.choice(methods)
            endpoint = random.choice(endpoints)
            duration = round(random.uniform(0.1, 3.0), 3)
            user_id = uuid.uuid4().hex[:8]
            module = random.choice(modules)
            ip = f"192.168.{random.randint(0,255)}.{random.randint(0,255)}"
            msg = f"[{timestamp}] {level} - {method} {endpoint} by user:{user_id} in {duration}s from {ip} ({module})"
            entries.append((msg, "log_entry"))
        return entries[:count]


# ----------------------------
# COMPRESSORS
# ----------------------------
class LZWCompressor:
    @staticmethod
    def compress(text):
        dict_size = 256
        dictionary = {chr(i): i for i in range(dict_size)}
        w = ""
        result = []
        for c in text:
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                dictionary[wc] = dict_size
                dict_size += 1
                w = c
        if w:
            result.append(dictionary[w])
        return result


class AECompressor:
    @staticmethod
    def compress(data: str):
        freq = Counter(data)
        freq['EOF'] = 1
        total = sum(freq.values())
        symbols = sorted(freq.keys())
        cum_counts = {}
        running = 0
        for sym in symbols:
            cum_counts[sym] = running
            running += freq[sym]
        low, high = 0.0, 1.0
        for c in list(data) + ['EOF']:
            width = high - low
            high = low + width * (cum_counts[c] + freq[c]) / total
            low = low + width * cum_counts[c] / total
        return (low + high) / 2, dict(freq)


class RLECompressor:
    @staticmethod
    def compress(data: str):
        if not data:
            return []
        result = []
        prev_char = data[0]
        count = 1
        for c in data[1:]:
            if c == prev_char:
                count += 1
            else:
                result.append((prev_char, count))
                prev_char = c
                count = 1
        result.append((prev_char, count))
        return result



def save_freq_json(freq_dict, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(freq_dict, f, ensure_ascii=True, indent=2)

        
# ----------------------------
# SAVE TO FILES
# ----------------------------
def save_json_and_text(data, algo, compressor, base_out="datasets", json_out="processed_datasets"):
    out_path = os.path.join(base_out, algo)
    os.makedirs(f"{out_path}/input", exist_ok=True)
    os.makedirs(f"{out_path}/output", exist_ok=True)
    json_dir = os.path.join(json_out, algo)
    os.makedirs(json_dir, exist_ok=True)

    ios = []
    for idx, (text, category) in enumerate(data):
        fname = f"{idx:03d}_{category}.txt"
        with open(f"{out_path}/input/{fname}", "w") as f:
            f.write(text)

        if algo == "lzw":
            result = compressor.compress(text)
            with open(f"{out_path}/output/{fname}", "w") as f:
                f.write(str(result))
            ios.append({"input": text, "output": result, "category": category})

        elif algo == "ae":
            code, freq = compressor.compress(text)
            with open(f"{out_path}/output/{fname}", "w") as f:
                f.write(str(code))

            json_path = f"datasets/{algo}/output/{idx:03d}_{category}_freq.json"
            save_freq_json(freq, json_path)
            ios.append({"input": text, "output": code, "category": category})
            # ios.append({"input": {"input_string": text}, "output": {"code": code, "freq": freq}, "category": category})

        elif algo == "rle":
            result = compressor.compress(text)
            with open(f"{out_path}/output/{fname}", "w") as f:
                json.dump(result, f)
            ios.append({"input": text, "output": result, "category": category})



    structured = {
        "problem_description": f"Compress the input string using {algo.upper()} compression.",
        "io_requirements": "Input:\n  `input_string` (str): The string to be compressed.\n\nOutput:\n  `return`: compressed representation",
        "function_name": "main_solution",
        "source": "mixed",
        "algorithm": algo.upper(),
        "meta": {"msgidx": 1},
        "ios": ios
    }

    with open(f"{json_dir}/data.json", "w") as f:
        json.dump(structured, f, indent=4)
    with open(f"{json_dir}/data.jsonl", "w") as f:
        json.dump(structured, f)
        f.write("\n")


# ----------------------------
# MAIN CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic compressed datasets (LZW, AE, RLE)")
    parser.add_argument("--algorithms", nargs="+", choices=["lzw", "ae", "rle"], required=True,
                        help="Compression algorithms to run (e.g., lzw ae rle)")
    parser.add_argument("--source", choices=["string", "log", "mixed"], default="mixed",
                        help="Type of input data")
    parser.add_argument("--count", type=int, default=100, help="Number of examples per source")
    args = parser.parse_args()

    gen = SyntheticDataGenerator()

    if args.source == "string":
        data = gen.generate_string_data(count=args.count)
    elif args.source == "log":
        data = gen.generate_log_data(count=args.count)
    elif args.source == "mixed":
        data = gen.generate_string_data(count=args.count) + gen.generate_log_data(count=args.count)

    if "lzw" in args.algorithms:
        save_json_and_text(data, "lzw", LZWCompressor)
    if "ae" in args.algorithms:
        save_json_and_text(data, "ae", AECompressor)
    if "rle" in args.algorithms:
        save_json_and_text(data, "rle", RLECompressor)


if __name__ == "__main__":
    main()
