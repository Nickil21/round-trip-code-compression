#!/usr/bin/env python3
import os
import json
import random
import string
import argparse
from collections import Counter
from datetime import datetime, timedelta

# ----------------------------
# DATA GENERATOR
# ----------------------------
class SyntheticDataGenerator:
    def __init__(self, seed=42):
        # 1) seed the RNG for full reproducibility
        random.seed(seed)

        # 2) create a fixed list of 10 timestamps, 1 minute apart
        base = datetime(2025, 1, 1, 0, 0, 0)
        self.timestamps = [
            (base + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(10)
        ]

    def generate_string_data(self, count=100):
        entries = []
        # 14 categories (≥10)
        for _ in range(count // 14):
            char = random.choice(string.ascii_uppercase)
            entries.append((char * random.randint(5, 30), "char_repeat"))

        for size in range(2, 7):
            pattern = ''.join(string.ascii_uppercase[:size])
            repeat = random.randint(3, 10)
            entries.append(((pattern * repeat)[:random.randint(size * 3, size * 10)],
                            "alternating_pattern"))

        for size in range(3, 9):
            block = ''.join(random.choices(string.ascii_uppercase, k=size))
            entries.append((block * random.randint(2, 8), "block_repeat"))

        for _ in range(count // 14):
            base3 = ''.join(random.choices(string.ascii_uppercase, k=3))
            nested = base3 + base3[::-1] + base3
            entries.append((nested * random.randint(2, 5), "nested_repeat"))

        for _ in range(count // 14):
            half = ''.join(random.choices(string.ascii_uppercase, k=5))
            entries.append((half + half[::-1], "palindrome"))

        for _ in range(count // 14):
            s = ''.join(random.choices(string.ascii_uppercase, k=11))
            mid = random.randint(0, len(s)-1)
            entries.append((s[:mid] + s[mid] + s[:mid][::-1], "near_palindrome"))

        pangram = "THE_QUICK_BROWN_FOX_JUMPS_OVER_THE_LAZY_DOG"
        entries.append((pangram, "pangram"))
        entries.append((pangram * 2, "pangram"))

        for _ in range(count // 14):
            entries.append((' '.join(random.sample(pangram.split('_'), 5)),
                            "pangram_mixed"))

        row = "QWERTYUIOP"
        entries.append((row, "keyboard"))
        entries.append((row[::-1], "keyboard"))

        for _ in range(count // 14):
            start = random.randint(0, len(row)-5)
            end = random.randint(5, len(row))
            chunk = row[start:end]
            entries.append((chunk * random.randint(2, 6), "keyboard_repeat"))

        for _ in range(count // 14):
            base4 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            s = ''.join(random.choices(string.ascii_uppercase + string.digits,
                                       k=random.randint(5, 15)))
            for _ in range(3):
                i = random.randint(0, len(s))
                s = s[:i] + base4 + s[i:]
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

        for _ in range(count // 14):
            s = ''.join(chr(random.randint(32, 126)) for _ in range(100))
            motif = ''.join(random.choices(string.ascii_lowercase, k=5))
            idx = random.randint(0, 95)
            entries.append((s[:idx] + motif + s[idx:], "random_motif"))

        # prefix every category with "string_"
        return [(text, f"string_{cat}") for text, cat in entries][:count]

    def generate_log_data(self, count=100):
        """
        Categories:
          slow_request, status_check, metrics_request, auth_login,
          db_error, auth_failure,
          info, debug, warning, error, critical
        """
        entries = []
        levels    = ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
        endpoints = ["/api/v1/user", "/api/v1/order", "/status",
                     "/metrics", "/auth/login", "/home"]
        modules   = ["auth", "server", "db", "cache", "worker"]
        methods   = ["GET", "POST", "PUT", "DELETE"]

        for _ in range(count):
            ts       = random.choice(self.timestamps)
            level    = random.choice(levels)
            endpoint = random.choice(endpoints)
            duration = round(random.uniform(0.1, 3.0), 3)
            module   = random.choice(modules)
            method   = random.choice(methods)
            ip       = f"192.168.{random.randint(0,255)}.{random.randint(0,255)}"

            # deterministic “UUID” via our seeded RNG:
            hex_id = f"{random.getrandbits(32):08x}"

            msg = (f"[{ts}] {level} - {method} {endpoint} "
                   f"by user:{hex_id} in {duration}s from {ip} ({module})")

            if duration > 2.5:
                category = "slow_request"
            elif endpoint == "/status":
                category = "status_check"
            elif endpoint == "/metrics":
                category = "metrics_request"
            elif endpoint == "/auth/login":
                category = "auth_login"
            elif module == "db" and level == "ERROR":
                category = "db_error"
            elif module == "auth" and level in ("WARNING", "ERROR", "CRITICAL"):
                category = "auth_failure"
            else:
                category = level.lower()

            entries.append((msg, category, {
                "method":   method,
                "endpoint": endpoint,
                "module":   module,
                "duration": duration
            }))

        # prefix every category with "log_"
        return [(text, f"log_{cat}", meta) for text, cat, meta in entries][:count]

    def generate_yaml_data(self, count=100):
        """
        Subtypes include:
          app_config, k8s_deployment, docker_compose,
          helm_values, ansible_playbook, prometheus_config,
          github_actions, circleci_config,
          cloudformation, terraform_module
        """
        samples = [
            # Application config
            ("""app:
  name: MyService
  version: 1.2.3
  debug: false
""", "app_config"),

            # Kubernetes Deployment
            ("""apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
        - name: frontend
          image: example/web:stable
          ports:
            - containerPort: 80
""", "k8s_deployment"),

            # Docker Compose
            ("""version: "3.8"
services:
  app:
    build: .
    ports:
      - "8080:80"
""", "docker_compose"),

            # Helm values (multi-doc)
            ("""# Default values for mychart
---
replicaCount: 2
image:
  repository: nginx
  tag: stable
""", "helm_values"),

            # Ansible playbook
            ("""- hosts: all
  tasks:
    - name: ensure git installed
      apt:
        name: git
        state: present
""", "ansible_playbook"),

            # Prometheus config
            ("""global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
""", "prometheus_config"),

            # GitHub Actions workflow
            ("""name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
""", "github_actions"),

            # CircleCI config
            ("""version: 2.1
jobs:
  build:
    docker:
      - image: cimg/python:3.9
    steps:
      - checkout
      - run: pytest
""", "circleci_config"),

            # AWS CloudFormation
            ("""AWSTemplateFormatVersion: '2010-09-09'
Resources:
  MyBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: my-unique-bucket
""", "cloudformation"),

            # Terraform module
            ("""terraform {
  required_version = ">= 0.14"
}
resource "aws_s3_bucket" "b" {
  bucket = "my-tf-bucket"
  acl    = "private"
}
""", "terraform_module"),
        ]

        entries = [random.choice(samples) for _ in range(count)]
        # prefix every category with "yaml_"
        return [(text, f"yaml_{cat}") for text, cat in entries]

    def generate_tabular_data(self, count=100):
        """
        10 categories: csv_numeric, csv_alphanumeric, csv_mixed_types,
                       csv_repeated_header, csv_sparse,
                       tsv_numeric, tsv_alphanumeric, tsv_mixed_types,
                       tsv_repeated_header, tsv_sparse
        """
        categories = [
            "csv_numeric", "csv_alphanumeric", "csv_mixed_types",
            "csv_repeated_header", "csv_sparse",
            "tsv_numeric", "tsv_alphanumeric", "tsv_mixed_types",
            "tsv_repeated_header", "tsv_sparse"
        ]
        entries = []
        per_cat = max(1, count // len(categories))

        for cat in categories:
            is_csv = cat.startswith("csv")
            sep    = "," if is_csv else "\t"
            for _ in range(per_cat):
                cols = [f"col{i}" for i in range(1,6)]
                rows = []
                for __ in range(5):
                    if "numeric" in cat:
                        rows.append([str(random.randint(0,1000)) for _ in cols])
                    elif "alphanumeric" in cat:
                        rows.append([
                            ''.join(random.choices(string.ascii_uppercase+string.digits, k=5))
                            for _ in cols
                        ])
                    elif "mixed_types" in cat:
                        rows.append([
                            random.choice([
                                str(random.randint(0,500)),
                                random.choice(["TRUE","FALSE","NULL"]),
                                ''.join(random.choices(string.ascii_lowercase, k=4))
                            ])
                            for _ in cols
                        ])
                    elif "sparse" in cat:
                        rows.append([
                            str(random.randint(0,100)) if random.random()<0.2 else ""
                            for _ in cols
                        ])
                    else:
                        rows.append([
                            ''.join(random.choices(string.ascii_letters, k=3))
                            for _ in cols
                        ])

                # build lines, with repeated header if needed
                lines = []
                header_count = 2 if "repeated_header" in cat else 1
                for __ in range(header_count):
                    lines.append(sep.join(cols))
                for r in rows:
                    lines.append(sep.join(r))

                entries.append(("\n".join(lines), cat))

        # prefix every category with "tabular_"
        return [(text, f"tabular_{cat}") for text, cat in entries][:count]


# ----------------------------
# COMPRESSORS
# ----------------------------
class LZWCompressor:
    @staticmethod
    def compress(text: str):
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

class HuffmanCompressor:
    @staticmethod
    def compress(data: str):
        from collections import namedtuple
        import heapq
        freq = Counter(data)
        Node = namedtuple("Node", ["freq", "symbol", "left", "right"])
        Node.__lt__ = lambda a, b: a.freq < b.freq
        heap = [Node(f, sym, None, None) for sym, f in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = Node(left.freq + right.freq, None, left, right)
            heapq.heappush(heap, merged)
        root = heap[0]
        codebook = {}
        def _walk(node, prefix):
            if node.symbol is not None:
                codebook[node.symbol] = prefix or "0"
            else:
                _walk(node.left,  prefix + "0")
                _walk(node.right, prefix + "1")
        _walk(root, "")
        bitstr = "".join(codebook[c] for c in data)
        padding = (-len(bitstr)) % 8
        bitstr += "0" * padding
        encoded_bytes = bytearray()
        for i in range(0, len(bitstr), 8):
            encoded_bytes.append(int(bitstr[i:i+8], 2))
        return list(encoded_bytes), codebook, padding

def save_freq_json(freq_dict, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(freq_dict, f, ensure_ascii=True, indent=2)

def validate_data_point(idx, item):
    if len(item) == 2:
        text, category = item
        metadata = {}
    elif len(item) == 3:
        text, category, metadata = item
    else:
        raise ValueError(f"[{idx}] Bad shape: {item!r}")
    if not category or not isinstance(category, str):
        raise ValueError(f"[{idx}] Bad category: {category!r}")
    return text, category, metadata

# ----------------------------
# SAVE TO FILES
# ----------------------------
def save_json_and_text(data, algo, compressor,
                       base_out="datasets", json_out="processed_datasets"):
    out_path = os.path.join(base_out, algo)
    os.makedirs(f"{out_path}/input", exist_ok=True)
    os.makedirs(f"{out_path}/output", exist_ok=True)
    json_dir = os.path.join(json_out, algo)
    os.makedirs(json_dir, exist_ok=True)

    ios = []
    for idx, item in enumerate(data):
        text, category, metadata = validate_data_point(idx, item)

        if category.startswith("yaml_"):
            ext = "yaml"
        elif category.startswith("log_"):
            ext = "log"
        else:
            ext = "txt"

        fname = f"{idx:03d}_{category}.{ext}"
        with open(f"{out_path}/input/{fname}", "w", encoding="utf-8") as f:
            f.write(text)

        if algo == "lzw":
            result = compressor.compress(text)
            with open(f"{out_path}/output/{fname}", "w", encoding="utf-8") as f:
                f.write(str(result))
            ios.append({
                "input": text,
                "output": result,
                "category": category,
                **metadata
            })

        elif algo == "ae":
            code, freq = compressor.compress(text)
            with open(f"{out_path}/output/{fname}", "w", encoding="utf-8") as f:
                f.write(str(code))
            json_path = f"{base_out}/{algo}/output/{idx:03d}_{category}_freq.json"
            save_freq_json(freq, json_path)
            ios.append({
                "input": text,
                "output": code,
                "category": category,
                **metadata
            })

        elif algo == "rle":
            result = compressor.compress(text)
            with open(f"{out_path}/output/{fname}", "w", encoding="utf-8") as f:
                json.dump(result, f)
            ios.append({
                "input": text,
                "output": result,
                "category": category,
                **metadata
            })

        elif algo == "huffman":
            encoded, codebook, padding = compressor.compress(text)
            with open(f"{out_path}/output/{fname}", "w", encoding="utf-8") as f:
                f.write(str(encoded))
            json_path = f"{base_out}/{algo}/output/{idx:03d}_{category}_codebook.json"
            save_freq_json({"codebook": codebook, "padding": padding}, json_path)
            ios.append({
                "input": text,
                "output": encoded,
                "category": category,
                **metadata
            })

    structured = {
        "problem_description": f"Compress the input string using {algo.upper()} compression.",
        "io_requirements": (
            "Input:\n"
            "  `input_string` (str): The string to be compressed.\n\n"
            "Output:\n"
            "  `return`: compressed representation"
        ),
        "function_name": "main_solution",
        "source": "mixed",
        "algorithm": algo.upper(),
        "meta": {"msgidx": 1},
        "ios": ios
    }

    with open(f"{json_dir}/data.json", "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=4)
    with open(f"{json_dir}/data.jsonl", "w", encoding="utf-8") as f:
        json.dump(structured, f)
        f.write("\n")

# ----------------------------
# MAIN CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic compressed datasets (LZW, AE, RLE, Huffman)"
    )
    parser.add_argument(
        "--algorithms", nargs="+",
        choices=["lzw", "ae", "rle", "huffman"],
        required=True,
        help="Compression algorithms to run"
    )
    parser.add_argument(
        "--source",
        choices=["string", "log", "yaml", "mixed"],
        default="mixed",
        help="Type of input data"
    )
    parser.add_argument(
        "--count", type=int, default=100,
        help="Number of examples per source"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    gen = SyntheticDataGenerator(seed=args.seed)
    if args.source == "string":
        data = gen.generate_string_data(count=args.count)
    elif args.source == "log":
        data = gen.generate_log_data(count=args.count)
    elif args.source == "yaml":
        data = gen.generate_yaml_data(count=args.count)
    elif args.source == "csv":
        data = gen.generate_tabular_data(count=args.count)
    else:
        data = (
            gen.generate_string_data(count=args.count) +
            gen.generate_log_data(count=args.count) +
            gen.generate_yaml_data(count=args.count) +
            gen.generate_tabular_data(count=args.count)
        )

    if "lzw" in args.algorithms:
        save_json_and_text(data, "lzw", LZWCompressor)
    if "ae" in args.algorithms:
        save_json_and_text(data, "ae", AECompressor)
    if "rle" in args.algorithms:
        save_json_and_text(data, "rle", RLECompressor)
    if "huffman" in args.algorithms:
        save_json_and_text(data, "huffman", HuffmanCompressor)

if __name__ == "__main__":
    main()
