#!/usr/bin/env python3
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
        # char_repeat
        for _ in range(10):
            char = random.choice(string.ascii_uppercase)
            length = random.randint(5, 30)
            entries.append((char * length, "char_repeat"))
        # alternating_pattern
        for size in range(2, 7):
            pattern = ''.join(string.ascii_uppercase[:size])
            repeat = random.randint(3, 10)
            entries.append(((pattern * repeat)[:random.randint(size * 3, size * 10)],
                            "alternating_pattern"))
        # block_repeat
        for size in range(3, 9):
            block = ''.join(random.choices(string.ascii_uppercase, k=size))
            repeat = random.randint(2, 8)
            entries.append((block * repeat, "block_repeat"))
        # nested_repeat
        for _ in range(10):
            base = ''.join(random.choices(string.ascii_uppercase, k=3))
            nested = base + base[::-1] + base
            entries.append((nested * random.randint(2, 5), "nested_repeat"))
        # palindrome
        for _ in range(5):
            half = ''.join(random.choices(string.ascii_uppercase, k=5))
            entries.append((half + half[::-1], "palindrome"))
        # near_palindrome
        for _ in range(5):
            s = ''.join(random.choices(string.ascii_uppercase, k=11))
            mid = random.randint(0, len(s) - 1)
            pal = s[:mid] + s[mid] + s[:mid][::-1]
            entries.append((pal, "near_palindrome"))
        # pangram
        pangram = "THE_QUICK_BROWN_FOX_JUMPS_OVER_THE_LAZY_DOG"
        entries.append((pangram, "pangram"))
        entries.append((pangram * 2, "pangram"))
        # pangram_mixed
        for _ in range(8):
            entries.append((' '.join(random.sample(pangram.split('_'), 5)),
                            "pangram_mixed"))
        # keyboard
        row = "QWERTYUIOP"
        entries.append((row, "keyboard"))
        entries.append((row[::-1], "keyboard"))
        # keyboard_repeat
        for _ in range(8):
            chunk = row[random.randint(0, len(row) - 5):
                        random.randint(5, len(row))]
            entries.append((chunk * random.randint(2, 6), "keyboard_repeat"))
        # pseudo_random
        for _ in range(10):
            base = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            s = ''.join(random.choices(string.ascii_uppercase + string.digits,
                                       k=random.randint(5, 15)))
            for _ in range(3):
                i = random.randint(0, len(s))
                s = s[:i] + base + s[i:]
            entries.append((s, "pseudo_random"))
        # natural_language
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
        # random_motif
        for _ in range(10):
            s = ''.join(chr(random.randint(32, 126)) for _ in range(100))
            motif = ''.join(random.choices(string.ascii_lowercase, k=5))
            idx = random.randint(0, 95)
            s = s[:idx] + motif + s[idx:]
            entries.append((s, "random_motif"))
        return entries[:count]

    def generate_log_data(self, count=100):
        """
        Returns a list of (log_text, category, metadata) tuples.
        Categories:
          • slow_request   – duration > 2.5s
          • status_check   – endpoint == "/status"
          • db_error       – module=="db" & level=="ERROR"
          • auth_failure   – module=="auth" & level in (WARNING, ERROR, CRITICAL)
        Fallback: lowercase log level.
        """
        entries = []
        levels    = ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
        endpoints = ["/api/v1/user", "/api/v1/order", "/status", "/metrics",
                     "/auth/login", "/home"]
        methods   = ["GET", "POST", "PUT", "DELETE"]
        modules   = ["auth", "server", "db", "cache", "worker"]
        for _ in range(count):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            level     = random.choice(levels)
            method    = random.choice(methods)
            endpoint  = random.choice(endpoints)
            duration  = round(random.uniform(0.1, 3.0), 3)
            user_id   = uuid.uuid4().hex[:8]
            module    = random.choice(modules)
            ip        = f"192.168.{random.randint(0,255)}.{random.randint(0,255)}"
            msg = (
                f"[{timestamp}] {level} - {method} {endpoint} "
                f"by user:{user_id} in {duration}s from {ip} ({module})"
            )
            if duration > 2.5:
                category = "slow_request"
            elif endpoint == "/status":
                category = "status_check"
            elif module == "db" and level == "ERROR":
                category = "db_error"
            elif module == "auth" and level in ("WARNING", "ERROR", "CRITICAL"):
                category = "auth_failure"
            else:
                category = level.lower()
            entries.append((
                msg,
                category,
                {
                    "method":   method,
                    "endpoint": endpoint,
                    "module":   module,
                    "duration": duration
                }
            ))
        return entries[:count]

    def generate_yaml_data(self, count=100):
        """
        Returns a list of (yaml_text, subtype) tuples.
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
database:
  host: db.example.com
  port: 5432
  user: service_user
  password: s3cr3t
features:
  - auth
  - payments
  - notifications
logging:
  level: INFO
  outputs:
    - console
    - file: /var/log/myservice.log
""", "app_config"),
            # Kubernetes Deployment w/ anchors & aliases
            ("""apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 3
  template:
    metadata:
      labels: &labels { app: web }
    spec:
      containers:
        - name: frontend
          image: example/web:stable
          ports:
            - containerPort: 80
        - <<: *labels
          name: backend
          image: example/backend:latest
""", "k8s_deployment"),
            # Docker Compose
            ("""version: "3.8"
services:
  app:
    build: .
    ports:
      - "8080:80"
    volumes:
      - .:/usr/src/app
    environment:
      - DEBUG=1
  redis:
    image: redis:6-alpine
    restart: always
""", "docker_compose"),
            # Helm values + multi-doc
            ("""# Default values for mychart
---
replicaCount: 2
image:
  repository: nginx
  tag: stable
service:
  type: LoadBalancer
  port: 80
---
ingress:
  enabled: true
  hosts:
    - host: chart-example.local
      paths:
        - /
""", "helm_values"),
            # Ansible playbook
            ("""- hosts: webservers
  become: yes
  tasks:
    - name: ensure nginx is at the latest version
      apt:
        name: nginx
        state: latest
    - name: copy config file
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/nginx.conf
""", "ansible_playbook"),
            # Prometheus scrape config
            ("""global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
""", "prometheus_config"),
            # GitHub Actions Workflow
            ("""name: CI
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest
""", "github_actions"),
            # CircleCI config
            ("""version: 2.1
jobs:
  build:
    docker:
      - image: cimg/python:3.8
    steps:
      - checkout
      - run: pip install -r requirements.txt
      - run: pytest
""", "circleci_config"),
            # AWS CloudFormation
            ("""AWSTemplateFormatVersion: '2010-09-09'
Resources:
  MyBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: my-sample-bucket
""", "cloudformation"),
            # Terraform module
            ("""terraform
terraform:
  required_version: ">= 0.14"
  required_providers:
    aws:
      source: hashicorp/aws
      version: "~> 3.0"
resource:
  aws_s3_bucket:
    bucket: my-terraform-bucket
    acl: private
""", "terraform_module"),
        ]
        entries = []
        for _ in range(count):
            yaml_text, subtype = random.choice(samples)
            entries.append((yaml_text, subtype))
        return entries
    
    def generate_tabular_data(self, count=100):
        """
        Returns a list of (table_text, category) tuples, where `table_text`
        is either CSV or TSV, and category describes its pattern.
        
        Categories (10 total):
          • csv_numeric
          • csv_alphanumeric
          • csv_mixed_types
          • csv_repeated_header
          • csv_sparse
          • tsv_numeric
          • tsv_alphanumeric
          • tsv_mixed_types
          • tsv_repeated_header
          • tsv_sparse
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
                # choose 5 column names
                cols = [f"col{i}" for i in range(1,6)]
                
                # Build rows according to category
                rows = []
                for row_idx in range(10):
                    if "numeric" in cat:
                        row = [str(random.randint(0, 1000)) for _ in cols]
                    elif "alphanumeric" in cat:
                        row = [
                            ''.join(random.choices(string.ascii_uppercase+string.digits, k=5))
                            for _ in cols
                        ]
                    elif "mixed_types" in cat:
                        row = []
                        for i in range(len(cols)):
                            if i % 3 == 0:
                                row.append(str(random.randint(0, 500)))
                            elif i % 3 == 1:
                                row.append(random.choice(["TRUE","FALSE","NULL"]))
                            else:
                                row.append(
                                    ''.join(random.choices(string.ascii_lowercase, k=4))
                                )
                    elif "sparse" in cat:
                        # mostly empty, occasional value
                        row = [
                            (str(random.randint(0,100)) if random.random()<0.2 else "")
                            for _ in cols
                        ]
                    else:
                        # fallback to random strings
                        row = [
                            ''.join(random.choices(string.ascii_letters, k=3))
                            for _ in cols
                        ]
                    rows.append(row)
                
                # Possibly duplicate header for *_repeated_header
                all_lines = []
                rep_count = 2 if "repeated_header" in cat else 1
                for _ in range(rep_count):
                    all_lines.append(sep.join(cols))

                for r in rows:
                    all_lines.append(sep.join(r))
                
                table_text = "\n".join(all_lines)
                entries.append((table_text, cat))
        
        # trim to requested count
        return entries[:count]

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

        if category in {
            "app_config", "k8s_deployment", "docker_compose",
            "helm_values", "ansible_playbook", "prometheus_config",
            "github_actions", "circleci_config",
            "cloudformation", "terraform_module"
        }:
            ext = "yaml"
        elif category in {
            "info", "debug", "warning", "error", "critical",
            "slow_request", "status_check", "db_error", "auth_failure"
        }:
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
    args = parser.parse_args()

    gen = SyntheticDataGenerator()
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
