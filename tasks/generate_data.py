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

    # --- DNA helpers ---
    @staticmethod
    def _dna_rc(seq: str) -> str:
        comp = str.maketrans("ACGTNRYWSKMBDHV", "TGCANYRWSMKVHDB")
        return seq.translate(comp)[::-1]

    @staticmethod
    def _dna_random(length: int, alphabet="ACGT") -> str:
        return ''.join(random.choices(alphabet, k=length))

    @staticmethod
    def _dna_snp(seq: str, num_snps: int = 3) -> str:
        seq = list(seq)
        if not seq:
            return ""
        k = min(num_snps, len(seq))
        idxs = random.sample(range(len(seq)), k=k)
        for i in idxs:
            base = seq[i]
            choices = [b for b in "ACGT" if b != base]
            seq[i] = random.choice(choices)
        return ''.join(seq)

    def generate_dna_data(self, count=100):
        """
        DNA categories:
          dna_random, dna_motif_repeat, dna_palindrome_rc, dna_orf_like,
          dna_microsatellite, dna_snp_variants, dna_gc_clamp,
          dna_fasta_record, dna_codon_repeat, dna_kmer_concat
        """
        entries, seen = [], set()

        def add(text, cat):
            if text in seen:
                return False
            seen.add(text)
            entries.append((text, f"string_{cat}"))  # keep "string_" prefix family
            return True

        # 1) Random genomic-like strings
        target = max(1, count // 10)
        while len([c for _, c in entries if c == "string_dna_random"]) < target and len(entries) < count:
            add(self._dna_random(random.randint(60, 200)), "dna_random")

        # 2) Motif repeat (e.g., (CA)n, (ATG)n)
        motifs = ["CA", "AT", "TG", "GATA", "ATG"]
        while len([c for _, c in entries if c == "string_dna_motif_repeat"]) < target and len(entries) < count:
            m = random.choice(motifs)
            n = random.randint(8, max(8, 40 // max(1, len(m))))
            add(m * n, "dna_motif_repeat")

        # 3) Reverse-complement palindromes (restriction-site-like)
        seeds = ["GAATTC", "AAGCTT", "GGATCC", "CCCGGG", "GGTACC"]  # EcoRI, HindIII, BamHI, SmaI, KpnI
        while len([c for _, c in entries if c == "string_dna_palindrome_rc"]) < target and len(entries) < count:
            left = random.choice(seeds)
            pal = left + self._dna_rc(left)
            add(pal, "dna_palindrome_rc")

        # 4) ORF-like (start ATG ... stop TAG/TGA/TAA)
        stops = ["TAG", "TGA", "TAA"]
        while len([c for _, c in entries if c == "string_dna_orf_like"]) < target and len(entries) < count:
            codon_count = random.randint(20, 60)
            middle = ''.join(random.choice([
                "AAA","GCT","GCC","GCA","GCG","GTT","GTC","GTA","GTG",
                "TCT","TCC","TCA","TCG","GAA","GAG","CCT","CCA","CCG","CGA","CGG"
            ]) for _ in range(codon_count))
            seq = "ATG" + middle + random.choice(stops)
            if len(seq) % 3 == 0:
                add(seq, "dna_orf_like")

        # 5) Microsatellites with flanks
        micro_motifs = ["CA", "GA", "TTC", "AAT", "GAA"]
        while len([c for _, c in entries if c == "string_dna_microsatellite"]) < target and len(entries) < count:
            flankL = self._dna_random(20)
            core = random.choice(micro_motifs) * random.randint(6, 20)
            flankR = self._dna_random(20)
            add(flankL + core + flankR, "dna_microsatellite")

        # 6) SNP variant sets (baseline + a few mutated versions)
        while len([c for _, c in entries if c == "string_dna_snp_variants"]) < target and len(entries) < count:
            base = self._dna_random(120)
            variants = [self._dna_snp(base, n) for n in (1, 2, 3)]
            for v in [base] + variants:
                if len(entries) >= count:
                    break
                add(v, "dna_snp_variants")

        # 7) GC clamp (high-GC ends)
        while len([c for _, c in entries if c == "string_dna_gc_clamp"]) < target and len(entries) < count:
            core = self._dna_random(60)
            clamp_L = "G" * random.randint(6, 14) + self._dna_random(10, "GC")
            clamp_R = self._dna_random(10, "GC") + "C" * random.randint(6, 14)
            add(clamp_L + core + clamp_R, "dna_gc_clamp")

        # 8) FASTA-like record (single record, header + sequence lines)
        idx = 0
        while len([c for _, c in entries if c == "string_dna_fasta_record"]) < target and len(entries) < count:
            seq = self._dna_random(random.randint(120, 300))
            wrapped = "\n".join([seq[i:i+60] for i in range(0, len(seq), 60)])
            rec = f">seq_{idx}\n{wrapped}"
            idx += 1
            add(rec, "dna_fasta_record")

        # 9) Codon repeats (e.g., ATG repeated)
        codons = ["ATG", "GAA", "GCT", "TTT", "TGG", "CGA"]
        while len([c for _, c in entries if c == "string_dna_codon_repeat"]) < target and len(entries) < count:
            cod = random.choice(codons)
            add(cod * random.randint(20, 120), "dna_codon_repeat")

        # 10) k-mer concatenations
        while len([c for _, c in entries if c == "string_dna_kmer_concat"]) < target and len(entries) < count:
            k = random.choice([3, 4, 5, 6])
            kmers = {self._dna_random(k) for _ in range(random.randint(10, 50))}
            add(''.join(sorted(kmers)), "dna_kmer_concat")

        # Top up uniquely if needed
        while len(entries) < count:
            add(self._dna_random(150), "dna_random")

        return entries

    def generate_string_data(self, count=100):
        entries = []
        seen = set()

        def add(text, cat):
            if text not in seen:
                seen.add(text)
                entries.append((text, f"string_{cat}"))
                return True
            return False

        # 14+ categories (uniqueness-guarded)
        produced = 0
        # 1) repeated chars
        while produced < count // 14 and produced < count:
            char = random.choice(string.ascii_uppercase)
            text = char * random.randint(5, 30)
            if add(text, "char_repeat"):
                produced += 1

        # 2) alternating_pattern
        for size in range(2, 7):
            if len(entries) >= count:
                break
            pattern = ''.join(string.ascii_uppercase[:size])
            repeat = random.randint(3, 10)
            text = (pattern * repeat)[:random.randint(size * 3, size * 10)]
            add(text, "alternating_pattern")

        # 3) block_repeat
        for size in range(3, 9):
            if len(entries) >= count:
                break
            block = ''.join(random.choices(string.ascii_uppercase, k=size))
            add(block * random.randint(2, 8), "block_repeat")

        # 4) nested_repeat
        produced = 0
        while produced < count // 14 and len(entries) < count:
            base3 = ''.join(random.choices(string.ascii_uppercase, k=3))
            nested = base3 + base3[::-1] + base3
            if add(nested * random.randint(2, 5), "nested_repeat"):
                produced += 1

        # 5) palindrome
        produced = 0
        while produced < count // 14 and len(entries) < count:
            half = ''.join(random.choices(string.ascii_uppercase, k=5))
            if add(half + half[::-1], "palindrome"):
                produced += 1

        # 6) near_palindrome
        produced = 0
        while produced < count // 14 and len(entries) < count:
            s = ''.join(random.choices(string.ascii_uppercase, k=11))
            mid = random.randint(0, len(s)-1)
            if add(s[:mid] + s[mid] + s[:mid][::-1], "near_palindrome"):
                produced += 1

        # 7) pangram (only once; no duplicate)
        pangram = "THE_QUICK_BROWN_FOX_JUMPS_OVER_THE_LAZY_DOG"
        add(pangram, "pangram")

        # 8) pangram_mixed
        produced = 0
        tokens = pangram.split('_')
        while produced < count // 14 and len(entries) < count:
            text = ' '.join(random.sample(tokens, 5))
            if add(text, "pangram_mixed"):
                produced += 1

        # 9) keyboard + reversed (distinct)
        row = "QWERTYUIOP"
        add(row, "keyboard")
        add(row[::-1], "keyboard")

        # 10) keyboard_repeat
        produced = 0
        while produced < count // 14 and len(entries) < count:
            start = random.randint(0, len(row)-5)
            end = random.randint(start+5, len(row))
            chunk = row[start:end]
            if add(chunk * random.randint(2, 6), "keyboard_repeat"):
                produced += 1

        # 11) pseudo_random
        produced = 0
        while produced < count // 14 and len(entries) < count:
            base4 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            s = ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(5, 15)))
            for _ in range(3):
                i = random.randint(0, len(s))
                s = s[:i] + base4 + s[i:]
            if add(s, "pseudo_random"):
                produced += 1

        # 12) natural language + repeats (kept distinct)
        sentences = [
            "compression is the transformation of data to reduce its size",
            "lossless algorithms preserve every bit of the original data",
            "entropy measures unpredictability in information theory",
            "run length encoding compresses runs of repeated symbols",
            "huffman coding builds optimal prefix trees based on frequencies"
        ]
        for s in sentences:
            if len(entries) >= count:
                break
            add(s, "natural_language")
            if len(entries) >= count:
                break
            add(s + " " + s, "natural_language_repeat")

        # 13) random with motif
        produced = 0
        while produced < count // 14 and len(entries) < count:
            s = ''.join(chr(random.randint(32, 126)) for _ in range(100))
            motif = ''.join(random.choices(string.ascii_lowercase, k=5))
            idx = random.randint(0, 95)
            if add(s[:idx] + motif + s[idx:], "random_motif"):
                produced += 1

        # Top up to exact 'count' with unique alphanum strings if needed
        while len(entries) < count:
            text = ''.join(random.choices(string.ascii_letters + string.digits, k=24))
            add(text, "filler_unique")

        return entries

    def generate_log_data(self, count=100):
        """
        Categories:
          slow_request, status_check, metrics_request, auth_login,
          db_error, auth_failure,
          info, debug, warning, error, critical
        """
        entries = []
        seen = set()
        levels    = ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
        endpoints = ["/api/v1/user", "/api/v1/order", "/status",
                     "/metrics", "/auth/login", "/home"]
        modules   = ["auth", "server", "db", "cache", "worker"]
        methods   = ["GET", "POST", "PUT", "DELETE"]

        def make_one():
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

            meta = {
                "method":   method,
                "endpoint": endpoint,
                "module":   module,
                "duration": duration
            }
            return msg, category, meta

        while len(entries) < count:
            msg, category, meta = make_one()
            if msg in seen:
                continue
            seen.add(msg)
            entries.append((msg, f"log_{category}", meta))

        return entries

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

        def mutate(i, text, cat):
            # produce a minimally changed but valid variant based on index
            if cat == "app_config":
                return text.replace("MyService", f"MyService{i}").replace("1.2.3", f"1.2.{(i%9)+1}")
            if cat == "k8s_deployment":
                return text.replace("web-app", f"web-app-{i}").replace("replicas: 3", f"replicas: {(i%5)+1}")
            if cat == "docker_compose":
                return text.replace('"8080:80"', f'"{8080 + i}:80"')
            if cat == "helm_values":
                return text.replace("replicaCount: 2", f"replicaCount: {(i%4)+1}").replace("stable", f"stable-{i%3}")
            if cat == "ansible_playbook":
                return text.replace("git", f"git{i%3}").replace("present", "latest" if i%2 else "present")
            if cat == "prometheus_config":
                return text.replace("localhost:9100", f"localhost:{9100 + (i%10)}")
            if cat == "github_actions":
                return text.replace("@v2", f"@v{2 + (i%3)}")
            if cat == "circleci_config":
                return text.replace("3.9", f"3.{9 - (i%3)}")
            if cat == "cloudformation":
                return text.replace("my-unique-bucket", f"my-unique-bucket-{i}")
            if cat == "terraform_module":
                return text.replace("my-tf-bucket", f"my-tf-bucket-{i}")
            return text

        seen = set()
        entries = []
        i = 0
        while len(entries) < count:
            base_text, cat = samples[i % len(samples)]
            variant = mutate(len(entries), base_text, cat)
            if variant in seen:
                i += 1
                continue
            seen.add(variant)
            entries.append((variant, f"yaml_{cat}"))
            i += 1
        return entries

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

        def one_table(cat):
            is_csv = cat.startswith("csv")
            sep    = "," if is_csv else "\t"
            cols = [f"col{i}" for i in range(1,6)]
            rows = []
            for _ in range(5):
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

            return "\n".join(lines)

        per_cat = max(1, count // len(categories))
        entries = []
        seen = set()

        for cat in categories:
            while sum(1 for _, c in entries if c == f"tabular_{cat}") < per_cat and len(entries) < count:
                t = one_table(cat)
                if t in seen:  # regenerate until unique
                    continue
                seen.add(t)
                entries.append((t, f"tabular_{cat}"))

        # Top up to exact count (unique) if rounding left us short
        i = 0
        while len(entries) < count:
            cat = categories[i % len(categories)]
            t = one_table(cat)
            if t not in seen:
                seen.add(t)
                entries.append((t, f"tabular_{cat}"))
            i += 1

        return entries


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
        if not data:
            return [], {}, 0
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

def enforce_unique_texts(data):
    """Ensure uniqueness across combined sources, keeping first occurrence."""
    seen = set()
    unique = []
    for item in data:
        text = item[0]
        if text in seen:
            continue
        seen.add(text)
        unique.append(item)
    return unique

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
        choices=["string", "log", "yaml", "csv", "dna", "mixed"],
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
    elif args.source == "dna":
        data = gen.generate_dna_data(count=args.count)
    else:
        data = (
            gen.generate_string_data(count=args.count) +
            gen.generate_log_data(count=args.count) +
            gen.generate_yaml_data(count=args.count) +
            gen.generate_tabular_data(count=args.count) +
            gen.generate_dna_data(count=args.count)
        )

    # final guard: ensure distinct inputs across the combined set
    data = enforce_unique_texts(data)

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
