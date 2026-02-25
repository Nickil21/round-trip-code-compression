"""
Build alternative-format prompt files for the tokenization ablation study.

Generates data/processed/{algo}/codeio_alt_{variant}_msg.jsonl where each
sample uses a tokenizer-neutral output representation instead of the original
numeric list format. Only output_execution_prediction tasks are generated since
input tasks output strings and are unaffected by output format.

Usage:
    python src/ablation/build_tokenization_ablation.py \
        --algorithm huffman --variant base64 \
        --input_file data/processed/huffman/data.jsonl \
        --output_file data/processed/huffman/codeio_alt_base64_msg.jsonl
"""

import os
import sys
import json
import base64
import heapq
from collections import Counter, namedtuple
from tqdm import tqdm

# Allow imports from parent src/ directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.utils import load_jsonl_yield, build_messages, get_freq_dict
from data.build_codeio_msg import build_io_pred, _sanitize


# ─────────────────────────────────────────────────────────────────────────────
# Huffman helpers (used both for ground-truth computation and refcode building)
# ─────────────────────────────────────────────────────────────────────────────

def _huffman_encode_bytes(input_str: str) -> bytes:
    """
    Encode *input_str* with a Huffman tree built from Counter(input_str).
    Returns the raw byte array (same logic embedded in the alternative refcode).
    """
    freq = dict(Counter(input_str))
    Node = namedtuple("Node", ["freq", "symbol", "left", "right"])
    Node.__lt__ = lambda a, b: a.freq < b.freq
    heap = [Node(f, sym, None, None) for sym, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        l = heapq.heappop(heap)
        r = heapq.heappop(heap)
        heapq.heappush(heap, Node(l.freq + r.freq, None, l, r))
    codebook: dict = {}

    def walk(node, prefix):
        if node.symbol is not None:
            codebook[node.symbol] = prefix or "0"
        else:
            walk(node.left, prefix + "0")
            walk(node.right, prefix + "1")

    walk(heap[0], "")
    bitstr = "".join(codebook[c] for c in input_str)
    padding = (-len(bitstr)) % 8
    bitstr += "0" * padding
    return bytes([int(bitstr[i : i + 8], 2) for i in range(0, len(bitstr), 8)])


# ─────────────────────────────────────────────────────────────────────────────
# Refcode / io_req builders
# ─────────────────────────────────────────────────────────────────────────────

def _make_huffman_base64_refcode(freq: dict) -> str:
    return (
        "def main_solution(uncompressed):\n"
        "    from collections import namedtuple\n"
        "    import heapq, base64\n"
        f"    freq = {freq}\n"
        "    Node = namedtuple('Node', ['freq','symbol','left','right'])\n"
        "    Node.__lt__ = lambda a,b: a.freq < b.freq\n"
        "    heap = [Node(f,sym,None,None) for sym,f in freq.items()]\n"
        "    heapq.heapify(heap)\n"
        "    while len(heap) > 1:\n"
        "        l = heapq.heappop(heap)\n"
        "        r = heapq.heappop(heap)\n"
        "        heapq.heappush(heap, Node(l.freq+r.freq, None, l, r))\n"
        "    codebook = {}\n"
        "    def walk(node, prefix):\n"
        "        if node.symbol is not None:\n"
        "            codebook[node.symbol] = prefix or '0'\n"
        "        else:\n"
        "            walk(node.left, prefix+'0')\n"
        "            walk(node.right, prefix+'1')\n"
        "    walk(heap[0], '')\n"
        "    bitstr = ''.join(codebook[c] for c in uncompressed)\n"
        "    padding = (-len(bitstr)) % 8\n"
        "    bitstr += '0'*padding\n"
        "    raw = bytes([int(bitstr[i:i+8],2) for i in range(0,len(bitstr),8)])\n"
        "    return base64.b64encode(raw).decode()"
    )


def _make_huffman_hex_refcode(freq: dict) -> str:
    return (
        "def main_solution(uncompressed):\n"
        "    from collections import namedtuple\n"
        "    import heapq\n"
        f"    freq = {freq}\n"
        "    Node = namedtuple('Node', ['freq','symbol','left','right'])\n"
        "    Node.__lt__ = lambda a,b: a.freq < b.freq\n"
        "    heap = [Node(f,sym,None,None) for sym,f in freq.items()]\n"
        "    heapq.heapify(heap)\n"
        "    while len(heap) > 1:\n"
        "        l = heapq.heappop(heap)\n"
        "        r = heapq.heappop(heap)\n"
        "        heapq.heappush(heap, Node(l.freq+r.freq, None, l, r))\n"
        "    codebook = {}\n"
        "    def walk(node, prefix):\n"
        "        if node.symbol is not None:\n"
        "            codebook[node.symbol] = prefix or '0'\n"
        "        else:\n"
        "            walk(node.left, prefix+'0')\n"
        "            walk(node.right, prefix+'1')\n"
        "    walk(heap[0], '')\n"
        "    bitstr = ''.join(codebook[c] for c in uncompressed)\n"
        "    padding = (-len(bitstr)) % 8\n"
        "    bitstr += '0'*padding\n"
        "    raw = bytes([int(bitstr[i:i+8],2) for i in range(0,len(bitstr),8)])\n"
        "    return raw.hex()"
    )


def _make_huffman_spaced_decimal_refcode(freq: dict) -> str:
    return (
        "def main_solution(uncompressed):\n"
        "    from collections import namedtuple\n"
        "    import heapq\n"
        f"    freq = {freq}\n"
        "    Node = namedtuple('Node', ['freq','symbol','left','right'])\n"
        "    Node.__lt__ = lambda a,b: a.freq < b.freq\n"
        "    heap = [Node(f,sym,None,None) for sym,f in freq.items()]\n"
        "    heapq.heapify(heap)\n"
        "    while len(heap) > 1:\n"
        "        l = heapq.heappop(heap)\n"
        "        r = heapq.heappop(heap)\n"
        "        heapq.heappush(heap, Node(l.freq+r.freq, None, l, r))\n"
        "    codebook = {}\n"
        "    def walk(node, prefix):\n"
        "        if node.symbol is not None:\n"
        "            codebook[node.symbol] = prefix or '0'\n"
        "        else:\n"
        "            walk(node.left, prefix+'0')\n"
        "            walk(node.right, prefix+'1')\n"
        "    walk(heap[0], '')\n"
        "    bitstr = ''.join(codebook[c] for c in uncompressed)\n"
        "    padding = (-len(bitstr)) % 8\n"
        "    bitstr += '0'*padding\n"
        "    raw = bytes([int(bitstr[i:i+8],2) for i in range(0,len(bitstr),8)])\n"
        "    return ' '.join(str(b) for b in raw)"
    )


def _make_huffman_char_hex_refcode(freq: dict) -> str:
    return (
        "def main_solution(uncompressed):\n"
        "    from collections import namedtuple\n"
        "    import heapq\n"
        f"    freq = {freq}\n"
        "    Node = namedtuple('Node', ['freq','symbol','left','right'])\n"
        "    Node.__lt__ = lambda a,b: a.freq < b.freq\n"
        "    heap = [Node(f,sym,None,None) for sym,f in freq.items()]\n"
        "    heapq.heapify(heap)\n"
        "    while len(heap) > 1:\n"
        "        l = heapq.heappop(heap)\n"
        "        r = heapq.heappop(heap)\n"
        "        heapq.heappush(heap, Node(l.freq+r.freq, None, l, r))\n"
        "    codebook = {}\n"
        "    def walk(node, prefix):\n"
        "        if node.symbol is not None:\n"
        "            codebook[node.symbol] = prefix or '0'\n"
        "        else:\n"
        "            walk(node.left, prefix+'0')\n"
        "            walk(node.right, prefix+'1')\n"
        "    walk(heap[0], '')\n"
        "    bitstr = ''.join(codebook[c] for c in uncompressed)\n"
        "    padding = (-len(bitstr)) % 8\n"
        "    bitstr += '0'*padding\n"
        "    raw = bytes([int(bitstr[i:i+8],2) for i in range(0,len(bitstr),8)])\n"
        "    return ' '.join(raw.hex())"
    )


def _make_huffman_binary_refcode(freq: dict) -> str:
    return (
        "def main_solution(uncompressed):\n"
        "    from collections import namedtuple\n"
        "    import heapq\n"
        f"    freq = {freq}\n"
        "    Node = namedtuple('Node', ['freq','symbol','left','right'])\n"
        "    Node.__lt__ = lambda a,b: a.freq < b.freq\n"
        "    heap = [Node(f,sym,None,None) for sym,f in freq.items()]\n"
        "    heapq.heapify(heap)\n"
        "    while len(heap) > 1:\n"
        "        l = heapq.heappop(heap)\n"
        "        r = heapq.heappop(heap)\n"
        "        heapq.heappush(heap, Node(l.freq+r.freq, None, l, r))\n"
        "    codebook = {}\n"
        "    def walk(node, prefix):\n"
        "        if node.symbol is not None:\n"
        "            codebook[node.symbol] = prefix or '0'\n"
        "        else:\n"
        "            walk(node.left, prefix+'0')\n"
        "            walk(node.right, prefix+'1')\n"
        "    walk(heap[0], '')\n"
        "    bitstr = ''.join(codebook[c] for c in uncompressed)\n"
        "    padding = (-len(bitstr)) % 8\n"
        "    bitstr += '0'*padding\n"
        "    raw = bytes([int(bitstr[i:i+8],2) for i in range(0,len(bitstr),8)])\n"
        "    return ' '.join(format(b, '08b') for b in raw)"
    )


_LZW_CSV_REFCODE = (
    "def main_solution(uncompressed):\n"
    "    dict_size = 256\n"
    "    dictionary = {chr(i): i for i in range(dict_size)}\n"
    "    w = \"\"\n"
    "    result = []\n"
    "    for c in uncompressed:\n"
    "        wc = w + c\n"
    "        if wc in dictionary:\n"
    "            w = wc\n"
    "        else:\n"
    "            result.append(dictionary[w])\n"
    "            dictionary[wc] = dict_size\n"
    "            dict_size += 1\n"
    "            w = c\n"
    "    if w:\n"
    "        result.append(dictionary[w])\n"
    '    return ",".join(str(x) for x in result)'
)

_LZW_SPACED_REFCODE = (
    "def main_solution(uncompressed):\n"
    "    dict_size = 256\n"
    "    dictionary = {chr(i): i for i in range(dict_size)}\n"
    "    w = \"\"\n"
    "    result = []\n"
    "    for c in uncompressed:\n"
    "        wc = w + c\n"
    "        if wc in dictionary:\n"
    "            w = wc\n"
    "        else:\n"
    "            result.append(dictionary[w])\n"
    "            dictionary[wc] = dict_size\n"
    "            dict_size += 1\n"
    "            w = c\n"
    "    if w:\n"
    "        result.append(dictionary[w])\n"
    '    return " ".join(str(x) for x in result)'
)

_RLE_COMPACT_REFCODE = (
    "def main_solution(uncompressed):\n"
    "    if not uncompressed:\n"
    "        return ''\n"
    "    result = []\n"
    "    prev_char = uncompressed[0]\n"
    "    count = 1\n"
    "    for c in uncompressed[1:]:\n"
    "        if c == prev_char:\n"
    "            count += 1\n"
    "        else:\n"
    "            result.append((prev_char, count))\n"
    "            prev_char = c\n"
    "            count = 1\n"
    "    result.append((prev_char, count))\n"
    "    return ''.join(f'{c}{n}' for c,n in result)"
)


_RLE_SPACED_REFCODE = (
    "def main_solution(uncompressed):\n"
    "    if not uncompressed:\n"
    "        return ''\n"
    "    result = []\n"
    "    prev_char = uncompressed[0]\n"
    "    count = 1\n"
    "    for c in uncompressed[1:]:\n"
    "        if c == prev_char:\n"
    "            count += 1\n"
    "        else:\n"
    "            result.append((prev_char, count))\n"
    "            prev_char = c\n"
    "            count = 1\n"
    "    result.append((prev_char, count))\n"
    "    return ' '.join(token for c, n in result for token in (c, str(n)))"
)


# ─────────────────────────────────────────────────────────────────────────────
# AE helpers (fraction format)
# ─────────────────────────────────────────────────────────────────────────────

def _ae_encode_fraction(input_str: str, freq: dict) -> str:
    """
    Re-run AE encoding with exact Fraction arithmetic and return the midpoint
    as a 'p/q' string instead of a float.  The freq dict must include 'EOF'.
    """
    from fractions import Fraction
    total = sum(freq.values())
    symbols = sorted(freq.keys())
    cum_counts: dict = {}
    running = 0
    for sym in symbols:
        cum_counts[sym] = running
        running += freq[sym]
    low, high = Fraction(0), Fraction(1)
    for c in list(input_str) + ["EOF"]:
        width = high - low
        high = low + width * Fraction(cum_counts[c] + freq[c], total)
        low  = low + width * Fraction(cum_counts[c], total)
    return str((low + high) / 2)   # e.g. "28/29"


def _make_ae_fraction_refcode(freq: dict) -> str:
    return (
        "def main_solution(uncompressed):\n"
        "    from fractions import Fraction\n"
        f"    freq = {freq}\n"
        "    total = sum(freq.values())\n"
        "    symbols = sorted(freq.keys())\n"
        "    cum_counts = {}\n"
        "    running = 0\n"
        "    for sym in symbols:\n"
        "        cum_counts[sym] = running\n"
        "        running += freq[sym]\n"
        "    low, high = Fraction(0), Fraction(1)\n"
        "    for c in list(uncompressed) + ['EOF']:\n"
        "        width = high - low\n"
        "        high = low + width * Fraction(cum_counts[c] + freq[c], total)\n"
        "        low  = low + width * Fraction(cum_counts[c], total)\n"
        "    return str((low + high) / 2)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Return-type hints (injected into the prompt via the <<<<return_type>>>> slot)
# ─────────────────────────────────────────────────────────────────────────────

_VARIANT_RETURN_TYPE_HINTS = {
    ("huffman", "base64"):         'str (Base64-encoded bytes, e.g. "qg==")',
    ("huffman", "hex"):            'str (hexadecimal bytes, e.g. "aa00ff")',
    ("huffman", "spaced_decimal"): 'str (space-separated byte values, e.g. "170 0 255")',
    ("huffman", "char_hex"):       'str (space-separated hex characters, e.g. "a a 0 0 f f")',
    ("huffman", "binary"):         'str (space-separated 8-bit binary strings, e.g. "10101010 00000000")',
    ("lzw",     "csv"):            'str (comma-separated integers, e.g. "85,256,257")',
    ("lzw",     "spaced"):         'str (space-separated integers, e.g. "85 256 257")',
    ("rle",     "compact"):        'str (compact RLE encoding, e.g. "U8A3")',
    ("rle",     "spaced"):         'str (space-separated char/count pairs, e.g. "U 8 A 3")',
    ("ae",      "fraction"):       'str (exact fraction, e.g. "28/29")',
}


IO_REQS = {
    ("huffman", "base64"): (
        "Input:\n"
        "  `uncompressed` (str): The input string to be compressed.\n"
        "Output:\n"
        "  `return` (str): The Huffman-encoded bytes as a Base64 string."
    ),
    ("huffman", "hex"): (
        "Input:\n"
        "  `uncompressed` (str): The input string to be compressed.\n"
        "Output:\n"
        "  `return` (str): The Huffman-encoded bytes as a hexadecimal string."
    ),
    ("lzw", "csv"): (
        "Input:\n"
        "  `uncompressed` (str): The input string to be compressed. "
        "It should consist of standard ASCII characters.\n\n"
        "Output:\n"
        "  `return` (str): The LZW-compressed codes as a comma-separated string of integers."
    ),
    ("rle", "compact"): (
        "Input:\n"
        "  `uncompressed` (str): The input string to be compressed.\n\n"
        "Output:\n"
        '  `return` (str): The RLE-compressed string in compact format (e.g., "U8A3" '
        'for 8 consecutive "U"s followed by 3 consecutive "A"s).'
    ),
    ("huffman", "spaced_decimal"): (
        "Input:\n"
        "  `uncompressed` (str): The input string to be compressed.\n"
        "Output:\n"
        '  `return` (str): The Huffman-encoded bytes as space-separated decimal integers '
        '(e.g., "170 0 255").'
    ),
    ("huffman", "char_hex"): (
        "Input:\n"
        "  `uncompressed` (str): The input string to be compressed.\n"
        "Output:\n"
        '  `return` (str): The Huffman-encoded bytes as space-separated individual '
        'hexadecimal characters (e.g., "a a 0 0 f f").'
    ),
    ("huffman", "binary"): (
        "Input:\n"
        "  `uncompressed` (str): The input string to be compressed.\n"
        "Output:\n"
        '  `return` (str): The Huffman-encoded bytes as space-separated 8-bit binary '
        'strings (e.g., "10101010 00000000 11111111").'
    ),
    ("lzw", "spaced"): (
        "Input:\n"
        "  `uncompressed` (str): The input string to be compressed. "
        "It should consist of standard ASCII characters.\n\n"
        "Output:\n"
        '  `return` (str): The LZW-compressed codes as space-separated integers '
        '(e.g., "85 256 257").'
    ),
    ("rle", "spaced"): (
        "Input:\n"
        "  `uncompressed` (str): The input string to be compressed.\n\n"
        "Output:\n"
        '  `return` (str): The RLE-compressed string as space-separated character/count '
        'pairs (e.g., "U 8 A 3" for 8 consecutive "U"s followed by 3 consecutive "A"s).'
    ),
    ("ae", "fraction"): (
        "Input:\n"
        "  `uncompressed` (str): The input string to be compressed.\n\n"
        "Output:\n"
        '  `return` (str): The Arithmetic Encoding midpoint expressed as an exact '
        'fraction string (e.g., "28/29").'
    ),
}

_INPUT_RETURN_TYPE_HINT = 'str (e.g. "hello world")'

_INPUT_TOKENIZATION_DESCRIPTIONS = {
    "raw": "raw string",
    "base64": "Base64-encoded UTF-8 string",
    "hex": "hex-encoded UTF-8 bytes",
    "codepoints": "space-separated Unicode code points",
    "unicode_escape": "Python unicode-escaped string",
}

# IO requirements for the *decoding* direction — used by the input inversion task,
# where the model is shown the alt-format encoder and must predict the original string.
IO_REQS_DECODING = {
    ("huffman", "base64"): (
        "Input:\n"
        "  `compressed` (str): The Huffman-encoded bytes as a Base64 string.\n"
        "Output:\n"
        "  `return` (str): The original uncompressed string."
    ),
    ("huffman", "hex"): (
        "Input:\n"
        "  `compressed` (str): The Huffman-encoded bytes as a hexadecimal string.\n"
        "Output:\n"
        "  `return` (str): The original uncompressed string."
    ),
    ("huffman", "spaced_decimal"): (
        "Input:\n"
        '  `compressed` (str): The Huffman-encoded bytes as space-separated decimal integers.\n'
        "Output:\n"
        "  `return` (str): The original uncompressed string."
    ),
    ("huffman", "char_hex"): (
        "Input:\n"
        '  `compressed` (str): The Huffman-encoded bytes as space-separated individual hexadecimal characters.\n'
        "Output:\n"
        "  `return` (str): The original uncompressed string."
    ),
    ("huffman", "binary"): (
        "Input:\n"
        '  `compressed` (str): The Huffman-encoded bytes as space-separated 8-bit binary strings.\n'
        "Output:\n"
        "  `return` (str): The original uncompressed string."
    ),
    ("lzw", "csv"): (
        "Input:\n"
        "  `compressed` (str): The LZW-compressed codes as a comma-separated string of integers.\n"
        "Output:\n"
        "  `return` (str): The original uncompressed string."
    ),
    ("lzw", "spaced"): (
        "Input:\n"
        '  `compressed` (str): The LZW-compressed codes as space-separated integers.\n'
        "Output:\n"
        "  `return` (str): The original uncompressed string."
    ),
    ("rle", "compact"): (
        "Input:\n"
        '  `compressed` (str): The RLE-compressed string in compact format (e.g., "U8A3").\n'
        "Output:\n"
        "  `return` (str): The original uncompressed string."
    ),
    ("rle", "spaced"): (
        "Input:\n"
        '  `compressed` (str): The RLE-compressed string as space-separated character/count pairs.\n'
        "Output:\n"
        "  `return` (str): The original uncompressed string."
    ),
    ("ae", "fraction"): (
        "Input:\n"
        '  `compressed` (str): The Arithmetic Encoding midpoint as an exact fraction string (e.g., "28/29").\n'
        "Output:\n"
        "  `return` (str): The original uncompressed string."
    ),
}


def get_refcode(algorithm: str, variant: str, input_xx: str, freq: dict = None) -> str:
    """Return the alternative encoding refcode for this algo/variant/input."""
    if algorithm == "huffman":
        freq = dict(Counter(input_xx))
        if variant == "base64":
            return _make_huffman_base64_refcode(freq)
        elif variant == "hex":
            return _make_huffman_hex_refcode(freq)
        elif variant == "spaced_decimal":
            return _make_huffman_spaced_decimal_refcode(freq)
        elif variant == "char_hex":
            return _make_huffman_char_hex_refcode(freq)
        elif variant == "binary":
            return _make_huffman_binary_refcode(freq)
    elif algorithm == "lzw":
        if variant == "csv":
            return _LZW_CSV_REFCODE
        elif variant == "spaced":
            return _LZW_SPACED_REFCODE
    elif algorithm == "rle":
        if variant == "compact":
            return _RLE_COMPACT_REFCODE
        elif variant == "spaced":
            return _RLE_SPACED_REFCODE
    elif algorithm == "ae":
        if variant == "fraction":
            if freq is None:
                raise ValueError("AE/fraction requires a freq dict (pass freq=get_freq_dict(ioid, 'ae'))")
            return _make_ae_fraction_refcode(freq)
    raise ValueError(f"Unsupported algorithm/variant: {algorithm}/{variant}")


def _encode_input_for_prompt(input_xx: str, input_tokenization: str) -> str:
    """Convert raw input string into the requested tokenization variant for prompt display."""
    if input_tokenization == "raw":
        return input_xx
    if input_tokenization == "base64":
        return base64.b64encode(input_xx.encode("utf-8")).decode("ascii")
    if input_tokenization == "hex":
        return input_xx.encode("utf-8").hex()
    if input_tokenization == "codepoints":
        return " ".join(str(ord(ch)) for ch in input_xx)
    if input_tokenization == "unicode_escape":
        return input_xx.encode("unicode_escape").decode("ascii")
    raise ValueError(f"Unsupported input_tokenization: {input_tokenization}")


def _wrap_refcode_for_input_tokenization(refcode: str, input_tokenization: str) -> str:
    """
    Wrap refcode so main_solution accepts a tokenized input representation and
    decodes it before executing the original compression logic.
    """
    if input_tokenization == "raw":
        return refcode

    marker = "def main_solution(uncompressed):"
    if marker not in refcode:
        raise ValueError("Unexpected refcode format: cannot find main_solution(uncompressed)")

    inner_refcode = refcode.replace(marker, "def _main_solution_raw(uncompressed):", 1)

    if input_tokenization == "base64":
        decode_lines = [
            "    import base64",
            "    _raw = base64.b64decode(uncompressed).decode('utf-8')",
        ]
    elif input_tokenization == "hex":
        decode_lines = [
            "    _raw = bytes.fromhex(uncompressed).decode('utf-8')",
        ]
    elif input_tokenization == "codepoints":
        decode_lines = [
            "    _raw = '' if not uncompressed.strip() else ''.join(chr(int(tok)) for tok in uncompressed.split())",
        ]
    elif input_tokenization == "unicode_escape":
        decode_lines = [
            "    _raw = bytes(uncompressed, 'utf-8').decode('unicode_escape')",
        ]
    else:
        raise ValueError(f"Unsupported input_tokenization: {input_tokenization}")

    wrapper_lines = [
        "",
        "def main_solution(uncompressed):",
        *decode_lines,
        "    return _main_solution_raw(_raw)",
    ]
    return inner_refcode + "\n".join(wrapper_lines)


def _augment_io_req_for_input_tokenization(io_req: str, input_tokenization: str) -> str:
    """Add a short note to IO requirements for tokenized input variants."""
    if input_tokenization == "raw":
        return io_req
    desc = _INPUT_TOKENIZATION_DESCRIPTIONS[input_tokenization]
    note = (
        "Input format note:\n"
        f"  The provided `uncompressed` value is a {desc}. "
        "Decode it to the original string before applying the algorithm.\n\n"
    )
    return note + io_req


def compute_output_alt(algorithm: str, variant: str, input_xx: str, original_output, freq: dict = None) -> str:
    """
    Compute the ground-truth alternative output for a sample.

    For huffman variants we re-run the encoding with Counter(input_xx) to
    ensure the ground truth matches the embedded freq dict in the refcode
    (the original data may have used a slightly different freq dict that
    included an EOF symbol).  For lzw/csv and rle/compact the algorithm is
    unchanged so we convert the stored output directly.
    """
    if algorithm == "huffman":
        raw = _huffman_encode_bytes(input_xx)
        if variant == "base64":
            return base64.b64encode(raw).decode()
        elif variant == "hex":
            return raw.hex()
        elif variant == "spaced_decimal":
            return " ".join(str(b) for b in raw)
        elif variant == "char_hex":
            return " ".join(raw.hex())
        elif variant == "binary":
            return " ".join(format(b, "08b") for b in raw)
    elif algorithm == "lzw":
        if variant == "csv":
            return ",".join(str(x) for x in original_output)
        elif variant == "spaced":
            return " ".join(str(x) for x in original_output)
    elif algorithm == "rle":
        if variant == "compact":
            return "".join(f"{c}{n}" for c, n in original_output)
        elif variant == "spaced":
            return " ".join(token for c, n in original_output for token in (c, str(n)))
    elif algorithm == "ae":
        if variant == "fraction":
            if freq is None:
                raise ValueError("AE/fraction requires a freq dict")
            return _ae_encode_fraction(input_xx, freq)
    raise ValueError(f"Unsupported algorithm/variant: {algorithm}/{variant}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

VALID_COMBOS = {
    ("huffman", "base64"),
    ("huffman", "hex"),
    ("huffman", "spaced_decimal"),
    ("huffman", "char_hex"),
    ("huffman", "binary"),
    ("lzw", "csv"),
    ("lzw", "spaced"),
    ("rle", "compact"),
    ("rle", "spaced"),
    ("ae",  "fraction"),
}

OUTPUT_TASKS = [
    "output_execution_prediction",
    # output_execution_prediction_with_inversion is intentionally omitted:
    # it requires an alt-format *decoder* as the refcode, but this module only
    # defines alt-format encoders.  Using the encoder with inversion=True would
    # produce a prompt that contradicts itself ("this function is a decoder").
]

INPUT_TASKS = [
    # input_execution_prediction is intentionally omitted: it requires an
    # alt-format decoder as the refcode (model runs decoder(Z) → X directly).
    "input_execution_prediction_with_inversion",
    # Model is shown the alt-format *encoder* + the alt-format compressed output Z,
    # and must predict the original string X by inverting the encoder mentally.
    # This tests whether tokenization of Z affects inverse-reasoning accuracy.
]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build tokenization-ablation prompt files."
    )
    parser.add_argument(
        "--algorithm", required=True, choices=["huffman", "lzw", "rle", "ae"],
        help="Compression algorithm to ablate."
    )
    parser.add_argument(
        "--variant", required=True,
        choices=["base64", "hex", "csv", "compact", "spaced_decimal", "char_hex", "binary", "spaced", "fraction"],
        help="Alternative output format variant."
    )
    parser.add_argument("--input_file",  required=True,  help="Path to data.jsonl")
    parser.add_argument("--output_file", required=True,  help="Path to write msg.jsonl")
    parser.add_argument(
        "--input_tokenization",
        default="raw",
        choices=["raw", "base64", "hex", "codepoints", "unicode_escape"],
        help="How to represent the input string in output-prediction prompts.",
    )
    parser.add_argument(
        "--prompt_type", default="zero_shot", choices=["zero_shot"],
        help="Prompt style (only zero_shot supported)."
    )
    args = parser.parse_args()

    combo = (args.algorithm, args.variant)
    if combo not in VALID_COMBOS:
        parser.error(
            f"Invalid combination: --algorithm {args.algorithm} --variant {args.variant}. "
            f"Valid combinations: {sorted(VALID_COMBOS)}"
        )

    io_req          = _augment_io_req_for_input_tokenization(IO_REQS[combo], args.input_tokenization)
    io_req_decoding = IO_REQS_DECODING[combo]

    # ── Resume support ──────────────────────────────────────────────────────
    done_ids: set = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    done_ids.add(json.loads(_line).get("id"))
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} samples already in {args.output_file}")

    # ── Stream output ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    out_f = open(args.output_file, "a" if done_ids else "w", encoding="utf-8")

    dt = load_jsonl_yield(args.input_file)
    written = 0

    try:
        for iid, item in enumerate(tqdm(dt, desc=f"{args.algorithm}/{args.variant}")):
            # Apply the same Level-1 sanitization as the standard pipeline so that
            # "HUFFMAN compression" → "a custom algorithm" in the stored metadata.
            problem_description = _sanitize(item["problem_description"])
            algorithm = item.get("algorithm", args.algorithm)

            for ioid, io in enumerate(item["ios"]):
                input_xx  = io["input"]
                output_xx = io["output"]
                category  = io["category"]

                # AE needs the per-sample freq dict from disk; other algos derive it
                # from Counter(input_xx) or use a fixed refcode.
                ae_freq = None
                if args.algorithm == "ae":
                    ae_freq = get_freq_dict(ioid=ioid, algo="ae")
                    if not ae_freq:
                        print(f"[WARN] No freq dict for item={iid} io={ioid}; skipping")
                        continue

                try:
                    refcode = get_refcode(args.algorithm, args.variant, input_xx, freq=ae_freq)
                    output_alt = compute_output_alt(
                        args.algorithm, args.variant, input_xx, output_xx, freq=ae_freq
                    )
                except Exception as e:
                    print(
                        f"[WARN] Could not build refcode/output_alt for item={iid} io={ioid}: {e}"
                    )
                    continue

                return_type_hint = _VARIANT_RETURN_TYPE_HINTS.get(
                    (args.algorithm, args.variant), ""
                )

                for n_task, task in enumerate(OUTPUT_TASKS + INPUT_TASKS):
                    sample_id = f"{iid}_{ioid}_{n_task}_{args.input_tokenization}"
                    if sample_id in done_ids:
                        continue

                    is_input_task = task.startswith("input_")
                    if is_input_task:
                        # Input inversion: show encoder + alt-format Z, predict original X.
                        task_io_req      = io_req_decoding
                        task_io_dir      = "i"
                        task_outputx     = output_alt   # shown as "Given the following output: Z"
                        task_return_hint = _INPUT_RETURN_TYPE_HINT
                        task_refcode     = refcode
                        task_inputx      = input_xx
                    else:
                        task_io_req      = io_req
                        task_io_dir      = "o"
                        task_outputx     = output_alt
                        task_return_hint = return_type_hint
                        task_refcode     = _wrap_refcode_for_input_tokenization(
                            refcode, args.input_tokenization
                        )
                        task_inputx      = _encode_input_for_prompt(
                            input_xx, args.input_tokenization
                        )

                    prompt = build_io_pred(
                        problem_description,
                        task_io_req,
                        task_refcode,
                        task_inputx,
                        task_outputx,
                        prompt_type=args.prompt_type,
                        io=task_io_dir,
                        inversion=is_input_task,   # inversion=True for input tasks
                        return_type_hint=task_return_hint,
                    )
                    msg = build_messages(
                        prompt=prompt,
                        system_message=(
                            "You are a helpful programming assistant designed to execute code. "
                            "You must verify your own output via a round\u2011trip check and "
                            "self\u2011correct before returning the final JSON."
                        ),
                    )

                    sample = {
                        "problem_description": problem_description,
                        "io_requirements":     task_io_req,
                        "messages":            msg,
                        "itemid":              iid,
                        "ioid":                ioid,
                        "io_pred":             task,
                        "category":            category,
                        "algorithm":           algorithm,
                        "variant":             args.variant,
                        "input_tokenization":  args.input_tokenization,
                        "refcode":             refcode,
                        "input":               input_xx,
                        "input_prompt":        task_inputx,
                        "output":              output_xx,
                        "output_alt":          output_alt,
                        "id":                  sample_id,
                    }

                    out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    out_f.flush()
                    written += 1
    finally:
        out_f.close()

    print(f"Done. Wrote {written} new samples to {args.output_file}")


if __name__ == "__main__":
    main()
