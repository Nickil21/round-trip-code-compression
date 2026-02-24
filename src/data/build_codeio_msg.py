import os
import json
import re
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

from core.utils import load_jsonl_yield, build_messages, get_freq_dict, write_jsonl
from tqdm import tqdm
from core.prompts import INPUT_PRED_TEMPLATE, INPUT_PRED_TEMPLATE_INV_EXTRA, OUTPUT_PRED_TEMPLATE, OUTPUT_PRED_TEMPLATE_INV_EXTRA

# Level 1: regex to strip known algorithm names from text
_ALGO_RE = re.compile(
    r'\b(?:RLE|Run[- ]?Length(?: Encoding| Compression)?|'
    r'AE|Arithmetic(?: Encoding| Coding| Compression)?|'
    r'LZW|Lempel[- ]?Ziv[- ]?Welch|'
    r'Huffman(?: Coding| Encoding| Compression)?)'
    r'(?:\s+(?:compression|encoding|coding|algorithm))?\b',
    re.IGNORECASE
)

def _sanitize(text):
    """Strip known compression algorithm names from text (Level 1)."""
    return _ALGO_RE.sub('a custom algorithm', text)


_OUTPUT_RETURN_TYPE_HINTS = {
    "lzw":     "list of int (e.g. [256, 97, 258])",
    "ae":      "float (e.g. 0.6319081203782178)",
    "rle":     'list of [str, int] pairs (e.g. [["a", 3], ["b", 2]])',
    "huffman": "list of int (e.g. [102, 114, 111])",
}
_INPUT_RETURN_TYPE_HINT = 'str (e.g. "hello world")'


def _validate_io_types(input_xx, output_xx, algo, iid, ioid):
    """Raise TypeError if ground-truth input/output values have unexpected types for algo."""
    if not isinstance(input_xx, str):
        raise TypeError(
            f"item {iid} io {ioid}: 'input' expected str, got {type(input_xx).__name__}"
        )
    if algo == "ae":
        if not isinstance(output_xx, (int, float)):
            raise TypeError(
                f"item {iid} io {ioid}: 'output' expected float for ae, got {type(output_xx).__name__}"
            )
    elif algo in ("lzw", "huffman"):
        if not isinstance(output_xx, list):
            raise TypeError(
                f"item {iid} io {ioid}: 'output' expected list for {algo}, got {type(output_xx).__name__}"
            )
    elif algo == "rle":
        if not isinstance(output_xx, list) or not all(
            isinstance(el, (list, tuple)) and len(el) == 2 for el in output_xx
        ):
            raise TypeError(
                f"item {iid} io {ioid}: 'output' expected list of 2-tuples for rle, got {type(output_xx).__name__}"
            )


def build_io_pred(problem_statement, io_req, refcode, inputx, outputx, prompt_type, io="i", w_refcode=True, inversion=False, return_type_hint=""):

    if prompt_type == "zero_shot":
        if inversion:
            template = INPUT_PRED_TEMPLATE_INV_EXTRA if io=="i" else OUTPUT_PRED_TEMPLATE_INV_EXTRA
        else:
            template = INPUT_PRED_TEMPLATE if io=="i" else OUTPUT_PRED_TEMPLATE

    # elif prompt_type == "one_shot":
    #     template = input_exec_pred_template + "\n\n" + few_shot_template_input if io=="i" else output_exec_pred_template + "\n\n" + few_shot_template_output

    else:
        # Fix: was missing — passing an unsupported prompt_type would reach `prompt = template.replace(...)`
        # with `template` undefined, raising an UnboundLocalError.
        raise NotImplementedError(f"prompt_type '{prompt_type}' is not implemented")

    prompt = template.replace("<<<<query>>>>", problem_statement).replace("<<<<io_req>>>>", io_req).replace("<<<<refcode>>>>", refcode)
    tag = "<<<<output>>>>" if io=="i" else "<<<<input>>>>"
    inputxx = f"{inputx}"
    outputxx = f"{outputx}"
    prompt = prompt.replace(tag, outputxx if io=="i" else inputxx)
    prompt = prompt.replace("<<<<return_type>>>>", return_type_hint)
    # if w_refcode:
    #     if inversion:
    #         refcodepart = refcode_template_inversion.replace("<<<<refcode>>>>", refcode)
    #     else:
    #         refcodepart = refcode_template.replace("<<<<refcode>>>>", refcode)
    #     prompt+="\n\n"+refcodepart

    # print("PROMPT", prompt)

    return prompt


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='data/pyedu_toy.jsonl')
    parser.add_argument('--output_file', type=str, default='data/codeio_1k_msg.jsonl')
    parser.add_argument('--algorithm', type=str)
    # Fix: removed "one_shot" — the branch is commented out; accepting it caused UnboundLocalError at runtime.
    parser.add_argument('--prompt_type', type=str, default="zero_shot", choices=["zero_shot"])
    parser.add_argument('--blind', action='store_true',
                        help='Anonymize algorithm: strip names from text (L1), obfuscate '
                             'variable names in code (L2), rename function parameters to x (L3).')
    args = parser.parse_args()

    fn = args.input_file
    ofn = args.output_file

    # Fix: resume support — collect already-written sample IDs so interrupted runs can continue.
    done_ids: set = set()
    if os.path.exists(ofn):
        with open(ofn, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    done_ids.add(json.loads(_line).get("id"))
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} samples already in {ofn}")

    dt = load_jsonl_yield(fn)
    # Fix: stream each sample to disk immediately instead of buffering all in memory.
    out_f = open(ofn, "a" if done_ids else "w", encoding="utf-8")
    for iid,item in enumerate(tqdm(dt)):
        # print(item)
        problem_description = item['problem_description']
        # Level 1: strip algorithm name from problem description
        if args.blind:
            problem_description = _sanitize(problem_description)
        # Fix: removed dead `io_req = item['io_requirements']` — it was immediately overwritten
        # inside the task loop and never read.
        algorithm = item['algorithm']
        # refcode = item['refcode']
        for ioid,io in enumerate(item['ios']):
            # uplimit = 3
            # if ioid>=uplimit:break # we now first only use the first 3 io
            input_xx = io['input']
            output_xx = io['output']
            category = io['category']

            try:
                _validate_io_types(input_xx, output_xx, args.algorithm, iid, ioid)
            except TypeError as e:
                print(f"WARNING: skipping ioid={ioid} of item {iid} — bad ground-truth types: {e}")
                continue

            if args.algorithm == "lzw":
                encoding_fn = (
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
                    "    return result"
                )

                decoding_fn = (
                    "def main_solution(compressed):\n"
                    "    dict_size = 256\n"
                    "    dictionary = {i: chr(i) for i in range(dict_size)}\n"
                    "    result = []\n"
                    "    w = chr(compressed.pop(0))\n"
                    "    result.append(w)\n"
                    "    for k in compressed:\n"
                    "        if k in dictionary:\n"
                    "            entry = dictionary[k]\n"
                    "        elif k == dict_size:\n"
                    "            entry = w + w[0]\n"
                    "        else:\n"
                    "            raise ValueError(\"Bad compressed k: %s\" % k)\n"
                    "        result.append(entry)\n"
                    "        dictionary[dict_size] = w + entry[0]\n"
                    "        dict_size += 1\n"
                    "        w = entry\n"
                    "    return \"\".join(result)"
                )

                io_req_encoding = (
                    "Input:\n  `uncompressed` (str): The input string to be compressed. "
                    "It should consist of standard ASCII characters.\n\n"
                    "Output:\n  `return` (list of int): A list of integers representing "
                    "the compressed form of the input string using LZW encoding."
                )

                io_req_decoding = (
                    "Input:\n  `compressed` (list of int): A list of integers representing "
                    "the compressed form of the input string using LZW encoding.\n\n"
                    "Output:\n  `return` (str): The original input string before compression."
                )

                if args.blind:
                    # Level 2+3: obfuscated variable names, parameter renamed to x
                    encoding_fn = (
                        "def main_solution(x):\n"
                        "    n = 256\n"
                        "    d = {chr(i): i for i in range(n)}\n"
                        "    w = \"\"\n"
                        "    r = []\n"
                        "    for c in x:\n"
                        "        wc = w + c\n"
                        "        if wc in d:\n"
                        "            w = wc\n"
                        "        else:\n"
                        "            r.append(d[w])\n"
                        "            d[wc] = n\n"
                        "            n += 1\n"
                        "            w = c\n"
                        "    if w:\n"
                        "        r.append(d[w])\n"
                        "    return r"
                    )
                    decoding_fn = (
                        "def main_solution(x):\n"
                        "    n = 256\n"
                        "    d = {i: chr(i) for i in range(n)}\n"
                        "    r = []\n"
                        "    w = chr(x.pop(0))\n"
                        "    r.append(w)\n"
                        "    for k in x:\n"
                        "        if k in d:\n"
                        "            e = d[k]\n"
                        "        elif k == n:\n"
                        "            e = w + w[0]\n"
                        "        else:\n"
                        "            raise ValueError(\"Bad k: %s\" % k)\n"
                        "        r.append(e)\n"
                        "        d[n] = w + e[0]\n"
                        "        n += 1\n"
                        "        w = e\n"
                        "    return \"\".join(r)"
                    )
                    io_req_encoding = (
                        "Input:\n  `x` (str): The input string.\n\n"
                        "Output:\n  `return` (list of int): A list of integers representing the encoded form."
                    )
                    io_req_decoding = (
                        "Input:\n  `x` (list of int): A list of integers representing the encoded form.\n\n"
                        "Output:\n  `return` (str): The original string."
                    )

            elif args.algorithm == "ae":
                freq = get_freq_dict(ioid=ioid, algo="ae")  # you must provide ioid externally
                # Fix: replaced float arithmetic (0.0, 1.0) with exact Fraction arithmetic.
                # Python float accumulates rounding errors as the [low, high] interval shrinks,
                # causing the returned midpoint to diverge from the ground-truth value.
                encoding_fn = (
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
                    "        low = low + width * Fraction(cum_counts[c], total)\n"
                    "    return float((low + high) / 2)"
                )

                decoding_fn = (
                    "def main_solution(compressed):\n"
                    "    from fractions import Fraction\n"
                    f"    freq = {freq}\n"
                    "    total = sum(freq.values())\n"
                    "    symbols = sorted(freq.keys())\n"
                    "    cum_counts = {}\n"
                    "    running = 0\n"
                    "    for s in symbols:\n"
                    "        cum_counts[s] = running\n"
                    "        running += freq[s]\n"
                    "    low, high = Fraction(0), Fraction(1)\n"
                    "    value = Fraction(str(compressed))\n"
                    "    result = []\n"
                    "    while True:\n"
                    "        width = high - low\n"
                    "        scaled = (value - low) / width * total\n"
                    "        for s in symbols:\n"
                    "            if cum_counts[s] <= scaled < cum_counts[s] + freq[s]:\n"
                    "                symbol = s\n"
                    "                break\n"
                    "        if symbol == 'EOF':\n"
                    "            break\n"
                    "        result.append(symbol)\n"
                    "        high = low + width * Fraction(cum_counts[symbol] + freq[symbol], total)\n"
                    "        low = low + width * Fraction(cum_counts[symbol], total)\n"
                    "    return ''.join(result)"
                )

                io_req_encoding = (
                    "Input:\n  `uncompressed` (str): The input string to be compressed. "
                    "It should consist of standard ASCII characters.\n\n"
                    "Output:\n  `return` (float): A probability value representing the "
                    "compressed form of the input string using Arithmetic Encoding."
                )

                io_req_decoding = (
                    "Input:\n  `compressed` (float): A probability value representing the "
                    "compressed form of the input string using Arithmetic Encoding.\n\n"
                    "Output:\n  `return` (str): The original input string before compression."
                )

                if args.blind:
                    # Level 2+3: obfuscated variable names, parameter renamed to x
                    encoding_fn = (
                        "def main_solution(x):\n"
                        "    from fractions import Fraction\n"
                        f"    p = {freq}\n"
                        "    t = sum(p.values())\n"
                        "    s = sorted(p.keys())\n"
                        "    c = {}\n"
                        "    n = 0\n"
                        "    for k in s:\n"
                        "        c[k] = n\n"
                        "        n += p[k]\n"
                        "    lo, hi = Fraction(0), Fraction(1)\n"
                        "    for a in list(x) + ['EOF']:\n"
                        "        w = hi - lo\n"
                        "        hi = lo + w * Fraction(c[a] + p[a], t)\n"
                        "        lo = lo + w * Fraction(c[a], t)\n"
                        "    return float((lo + hi) / 2)"
                    )
                    decoding_fn = (
                        "def main_solution(x):\n"
                        "    from fractions import Fraction\n"
                        f"    p = {freq}\n"
                        "    t = sum(p.values())\n"
                        "    s = sorted(p.keys())\n"
                        "    c = {}\n"
                        "    n = 0\n"
                        "    for k in s:\n"
                        "        c[k] = n\n"
                        "        n += p[k]\n"
                        "    lo, hi = Fraction(0), Fraction(1)\n"
                        "    v = Fraction(str(x))\n"
                        "    r = []\n"
                        "    while True:\n"
                        "        w = hi - lo\n"
                        "        sc = (v - lo) / w * t\n"
                        "        for a in s:\n"
                        "            if c[a] <= sc < c[a] + p[a]:\n"
                        "                sym = a\n"
                        "                break\n"
                        "        if sym == 'EOF':\n"
                        "            break\n"
                        "        r.append(sym)\n"
                        "        hi = lo + w * Fraction(c[sym] + p[sym], t)\n"
                        "        lo = lo + w * Fraction(c[sym], t)\n"
                        "    return ''.join(r)"
                    )
                    io_req_encoding = (
                        "Input:\n  `x` (str): The input string.\n\n"
                        "Output:\n  `return` (float): A floating-point value representing the encoded form."
                    )
                    io_req_decoding = (
                        "Input:\n  `x` (float): A floating-point value representing the encoded form.\n\n"
                        "Output:\n  `return` (str): The original string."
                    )

            elif args.algorithm == "rle":
                encoding_fn = (
                    "def main_solution(uncompressed):\n"
                    "    if not uncompressed:\n"
                    "        return []\n"
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
                    "    return result"
                )

                decoding_fn = (
                    "def main_solution(compressed):\n"
                    "    result = []\n"
                    "    for char, count in compressed:\n"
                    "        result.append(char * count)\n"
                    "    return ''.join(result)"
                )

                io_req_encoding = (
                    "Input:\n  `uncompressed` (str): The input string to be compressed.\n\n"
                    "Output:\n  `return` (list of tuple): A list of (char, count) tuples representing "
                    "the RLE compressed string."
                )

                io_req_decoding = (
                    "Input:\n  `compressed` (list of tuple): A list of (char, count) tuples from the RLE compression.\n\n"
                    "Output:\n  `return` (str): The original uncompressed string."
                )

                if args.blind:
                    # Level 2+3: obfuscated variable names, parameter renamed to x
                    encoding_fn = (
                        "def main_solution(x):\n"
                        "    if not x:\n"
                        "        return []\n"
                        "    r = []\n"
                        "    a = x[0]\n"
                        "    b = 1\n"
                        "    for c in x[1:]:\n"
                        "        if c == a:\n"
                        "            b += 1\n"
                        "        else:\n"
                        "            r.append((a, b))\n"
                        "            a = c\n"
                        "            b = 1\n"
                        "    r.append((a, b))\n"
                        "    return r"
                    )
                    decoding_fn = (
                        "def main_solution(x):\n"
                        "    r = []\n"
                        "    for a, b in x:\n"
                        "        r.append(a * b)\n"
                        "    return ''.join(r)"
                    )
                    io_req_encoding = (
                        "Input:\n  `x` (str): The input string.\n\n"
                        "Output:\n  `return` (list of tuple): A list of tuples representing the encoded form."
                    )
                    io_req_decoding = (
                        "Input:\n  `x` (list of tuple): A list of tuples representing the encoded form.\n\n"
                        "Output:\n  `return` (str): The original string."
                    )

            elif args.algorithm == "huffman":
                # retrieve freq dict from metadata
                # Note: algo="ae" is intentional — AE and Huffman share the same char-frequency
                # file (*_freq.json); get_freq_dict(algo="huffman") returns codebook+padding, not freq.
                # Fix: strip 'EOF' from freq — the ground-truth data was generated without EOF in
                # the Huffman tree. Including EOF produces a different codebook and wrong byte values.
                freq = {k: v for k, v in get_freq_dict(ioid=ioid, algo="ae").items() if k != "EOF"}
                d = get_freq_dict(ioid=ioid, algo="huffman")
                codebook, padding = d['codebook'], d['padding']
                encoding_fn = f"""
def main_solution(uncompressed):
    from collections import namedtuple
    import heapq
    freq = {freq}
    Node = namedtuple('Node', ['freq','symbol','left','right'])
    Node.__lt__ = lambda a,b: a.freq < b.freq
    heap = [Node(f,sym,None,None) for sym,f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        l = heapq.heappop(heap)
        r = heapq.heappop(heap)
        heapq.heappush(heap, Node(l.freq+r.freq, None, l, r))
    codebook = {{}}
    def walk(node, prefix):
        if node.symbol is not None:
            codebook[node.symbol] = prefix or '0'
        else:
            walk(node.left, prefix+'0')
            walk(node.right, prefix+'1')
    walk(heap[0], '')
    bitstr = ''.join(codebook[c] for c in uncompressed)
    padding = (-len(bitstr)) % 8
    bitstr += '0'*padding
    return [int(bitstr[i:i+8],2) for i in range(0,len(bitstr),8)]
"""

                decoding_fn = f"""
def main_solution(compressed):
    from collections import namedtuple
    import heapq
    padding = {padding}
    freq = {freq}
    Node = namedtuple('Node', ['freq','symbol','left','right'])
    Node.__lt__ = lambda a,b: a.freq < b.freq
    heap = [Node(f,sym,None,None) for sym,f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        l = heapq.heappop(heap)
        r = heapq.heappop(heap)
        heapq.heappush(heap, Node(l.freq+r.freq, None, l, r))
    root = heap[0]
    bitstr = ''.join(map(lambda x: '{{:08b}}'.format(x), compressed))
    bitstr = bitstr[:-padding] if padding else bitstr
    result = []
    node = root
    for bit in bitstr:
        node = node.left if bit=='0' else node.right
        if node.symbol is not None:
            result.append(node.symbol)
            node = root
    return ''.join(result)
"""

                io_req_encoding = """
Input:
  `uncompressed` (str): The input string to be compressed.
Output:
  `return` (list of int): A list of integers representing the Huffman-encoded bytes.
"""
                io_req_decoding = """
Input:
  `compressed` (list of int): A list of integers representing the Huffman-encoded bytes.
Output:
  `return` (str): The original uncompressed string.
"""

                if args.blind:
                    # Level 2+3: obfuscated variable names, parameter renamed to x
                    encoding_fn = f"""
def main_solution(x):
    from collections import namedtuple
    import heapq
    p = {freq}
    N = namedtuple('N', ['f','s','l','r'])
    N.__lt__ = lambda a,b: a.f < b.f
    h = [N(v,k,None,None) for k,v in p.items()]
    heapq.heapify(h)
    while len(h) > 1:
        l = heapq.heappop(h)
        r = heapq.heappop(h)
        heapq.heappush(h, N(l.f+r.f, None, l, r))
    cb = {{}}
    def wk(nd, pre):
        if nd.s is not None:
            cb[nd.s] = pre or '0'
        else:
            wk(nd.l, pre+'0')
            wk(nd.r, pre+'1')
    wk(h[0], '')
    b = ''.join(cb[c] for c in x)
    pad = (-len(b)) % 8
    b += '0'*pad
    return [int(b[i:i+8],2) for i in range(0,len(b),8)]
"""
                    decoding_fn = f"""
def main_solution(x):
    from collections import namedtuple
    import heapq
    pad = {padding}
    p = {freq}
    N = namedtuple('N', ['f','s','l','r'])
    N.__lt__ = lambda a,b: a.f < b.f
    h = [N(v,k,None,None) for k,v in p.items()]
    heapq.heapify(h)
    while len(h) > 1:
        l = heapq.heappop(h)
        r = heapq.heappop(h)
        heapq.heappush(h, N(l.f+r.f, None, l, r))
    root = h[0]
    b = ''.join(map(lambda v: '{{:08b}}'.format(v), x))
    b = b[:-pad] if pad else b
    r = []
    nd = root
    for bit in b:
        nd = nd.l if bit=='0' else nd.r
        if nd.s is not None:
            r.append(nd.s)
            nd = root
    return ''.join(r)
"""
                    io_req_encoding = (
                        "Input:\n  `x` (str): The input string.\n\n"
                        "Output:\n  `return` (list of int): A list of integers representing the encoded form."
                    )
                    io_req_decoding = (
                        "Input:\n  `x` (list of int): A list of integers representing the encoded form.\n\n"
                        "Output:\n  `return` (str): The original string."
                    )

            else:
                raise ValueError(f"Unsupported algorithm: {args.algorithm}")

            for n_task, task in enumerate(["output_execution_prediction", "input_execution_prediction",
                        #  "output_reconstruction_prediction", "input_reconstruction_prediction",
                         "output_execution_prediction_with_inversion", "input_execution_prediction_with_inversion"]):

                # Fix: skip samples already written so interrupted runs can resume.
                sample_id = f"{iid}_{ioid}_{n_task}"
                if sample_id in done_ids:
                    continue

                # Select correct function and IO description based on the task.
                #
                # Paper definitions:
                #   output_execution_prediction          : Given (x, enc)  → predict z        [forward]
                #   input_execution_prediction           : Given (z, dec)  → predict x        [forward: run decoder directly]
                #   output_execution_prediction_with_inversion: Given (x, dec) → predict z   [backward: dec⁻¹≡enc]
                #   input_execution_prediction_with_inversion : Given (z, enc) → predict x   [backward: enc⁻¹≡dec]
                #
                # The _with_inversion tasks deliberately give the model the OPPOSITE function so
                # it must reason about the inverse rather than directly simulating the function.
                if task == "output_execution_prediction":
                    refcode = encoding_fn
                    io_req = io_req_encoding
                elif task == "input_execution_prediction":
                    refcode = decoding_fn   # model given dec; runs dec(z)=x directly
                    io_req = io_req_decoding
                elif task == "output_execution_prediction_with_inversion":
                    refcode = decoding_fn   # model given dec; must apply dec⁻¹(≡enc) to predict z
                    io_req = io_req_encoding  # describes the question: input=x, output=z
                elif task == "input_execution_prediction_with_inversion":
                    refcode = encoding_fn   # model given enc; must apply enc⁻¹(≡dec) to predict x
                    io_req = io_req_decoding  # describes the question: input=z, output=x
                else:
                    raise ValueError(f"Unknown task type: {task}")

                # input_execution_prediction: decoder takes z (output_xx) as input and
                # produces x (input_xx) as output.  Use the OUTPUT template with swapped
                # args so the prompt reads "Given input z, run decoder, predict output x".
                # All other input_ tasks (with_inversion) correctly use the INPUT template
                # because they show z as the *encoder's* output and ask for the input x.
                is_input_task = task.startswith("input_")
                return_type_hint = (
                    _INPUT_RETURN_TYPE_HINT
                    if is_input_task
                    else _OUTPUT_RETURN_TYPE_HINTS[args.algorithm]
                )
                if task == "input_execution_prediction":
                    prompt = build_io_pred(problem_description, io_req, refcode, output_xx, input_xx, prompt_type=args.prompt_type,
                                           io="o", inversion=False, return_type_hint=return_type_hint)
                else:
                    prompt = build_io_pred(problem_description, io_req, refcode, input_xx, output_xx, prompt_type=args.prompt_type,
                                           io="o" if task.startswith("output_") else "i", inversion=False if "inversion" not in task else True,
                                           return_type_hint=return_type_hint)

                msg = build_messages(prompt=prompt, system_message="You are a helpful programming assistant designed to execute code. You must verify your own output via a round\u2011trip check and self\u2011correct before returning the final JSON.")

                # Build sample dictionary
                sample = {
                    "problem_description": problem_description,
                    "io_requirements": io_req,
                    "messages": msg,
                    "itemid": iid,
                    "ioid": ioid,
                    "io_pred": task,
                    "category": category,
                    "algorithm": algorithm,
                    "refcode": refcode,
                    "input": input_xx,
                    "output": output_xx,
                    "id": sample_id,
                }

                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                out_f.flush()

    out_f.close()
