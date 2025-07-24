from utils import *
from tqdm import tqdm
# from codeio_utils import *
from prompts import input_exec_pred_template, output_exec_pred_template, input_exec_pred_template_inversion, output_exec_pred_template_inversion
from utils import get_freq_dict


def build_io_pred(problem_statement, io_req, refcode, inputx, outputx, prompt_type, io = "i", w_refcode=True, inversion=False):

    if prompt_type == "zero_shot":
        if inversion:
            template = input_exec_pred_template_inversion if io=="i" else output_exec_pred_template_inversion
        else:
            template = input_exec_pred_template if io=="i" else output_exec_pred_template

    # elif prompt_type == "one_shot":
    #     template = input_exec_pred_template + "\n\n" + few_shot_template_input if io=="i" else output_exec_pred_template + "\n\n" + few_shot_template_output

    prompt = template.replace("<<<<query>>>>", problem_statement).replace("<<<<io_req>>>>", io_req).replace("<<<<refcode>>>>", refcode)
    tag = "<<<<output>>>>" if io=="i" else "<<<<input>>>>"
    inputxx = f"{inputx}"
    outputxx = f"{outputx}"
    prompt = prompt.replace(tag, outputxx if io=="i" else inputxx)
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
    parser.add_argument('--prompt_type', type=str, default="zero_shot", choices=["zero_shot", "one_shot"])
    args = parser.parse_args()

    fn = args.input_file
    ofn = args.output_file
    adt = []
    dt = load_jsonl_yield(fn)
    for iid,item in enumerate(tqdm(dt)):
        # print(item)
        problem_description = item['problem_description']
        io_req = item['io_requirements']
        algorithm = item['algorithm']
        # refcode = item['refcode']
        for ioid,io in enumerate(item['ios']):
            # uplimit = 3
            # if ioid>=uplimit:break # we now first only use the first 3 io
            input_xx = io['input']
            output_xx = io['output']
            category = io['category']

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

                io_req_decoding_inversion = (
                    "Input:\n  `uncompressed` (str): The original input string before compression.\n\n"
                    "Output:\n  `return` (list of int): A list of integers representing "
                    "the compressed form of the input string using LZW encoding."
                )

            elif args.algorithm == "ae":
                freq = get_freq_dict(ioid=ioid, algo="ae")  # you must provide ioid externally
                encoding_fn = (
                    "def main_solution(uncompressed):\n"
                    f"    freq = {freq}\n"
                    "    total = sum(freq.values())\n"
                    "    symbols = sorted(freq.keys())\n"
                    "    cum_counts = {}\n"
                    "    running = 0\n"
                    "    for sym in symbols:\n"
                    "        cum_counts[sym] = running\n"
                    "        running += freq[sym]\n"
                    "    low, high = 0.0, 1.0\n"
                    "    for c in list(uncompressed) + ['EOF']:\n"
                    "        width = high - low\n"
                    "        high = low + width * (cum_counts[c] + freq[c]) / total\n"
                    "        low = low + width * cum_counts[c] / total\n"
                    "    return (low + high) / 2"
                )

                decoding_fn = (
                    "def main_solution(compressed):\n"
                    f"    freq = {freq}\n"
                    "    total = sum(freq.values())\n"
                    "    symbols = sorted(freq.keys())\n"
                    "    cum_counts = {}\n"
                    "    running = 0\n"
                    "    for s in symbols:\n"
                    "        cum_counts[s] = running\n"
                    "        running += freq[s]\n"
                    "    low, high = 0.0, 1.0\n"
                    "    result = []\n"
                    "    while True:\n"
                    "        width = high - low\n"
                    "        scaled = (compressed - low) / width * total\n"
                    "        for s in symbols:\n"
                    "            if cum_counts[s] <= scaled < cum_counts[s] + freq[s]:\n"
                    "                symbol = s\n"
                    "                break\n"
                    "        if symbol == 'EOF':\n"
                    "            break\n"
                    "        result.append(symbol)\n"
                    "        high = low + width * (cum_counts[symbol] + freq[symbol]) / total\n"
                    "        low = low + width * cum_counts[symbol] / total\n"
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

                io_req_decoding_inversion = (
                    "Input:\n  `uncompressed` (str): The original input string before compression.\n\n"
                    "Output:\n  `return` (float): A probability value representing the "
                    "compressed form of the input string using Arithmetic Encoding."
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

                io_req_decoding_inversion = (
                    "Input:\n  `uncompressed` (str): The original uncompressed string.\n\n"
                    "Output:\n  `return` (list of tuple): A list of (char, count) tuples from the RLE compression."
                )

            elif args.algorithm == "huffman":
                # retrieve freq dict from metadata
                freq = get_freq_dict(ioid=ioid, algo="ae")
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
    return [int(bitstr[i:i+8],2) for i in range(0,len(bitstr),8)], codebook, padding
"""

                decoding_fn = f"""
def main_solution(compressed):
    from collections import namedtuple
    import heapq
    codebook = {codebook}
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
  `return` (list,int,int): A tuple of (encoded_bytes, codebook, padding).
"""
                io_req_decoding = """
Input:
  `compressed` (tuple): The tuple (encoded_bytes, codebook, padding).
Output:
  `return` (str): The original uncompressed string.
"""
                io_req_decoding_inversion = (
                    "Input:\n  `uncompressed` (str): The original uncompressed string.\n\n"
                    "Output:\n  `return` (tuple): The tuple (encoded_bytes, codebook, padding)."
                )

            else:
                raise ValueError(f"Unsupported algorithm: {args.algorithm}")

            for n_task, task in enumerate(["output_execution_prediction", "input_execution_prediction", 
                        #  "output_reconstruction_prediction", "input_reconstruction_prediction", 
                         "output_execution_prediction_with_inversion", "input_execution_prediction_with_inversion"]):

                # Select correct function and IO description based on the task
                if task in ["output_execution_prediction", "input_execution_prediction", "input_execution_prediction_with_inversion"]:
                    refcode = encoding_fn
                    io_req = io_req_encoding
                elif task in ["output_execution_prediction_with_inversion"]:
                    refcode = decoding_fn
                    io_req = io_req_decoding_inversion
                else:
                    raise ValueError(f"Unknown task type: {args.task}")
            
                prompt = build_io_pred(problem_description, io_req, refcode, input_xx, output_xx, prompt_type=args.prompt_type,
                                       io="o" if task.startswith("output_") else "i", inversion=False if "inversion" not in task else True)
                                       
                msg = build_messages(prompt=prompt, system_message="You are a helpful programming assistant designed to execute code. You must verify your own output via a round‑trip check and self‑correct before returning the final JSON.")

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
                    "id": f"{iid}_{ioid}_{n_task}"
                }

                adt.append(sample)

    write_jsonl(adt,ofn)