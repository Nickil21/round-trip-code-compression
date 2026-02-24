# ---------- Shared building blocks ----------
_EXAMPLE_PREAMBLE = """You are given a Python function and an input to the function. Complete with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

The input and output requirements are as follows:

Input:
  `s` (str): The input string to be duplicated and wrapped.

Output:
  `return` (str): A string starting with `"b"`, followed by two copies of `s`, and ending with `"a"`.

Given the following input:

"hi"

Given the following function:

[PYTHON]
def main_solution(s):
    s = s + s
    return "b" + s + "a"
[/PYTHON]

Can you predict the output without writing any code? Do not include any explanations, reasoning, or extra text. Put your final answer in the following json format: {"output": <your output>}, where <your output> should strictly match the the output requirement as specified.

[THOUGHT]
Let's execute the code step by step:

1. The function `main_solution` is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]

[ANSWER]
{"output": "bhihia"}
[/ANSWER]
"""

_OUTPUT_TAIL = """
The input and output requirements are as follows:

<<<<io_req>>>>

Given the following input:

"<<<<input>>>>"

Given the following function:

[PYTHON]
<<<<refcode>>>>
[/PYTHON]
"""

_OUTPUT_TAIL_INV_EXTRA = """
The function `main_solution` performs decoding. You must use the inverse logic to implement the encoding function, `main_solution_inverse`, to infer your answer, not run or duplicate it directly.


Can you predict the output based on `main_solution_inverse`? Do not include any explanations, reasoning, or extra text. Put your final answer in the following json format: {"output": <your output>}, where <your output> should strictly match the the output requirement as specified.
The "output" value must be of type: <<<<return_type>>>>.
Return only:
[ANSWER]
{"output": <your output>}
[/ANSWER]
"""

_OUTPUT_TAIL_NORMAL_END = """
Can you predict the output without writing any code? Do not include any explanations, reasoning, or extra text. Put your final answer in the following json format: {"output": <your output>}, where <your output> should strictly match the the output requirement as specified.
The "output" value must be of type: <<<<return_type>>>>.
Return only:
[ANSWER]
{"output": <your output>}
[/ANSWER]
"""

# ---------- Preamble for input-prediction tasks ----------
_EXAMPLE_PREAMBLE_INPUT = """You are given a Python function and an output of the function. Predict a feasible input that produces the given output when executing the provided code, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide a feasible input in [ANSWER] and [/ANSWER] tags, following the examples.

The input and output requirements are as follows:

Input:
  `s` (str): The input string to be duplicated and wrapped.

Output:
  `return` (str): A string starting with `"b"`, followed by two copies of `s`, and ending with `"a"`.

Given the following output:

"bhihia"

Given the following function:

[PYTHON]
def main_solution(s):
    s = s + s
    return "b" + s + "a"
[/PYTHON]

Can you predict a feasible input without writing any code? Do not include any explanations, reasoning, or extra text. Put your final answer in the following json format: {"output": <your output>}, where <your output> should strictly match the input requirement as specified.

[THOUGHT]
Let's work backwards from the output:

1. The function `main_solution` takes a single argument s.
2. It concatenates s with itself, then returns "b" + s + s + "a".
3. The output is "bhihia", so "b" + s + s + "a" = "bhihia".
4. Stripping the leading "b" and trailing "a" gives s + s = "hihi".
5. Since s + s = "hihi", s must be "hi".
[/THOUGHT]

[ANSWER]
{"output": "hi"}
[/ANSWER]
"""

# ----- Input (inverse) templates shared parts -----
_INPUT_TAIL = """
The input and output requirements are as follows:

<<<<io_req>>>>

Given the following output:

<<<<output>>>>

Given the following function:

[PYTHON]
<<<<refcode>>>>
[/PYTHON]
"""

_INPUT_TAIL_INV_EXTRA = """
The function `main_solution` performs encoding. You must use the inverse logic to implement the decoding function, `main_solution_inverse`, to infer your answer, not run or duplicate it directly.

Can you predict a feasible input based on `main_solution_inverse`? Do not include any explanations, reasoning, or extra text. Put your final answer in the following json format: {"output": <your output>}, where <your output> should strictly match the input requirement as specified.
The "output" value must be of type: <<<<return_type>>>>.
Return only:
[ANSWER]
{"output": <your output>}
[/ANSWER]
"""

_INPUT_TAIL_NORMAL_END = """
Can you predict a feasible input without writing any code? Do not include any explanations, reasoning, or extra text. Put your final answer in the following json format: {"output": <your output>}, where <your output> should strictly match the input requirement as specified.
The "output" value must be of type: <<<<return_type>>>>.
Return only:
[ANSWER]
{"output": <your output>}
[/ANSWER]
"""


INPUT_PRED_TEMPLATE = _EXAMPLE_PREAMBLE_INPUT + _INPUT_TAIL + _INPUT_TAIL_NORMAL_END
INPUT_PRED_TEMPLATE_INV_EXTRA = _EXAMPLE_PREAMBLE_INPUT + _INPUT_TAIL + _INPUT_TAIL_INV_EXTRA

OUTPUT_PRED_TEMPLATE = _EXAMPLE_PREAMBLE + _OUTPUT_TAIL + _OUTPUT_TAIL_NORMAL_END
OUTPUT_PRED_TEMPLATE_INV_EXTRA = _EXAMPLE_PREAMBLE + _OUTPUT_TAIL + _OUTPUT_TAIL_INV_EXTRA
