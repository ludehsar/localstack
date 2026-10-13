#!/usr/bin/env python3
"""Score a chat model on coding problems with unit tests (HumanEval+ or MBPP+ from EvalPlus).

Each problem goes to an OpenAI-compatible /v1/chat/completions endpoint, so the same run works
against `lstk --serve` and against hosted flagship models (OpenRouter, OpenAI, or Anthropic's
OpenAI-compatible endpoint). The code in the reply is run against the problem's tests, and the
score is pass@1 with greedy decoding (temperature 0). Results are appended to a JSONL file, so an
interrupted run resumes where it stopped.

    ./build/lstk --model ~/Models/gguf/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf \\
        --stream-experts --pool-gib 2.5 --serve &
    python3 tools/code_eval.py --dataset humaneval

    OPENROUTER_API_KEY=... python3 tools/code_eval.py --dataset humaneval \\
        --base-url https://openrouter.ai/api/v1 --model moonshotai/kimi-k3 --api-key-env OPENROUTER_API_KEY

    python3 tools/code_eval.py --dataset humaneval --model canonical   # self-check: must score 100%
    python3 tools/code_eval.py --compare build/evals/humaneval-*.jsonl   # side-by-side summary

Model-written code runs on this machine, so each program runs in a fresh temporary directory under
macOS sandbox-exec (no network, no file writes outside that directory) with a timeout.
"""
import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "evals")
DATASETS = {
    "humaneval": ("https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/"
                  "HumanEvalPlus-OriginFmt.jsonl.gz", "he.jsonl.gz"),
    "mbpp": ("https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/"
             "MbppPlus-OriginFmt.jsonl.gz", "mbpp.jsonl.gz"),
}
# Tasks whose test fails even for the reference solution. HumanEval/32's OriginFmt test unpacks the
# function's float result with `*`, so no answer can pass it. MBPP+ 590 expects cmath output whose last
# digit differs on macOS, so the reference solution fails it here.
BROKEN = {"HumanEval/32", "590"}
SANDBOX = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{tmp}") (literal "/dev/null"))
"""


def load(name):
    url, fn = DATASETS[name]
    path = os.path.join(CACHE, fn)
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    tasks = (json.loads(line) for line in gzip.open(path, "rt"))
    return [t for t in tasks if str(t["task_id"]) not in BROKEN]


# Per benchmark: what the model is asked, the reference answer, and the test program built from an answer.
def task_prompt(name, t):
    if name == "humaneval":
        return ("Complete this Python function. Reply with the complete function, including any imports it "
                "needs, in one ```python code block.\n\n```python\n" + t["prompt"] + "```")
    return ("Write a Python function for this task. Reply with the code in one ```python code block.\n\n" +
            t["prompt"] + "\nYour code should pass this test:\n" + t["test_list"][0])


def reference(name, t):
    return t["prompt"] + t["canonical_solution"] if name == "humaneval" else t["code"]


def test_program(name, t, code):
    if name == "humaneval":
        # The prompt's stub (signature + docstring) is valid Python and brings the problem's imports and
        # helpers; the model's definition comes after it and replaces the stub.
        return t["prompt"] + "\n\n" + code + "\n\n" + t["test"] + "\n\ncheck(" + t["entry_point"] + ")\n"
    return code + "\n\n" + t["test"] + "\n"  # MBPP+ tests call the function by name at module level


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = re.findall(r"```(?:python|py)?[^\n]*\n(.*?)```", text, flags=re.S)
    return max(blocks, key=len) if blocks else text


def run_tests(program, timeout):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = os.path.realpath(tmp)
        path = os.path.join(tmp, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        cmd = ["sandbox-exec", "-p", SANDBOX.format(tmp=tmp), sys.executable, "-I", "-B", path]
        try:
            r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout,
                               env={"PATH": "/usr/bin:/bin", "HOME": tmp})
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return r.returncode == 0, r.stderr[-400:]


def chat(args, prompt):
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": args.max_tokens}
    headers = {"Content-Type": "application/json"}
    if args.api_key_env:
        headers["Authorization"] = "Bearer " + os.environ[args.api_key_env]
    req = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions", json.dumps(body).encode(), headers)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=args.request_timeout) as r:
                resp = json.load(r)
            return resp["choices"][0]["message"].get("content") or "", resp.get("usage") or {}
        except urllib.error.HTTPError as e:
            # Rate limits and server errors are worth retrying; anything else (bad key, bad model) is not.
            if e.code not in (429, 500, 502, 503, 529) or attempt == 3:
                sys.exit(f"request failed: HTTP {e.code}: {e.read()[:300]!r}")
            time.sleep(5 * 2 ** attempt)


def summarize(label, records):
    n = len(records)
    passed = sum(r["passed"] for r in records)
    secs = sum(r["gen_s"] for r in records)
    toks = sum(r.get("completion_tokens") or 0 for r in records)
    rate = f", {toks / secs:.1f} tok/s" if toks and secs else ""
    print(f"{label}: pass@1 {passed}/{n} = {100.0 * passed / max(n, 1):.1f}%   generation {secs / 60:.1f} min{rate}")


def compare(paths):
    runs = {p: {r["task_id"]: r for r in map(json.loads, open(p))} for p in paths}
    for p, recs in runs.items():
        summarize(os.path.basename(p), list(recs.values()))
    if len(paths) == 2:
        a, b = (runs[p] for p in paths)
        common = a.keys() & b.keys()
        only_a = sorted(t for t in common if a[t]["passed"] and not b[t]["passed"])
        only_b = sorted(t for t in common if b[t]["passed"] and not a[t]["passed"])
        print(f"on {len(common)} shared tasks: only first passes {len(only_a)}, only second passes {len(only_b)}")
        print("  only first:", " ".join(only_a[:30]))
        print("  only second:", " ".join(only_b[:30]))


def main():
    ap = argparse.ArgumentParser(description="pass@1 of a chat model on HumanEval+ / MBPP+")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="humaneval")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="localstack", help="model id sent to the API; 'canonical' scores the reference solutions")
    ap.add_argument("--api-key-env", help="name of the environment variable holding the API key")
    ap.add_argument("--limit", type=int, default=0, help="only the first N problems")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=30.0, help="seconds allowed for one problem's tests")
    ap.add_argument("--request-timeout", type=float, default=1800.0)
    ap.add_argument("--out", help="results JSONL (default build/evals/<dataset>-<model>.jsonl)")
    ap.add_argument("--compare", nargs="+", metavar="JSONL", help="summarize finished runs instead of running")
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare)
    if args.api_key_env and not os.environ.get(args.api_key_env):
        sys.exit(f"{args.api_key_env} is not set")

    out = args.out or os.path.join(CACHE, f"{args.dataset}-{re.sub(r'[^A-Za-z0-9._-]', '_', args.model)}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    done = {}
    if os.path.exists(out):
        done = {r["task_id"]: r for r in map(json.loads, open(out))}
    tasks = load(args.dataset)
    tasks = tasks[:args.limit] if args.limit else tasks

    with open(out, "a") as f:
        for i, t in enumerate(tasks):
            tid = str(t["task_id"])
            if tid in done:
                continue
            t0 = time.time()
            if args.model == "canonical":
                code, usage = reference(args.dataset, t), {}
            else:
                text, usage = chat(args, task_prompt(args.dataset, t))
                code = extract_code(text)
            gen_s = time.time() - t0
            ok, err = run_tests(test_program(args.dataset, t, code), args.timeout)
            rec = {"task_id": tid, "passed": ok, "gen_s": round(gen_s, 2),
                   "completion_tokens": usage.get("completion_tokens"), "error": "" if ok else err, "code": code}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[tid] = rec
            print(f"[{i + 1}/{len(tasks)}] {tid} {'PASS' if ok else 'FAIL'} ({gen_s:.1f}s)", flush=True)

    wanted = {str(t["task_id"]) for t in tasks}
    summarize(f"{args.dataset} {args.model}", [r for k, r in done.items() if k in wanted])
    print("results:", out)


if __name__ == "__main__":
    main()
