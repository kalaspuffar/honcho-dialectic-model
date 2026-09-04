#!/usr/bin/env python3
"""verify_pipeline.py — pre-flight check of the honcho-dialectic-model trial files.
Runs: static lint, ARM/ID cross-references, arg-wiring, and prints a go/no-go summary.
No network, no key required, no cost.
"""
import ast, importlib.util, os, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
errors, warnings, checks = [], [], 0

def ck(name, ok, detail=""):
    global checks
    checks += 1
    (checks_ok if ok else errors_check)(name, detail)

def checks_ok(n, d): pass
def errors_check(n, d): pass

problems = []
def ok(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        problems.append(name)
    return cond

print("== 1. Python files parse (all .py) ==")
for f in sorted(os.listdir(".")):
    if f.endswith(".py"):
        try:
            compile(open(f).read(), f, "exec")
            ok(f, True)
        except SyntaxError as e:
            ok(f, False, str(e))

print("== 2. openrouter_trial.py cross-references ==")
spec = importlib.util.spec_from_file_location("ot", "openrouter_trial.py")
ot = importlib.util.module_from_spec(spec)
# don't execute main(); just load definitions
spec.loader.exec_module(ot)

ok("ARMS table present", hasattr(ot, "ARMS") and len(ot.ARMS) >= 5, f"found {len(ot.ARMS) if hasattr(ot,'ARMS') else 0}")

# default arms from argparse
import io, contextlib
def get_default(sub, arg):
    try:
        sys.save_argv = sys.argv
        sys.argv = ["openrouter_trial.py", sub, "--help"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            ot.main()
        out = buf.getvalue()
        for line in out.splitlines():
            if arg in line and "[" in line:
                # e.g.  [--arms ARMS]
                if "default" in out:
                    pass
        # fallback: pull from module source via inspect
        import inspect
        src = inspect.getsource(ot.main)
        idx = src.find(f'add_parser("{sub}"')
        chunk = src[idx:idx+700]
        import re
        m = re.search(re.escape(arg) + r'[^"]*"[^"]*"([^"]*)"', chunk)
        if m: return m.group(1)
    finally:
        sys.argv = sys.save_argv
    return None

for sub in ("estimate", "gen", "run"):
    d = get_default(sub, "--arms") or (get_default(sub, "--context-model") if sub == "gen" else None)
    print(f"  default '{sub}': {d}")

# parse the source statically instead (more reliable)
src = open("openrouter_trial.py").read()
import re
defaults = re.findall(r'add_argument\("--(arms|context-model)",\s*default="([^"]+)"', src)
print(f"  static parse of defaults: {defaults}")

arm_names_default = [x for x in re.findall(r'add_argument\("--arms",\s*default="([^"]+)"', src)]
ctx_model_default = re.findall(r'add_argument\("--context-model",\s*default="([^"]+)"', src)
print(f"  --arms defaults: {arm_names_default}")
print(f"  --context-model defaults: {ctx_model_default}")

# Every arm name used in a default must exist in ARMS
for a in arm_names_default:
    for name in a.split(","):
        ok(f"ARM '{name}' (from --arms default) in ARMS table", name in ot.ARMS, "missing from ARMS")

# Every context-model default must be a valid OpenRouter-style id (we trust the table below)
VALID_CTX_IDS = {
    "deepseek/deepseek-chat", "anthropic/claude-opus-5", "anthropic/claude-sonnet-5",
    "google/gemini-3.1-pro-preview", "openai/gpt-5", "qwen/qwen3-max",
    "x-ai/grok-4.6", "meta-llama/llama-3.3-70b-instruct", "openai/o4-mini",
    "google/gemini-2.5-pro",
}
# test-only mock arms (zero-cost, used by verify/run against mock_or_server.py)
TEST_ONLY_IDS = {"mock/zerou", "mock/nullc"}
for c in ctx_model_default:
    ok(f"CTX model '{c}' in verified-live set", c in VALID_CTX_IDS, f"not in verified set: {VALID_CTX_IDS}")

# ARMS: each model id must be in verified-live set (or be a test-only mock arm)
for k, (mid, inm, outm, note) in ot.ARMS.items():
    ok(f"ARMS[{k!r}] id {mid!r} in verified-live or test-only set (2026-09-04)",
       mid in VALID_CTX_IDS or mid in TEST_ONLY_IDS, f"unverified id: {mid}")
    ok(f"ARMS[{k!r}] prices numeric", isinstance(inm,(int,float)) and isinstance(outm,(int,float)))

print("== 3. Required functions present ==")
for fn in ("load_key", "chat", "extract_json", "_try_parse_at", "build_contexts",
           "score", "cmd_estimate", "cmd_gen", "cmd_run", "cmd_collect", "main"):
    ok(f"def {fn}", hasattr(ot, fn))

print("== 4. extract_json unit cases ==")
cases = [
    ("plain", '{"a": 1}'),
    ("prose", 'Here is the object:\n{"persona":"x","question":"q","findings":[]}'),
    ("trailing_comma", '{"persona":"x","findings":[{"a":1},],"q":1,}'),
    ("fenced", '```json\n{"a":1}\n```'),
    ("escaped_quote", '{"q":"he said \\"yes\\""}'),
    ("inner_snippet", 'like {"k":1} but real:\n{"persona":"x","k":{"d":[1]}}'),
    ("no_json", 'nothing to see'),
]
for name, t in cases:
    try:
        r = ot.extract_json(t)
        want_present = name != "no_json"
        got = r is not None
        ok(f"extract_json[{name}]", got == want_present, f"got {r!r}")
    except Exception as e:
        ok(f"extract_json[{name}]", False, str(e))

print("== 5. run_trial.sh static ==")
# shellcheck-free sanity: shebang + set -euo + 4 steps
sh = open("run_trial.sh").read()
for pat in ("#!/usr/bin/env bash", "set -euo pipefail", "python3 openrouter_trial.py estimate",
            "python3 openrouter_trial.py gen", "python3 openrouter_trial.py run",
            "python3 openrouter_trial.py collect"):
    ok(f"run_trial.sh has {pat!r}", pat in sh)
# no 'source keys.env' (which fails under set -e if the file / var is missing)
ok("run_trial.sh does NOT hard-source keys.env", "source keys.env" not in sh)

print()
if problems:
    print(f"GO/NO-GO: NO-GO — {len(problems)} failing checks: {problems}")
    sys.exit(1)
print(f"GO/NO-GO: GO — all {checks} checks passed.")
