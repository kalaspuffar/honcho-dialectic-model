#!/usr/bin/env python3
"""
write_chosen_batch.py (stage 3, batch edition)
----------------------------------------------
Teacher writes the IDEAL terse (CHOSEN) answer per context — via Anthropic
Message Batches (50% rate, your Claude account, async) instead of one
synchronous call at a time. Same briefing prompt + payload shape as the
verified batch path in openrouter_trial.py (single style source).

Why batches: 3000 contexts at ~1-5 min of sync calls each is slow and
uninterruptible; Anthropic batches finish in < 1h and cost half (2.5/12.5
$/MTok for opus-5). No third party, no OpenRouter.

Keys: reads ANTHROPIC_API_KEY from the environment OR from keys.env next to
this script (see keys.env.example). Never prints key values.

No cost cap: the pre-spend estimate is printed for reference but NEVER aborts
the run (Daniel: a cap can interrupt a run he wants to complete).

Usage:
    # 1) submit the first 200-context batch
    python3 write_chosen_batch.py submit --contexts workspace/contexts_200.jsonl --arm opus

    # 2) check progress any time
    python3 write_chosen_batch.py status

    # 3) fetch + score once done (polls until 'ended' by default)
    python3 write_chosen_batch.py fetch                # newest manifest
    #    or pin one: --manifest results/chosen/batches/batch_opus_*.json
    #    output lands at the path recorded in the manifest (default next to contexts)

Output row format (matches write_chosen.py, mergeable with it):
    {"id": "...", "category": "...", "answer": "...", "teacher": "opus", "words": N}
Failed rows keep the "__FAILED__: ..." marker so you can filter/retry them.
"""
import argparse, datetime, json, os, statistics, time, urllib.error, urllib.request

from openrouter_trial import (
    ANTHROPIC, ANTHROPIC_ARMS, ANTHROPIC_BATCH_PRICE,
    BATCH_ANSWER_SYSTEM, BRIEFING, MAX_TOKENS_OUT,
    anth_get, anth_headers, batch_estimate, extract_json, load_anth_key,
    row_block, tokens_estimate,
)

R = "results/chosen"


def batches_dir():
    d = os.path.join(R, "batches")
    os.makedirs(d, exist_ok=True)
    return d


def default_out(contexts_file: str) -> str:
    base = os.path.basename(contexts_file)
    name = base.replace("contexts", "chosen").replace("context", "chosen")
    return os.path.join(os.path.dirname(os.path.abspath(contexts_file)), name)


def load_ctxs(contexts_file: str):
    ctxs = [json.loads(l) for l in open(contexts_file) if l.strip()]
    for required in ("id", "persona", "question", "findings"):
        missing = [c.get("id", "?") for c in ctxs if required not in c]
        if missing:
            raise SystemExit(f"contexts file missing '{required}' for ids: {missing[:5]}…")
    ids = [c["id"] for c in ctxs]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})[:10]
        raise SystemExit(f"duplicate context ids: {dup}…")
    return ctxs


def cmd_estimate(a):
    ctxs = load_ctxs(a.contexts)
    worst, best, tin, tout = batch_estimate(a.arm, ctxs)
    print(f"arm          {a.arm:9s} ({ANTHROPIC_ARMS[a.arm]})  {len(ctxs)} rows")
    print(f"est input    {tin/1e6:.2f} MTok (worst, no cache)")
    print(f"est output   {tout/1e6:.2f} MTok")
    print(f"estimate     ~${best:.3f} (with prompt cache) – ${worst:.3f} (no cache)")
    return 0


def cmd_submit(a):
    ctxs = load_ctxs(a.contexts)
    model = ANTHROPIC_ARMS[a.arm]
    worst, best, tin, tout = batch_estimate(a.arm, ctxs)
    print("pre-spend estimate (Anthropic batch 50% rate, NO network yet):")
    print(f"  {a.arm:9s} {model:20s} {len(ctxs):>5d} rows  ~${best:.4f} (w/ cache) – ${worst:.4f} (no cache)  [reference only — no cap]")

    key = load_anth_key()
    if not key:
        print("ANTHROPIC_API_KEY not set (env or keys.env). Aborting — nothing sent.")
        print("Copy keys.env.example to keys.env, add your Claude account key (the one the deriver uses).")
        return 1

    shared = BATCH_ANSWER_SYSTEM + "\n\n" + BRIEFING
    payload = {
        "requests": [
            {
                "custom_id": f"{a.arm}_{c['id']}",  # must match ^[a-zA-Z0-9_-]{1,64}$
                "params": {
                    "model": model,
                    "max_tokens": MAX_TOKENS_OUT,
                    # Shared prefix across all rows -> prompt-cache hit (stacks with batch 50%).
                    "system": [{"type": "text", "text": shared,
                                "cache_control": {"type": "ephemeral"}}],
                    "messages": [{"role": "user", "content": row_block(c)}],
                },
            }
            for c in ctxs
        ]
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        ANTHROPIC + "/v1/messages/batches", data=body, headers=anth_headers(key), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            b = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"  SUBMIT FAILED HTTP {e.code}: {e.read()[:300].decode(errors='replace')}")
        return 1
    except Exception as e:
        print(f"  SUBMIT FAILED {type(e).__name__}: {e}")
        return 1

    out = a.out or default_out(a.contexts)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mani = {
        "arm": a.arm, "model": model, "n": len(ctxs),
        "est_usd_low": round(best, 5), "est_usd_high": round(worst, 5),
        "batch_id": b["id"], "processing_status": b.get("processing_status"),
        "submitted_at": datetime.datetime.now().isoformat(),
        "expires_at": b.get("expires_at"),
        "source_contexts": os.path.abspath(a.contexts),
        "out": os.path.abspath(out),
        "poll_url": f"{ANTHROPIC}/v1/messages/batches/{b['id']}",
        "base_estimate_tokens_in": tin, "base_estimate_tokens_out": tout,
    }
    mp = os.path.join(batches_dir(), f"batch_{a.arm}_{stamp}.json")
    open(mp, "w").write(json.dumps(mani, indent=2))
    print(f"submitted batch {b['id']} ({len(ctxs)} requests)")
    print(f"manifest: {mp}")
    print(f"output (on fetch): {mani['out']}")
    print("\nBatch processing is async (most finish in < 1h, 24h max). When ready:\n"
          "  python3 write_chosen_batch.py fetch          # polls until ended, writes + scores\n"
          "  python3 write_chosen_batch.py status         # peek at progress")
    return 0


def cmd_status(a):
    key = load_anth_key()
    if not key:
        print("ANTHROPIC_API_KEY not set (env or keys.env) — cannot query status.")
        return 1
    bd = batches_dir()
    if not os.path.isdir(bd) or not [f for f in os.listdir(bd) if f.endswith(".json")]:
        print(f"no batch manifests under {bd} — run `submit` first.")
        return 1
    rc = 0
    for f in sorted(os.listdir(bd)):
        if not f.endswith(".json"):
            continue
        m = json.load(open(os.path.join(bd, f)))
        try:
            b = json.loads(anth_get(m["poll_url"], key))
            st = b.get("processing_status")
            counts = b.get("request_counts", {})
        except Exception as e:
            print(f"{f}: status check failed ({type(e).__name__}: {e})")
            rc = 1
            continue
        print(f"{f}: arm={m['arm']} n={m['n']} status={st} {counts}")
        if st != "ended":
            rc = 1  # signal: not everything done yet
    return rc


def fetch_results(m: dict, out: str) -> dict:
    key = load_anth_key()
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set (env or keys.env).")
    arm, model = m["arm"], m["model"]
    done_url = f"{ANTHROPIC.rstrip('/')}/v1/messages/batches/{m['batch_id']}/results"
    raw_data = anth_get(done_url, key, timeout=300)

    # Resume: keep already-good rows in the target file, refill only missing/failed.
    existing = {}
    if os.path.exists(out):
        for line in open(out):
            if line.strip():
                rec = json.loads(line)
                if not (rec.get("answer") or "").startswith("__FAILED__"):
                    existing[rec["id"]] = rec

    recs, n_err = {}, 0
    for line in (s.decode("utf-8", errors="replace") for s in raw_data.split(b"\n")):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            n_err += 1
            continue
        cid = obj.get("custom_id", "")
        ctx_id = cid.split("_", 1)[1] if "_" in cid else cid
        res = obj.get("result", {})
        ans, err = "", ""
        if res.get("type") == "succeeded":
            msg = res.get("message", {})
            text = ""
            for blk in (msg.get("content") or []):
                if isinstance(blk, dict) and blk.get("type") == "text":
                    text += blk.get("text", "")
            if not text and isinstance(msg.get("content"), str):
                text = msg["content"]
            j = extract_json(text)
            ans = (j or {}).get("answer") if isinstance(j, dict) else None
            ans = ans if isinstance(ans, str) and ans.strip() else (text.strip() if text else "")
            if not ans:
                err = "no answer text in batch result"
        else:
            err = f"batch result type={res.get('type')}: " \
                  f"{json.dumps(res.get('error', {}))[:300]}"
            n_err += 1
        recs[ctx_id] = {"ok": bool(ans), "answer": ans, "error": err,
                        "usage": (obj.get("result", {}).get("message", {}) or {}).get("usage")}

    # Merge with resume state, preserving original context order when possible.
    src = m.get("source_contexts", "")
    order = [json.loads(l)["id"] for l in open(src) if l.strip()] if os.path.exists(src) else list(recs)
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)
    lines, fresh = [], 0
    for cid in order:
        if cid in existing:
            lines.append(existing[cid])
            continue
        rc = recs.get(cid)
        ans = rc["answer"] if rc else ""
        rec = {
            "id": cid,
            "category": (existing.get(cid) or {}).get("category", ""),
            "answer": ans or (f"__FAILED__: {rc['error']}" if rc and rc.get("error")
                              else "__FAILED__: no result for this context id"),
            "teacher": arm,
            "words": len(ans.split()) if ans else 0,
        }
        lines.append(rec)
        fresh += 1
    with open(out, "w") as f:
        for rec in lines:
            f.write(json.dumps(rec) + "\n")
    failed = sum(1 for rec in lines if rec["answer"].startswith("__FAILED__"))
    words = [len(r["answer"].split()) for r in lines if not r["answer"].startswith("__FAILED__")]
    med = int(statistics.median(words)) if words else None
    inm, outm = ANTHROPIC_BATCH_PRICE.get(model, (0, 0))
    print(f"wrote {out}: {len(lines)} rows ({len(existing)} resumed, {fresh} fetched, {failed} failed)")
    if med is not None:
        print(f"median answer length: {med} words (target <= 50)")
    usage_note = ("actual cost: see usage in API console; usage-based estimate "
                  f"(upper bound, cached tokens billed at 10%): "
                  f"${tin_cost(m, recs):.3f}")
    print(usage_note)
    if failed:
        print(f"{failed} row(s) failed — re-run `submit` for the failed ids or re-run `fetch` "
              "after canceling/resubmitting; failed rows keep their __FAILED__ marker.")
    return {"n": len(lines), "failed": failed, "median_words": med}


def tin_cost(m: dict, recs: dict) -> float:
    """Rough upper-bound cost from per-line usage if present, else the estimate."""
    try:
        inm, outm = ANTHROPIC_BATCH_PRICE[m["model"]]
        tot = 0.0
        for r in recs.values():
            u = r.get("usage") or {}
            tot += u.get("input_tokens", 0) * inm + u.get("output_tokens", 0) * outm
        if tot:
            return tot / 1e6
    except Exception:
        pass
    return m.get("est_usd_high", 0.0)


def cmd_fetch(a):
    key = load_anth_key()
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set (env or keys.env).")
    bd = batches_dir()
    if a.manifest:
        manifests = [a.manifest]
    else:
        if not os.path.isdir(bd) or not [f for f in os.listdir(bd) if f.endswith(".json")]:
            print(f"no batch manifests under {bd} — run `submit` first.")
            return 1
        manifests = sorted(os.path.join(bd, f) for f in os.listdir(bd) if f.endswith(".json"))
        manifests = [manifests[-1]]  # newest
    m = json.load(open(manifests[0]))
    print(f"batch {m['batch_id']}  arm={m['arm']}  n={m['n']}  manifest={manifests[0]}")

    out = a.out or m.get("out") or default_out(m.get("source_contexts", "contexts.jsonl"))
    if not a.no_wait:
        while True:
            try:
                b = json.loads(anth_get(m["poll_url"], key))
            except Exception as e:
                print(f"  status poll error ({type(e).__name__}: {e}); retrying in {a.poll_interval}s")
                time.sleep(a.poll_interval)
                continue
            st, rc = b.get("processing_status"), b.get("request_counts", {})
            print(f"  status={st} {rc}", flush=True)
            if st == "ended":
                break
            if st in ("canceled", "cancelled"):
                print("batch was canceled — no results.")
                return 1
            time.sleep(a.poll_interval)
    else:
        b = json.loads(anth_get(m["poll_url"], key))
        if b.get("processing_status") != "ended":
            print(f"still {b.get('processing_status')}; run `fetch` again without --no-wait when it ends.")
            return 1

    fetch_results(m, out)
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Write the CHOSEN (teacher) answers via Anthropic Message Batches "
                    "(50% rate, reads keys.env). Subcommands: estimate / submit / status / fetch.")
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("submit", help="estimate cost, then submit a batch (requires keys.env)")
    s.add_argument("--contexts", required=True, help="JSONL context file (workspace/contexts_200.jsonl etc.)")
    s.add_argument("--arm", default="opus", choices=sorted(ANTHROPIC_ARMS))
    s.add_argument("--out", default=None, help="output JSONL path (default: chosen_<split>.jsonl next to input)")
    s.set_defaults(fn=cmd_submit)

    e = sub.add_parser("estimate", help="pure-local cost estimate, no network, no key needed")
    e.add_argument("--contexts", required=True)
    e.add_argument("--arm", default="opus", choices=sorted(ANTHROPIC_ARMS))
    e.set_defaults(fn=cmd_estimate)

    t = sub.add_parser("status", help="show processing status of all manifests")
    t.set_defaults(fn=cmd_status)

    f = sub.add_parser("fetch", help="poll until ended (default), then write + score")
    f.add_argument("--manifest", help="specific batch_*.json manifest (default: newest)")
    f.add_argument("--out", help="override output path recorded in the manifest")
    f.add_argument("--poll-interval", type=int, default=60)
    f.add_argument("--no-wait", action="store_true", help="check once; fail if not ended yet")
    f.set_defaults(fn=cmd_fetch)

    a = ap.parse_args()
    if not getattr(a, "fn", None):
        ap.print_help()
        return 1
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
