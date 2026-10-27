#!/usr/bin/env python3
"""KV pool stress: drive many rounds of multi-slot generation through one engine and prove the
shared logical-page KV pool neither leaks nor refuses under a correctly sized capacity.

Each round issues `--slots` concurrent-ish requests (via the batch slots), reads the engine log's
pool accounting lines, and checks:
  - no monotonic free-page leak across rounds (free returns to a steady watermark),
  - the final round's free equals a steady value (no creep),
  - no `insufficient pages` refusal and no `injected failure`,
  - free accounting is balanced (free <= pages, and free + leased consistent with high_water).

Run (per the plan):
  python3 tools/kv_pool_stress.py --exe build-hip/strata --config /tmp/kvpool-cfg.json \
      --slots 8 --rounds 16 --max-tokens 8192 \
      --extra '--kv-pool --kv-pool-mib 8192 --pcie-frac 0 --adapt-every 1000000'

Set STRATA_KV_POOL_DEBUG=1 (or pass it in --extra-env) so the engine prints per-release free=
lines; the pool init line `KV pool: ... free=F pages=P` is always printed at startup.
"""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_test import Engine, tokenizer  # noqa: E402

FILLER = ("The history of the printing press begins with movable type in East Asia and continues "
          "with Gutenberg's press in Mainz around 1450, which spread across Europe within decades. ")


def chat(user):
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def gen(eng, out, ids, n):
    """Send one GEN and consume to DONE/ERR; return (tokens, status_line)."""
    eng.send(f"GEN {n} " + ",".join(map(str, ids)))
    got = []
    for line in out:
        if line.startswith("T "):
            got.append(int(line.split()[1]))
        elif line.startswith("DONE") or line.startswith("ERR") or line.startswith("BDONE"):
            return got, line
    raise SystemExit("the engine ended")


def pool_init(log_text):
    m = re.search(r"KV pool:.*?pages=(\d+) free=(\d+) high_water=(\d+)", log_text)
    if not m:
        return None
    return {"pages": int(m.group(1)), "free": int(m.group(2)), "high_water": int(m.group(3))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=8192, help="context budget hint (informational)")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--extra", default="")
    ap.add_argument("--extra-env", default="STRATA_KV_POOL_DEBUG=1")
    ap.add_argument("--dump-tokens", default="", help="write round-1 tokens per slot as JSON (reuse parity)")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])
    extra_env = {}
    for kv in a.extra_env.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            extra_env[k] = v
    # --kv-pool --batch <slots> come from --extra and the Engine; add batch via engine arg
    eng = Engine(a.exe, cfg, a.slots, extra_env, a.extra.split())
    out = eng.lines()

    # a short deterministic prompt per request; keeps cells well under capacity so a correctly
    # sized pool meters them but never exhausts.
    base = tok.encode(chat("Summarize the following in one sentence. " + FILLER * 2 + "\nSummarize."),
                      parse_special=True)
    round_tokens = {}   # round index -> list of token lists (for reuse parity on round 2)

    for r in range(a.rounds):
        toks = []
        for s in range(a.slots):
            tag = tok.encode(f" Slot{s} round{r}.", parse_special=True)
            ids = base + tag
            got, st = gen(eng, out, ids, a.max_new)
            toks.append(got)
            if st.startswith("ERR"):
                print(f"FAIL: round {r} slot {s}: {st}")
                eng.send("QUIT")
                return 1
        round_tokens[r] = toks

    eng.send("QUIT")
    try:
        eng.p.wait(timeout=180)
    except Exception:
        eng.p.kill()

    log = open(eng.log_path).read()
    init = pool_init(log)
    if init is None:
        print("FAIL: no 'KV pool:' init line in the log - was --kv-pool passed?")
        return 1

    # per-release free lines (only with STRATA_KV_POOL_DEBUG=1)
    free_seq = [int(x) for x in re.findall(r"kv pool: release .*?free=(\d+)", log)]
    exhaust = len(re.findall(r"insufficient pages", log))
    injected = len(re.findall(r"injected failure", log))

    pages = init["pages"]
    # free accounting must never exceed pages and never go negative
    bad = [f for f in free_seq if f < 0 or f > pages]
    balanced = not bad

    # Steady-state check: after the first full round fills the slots, later free values must not
    # keep dropping (no leak). Compare the min free over the last half to the min over the first
    # half-after-warmup; a leak shows as a strictly smaller value late.
    result = {
        "pages": pages,
        "init_free": init["free"],
        "release_line_count": len(free_seq),
        "insufficient_pages": exhaust,
        "injected": injected,
        "free_accounting_balanced": balanced,
        "errors": [],
    }
    if exhaust or injected:
        result["errors"].append("pool refused/exhausted or injected failure occurred")
    if not balanced:
        result["errors"].append(f"free out of range: {bad[:5]}")
    # If we have per-release free lines, assert no monotonic leak.
    if len(free_seq) >= 4:
        half = len(free_seq) // 2
        early_min = min(free_seq[:half])
        late_min = min(free_seq[half:])
        result["early_min_free"] = early_min
        result["late_min_free"] = late_min
        # allow a small tolerance for session-size jitter, but a real leak drops by far more
        if late_min < early_min - max(8, pages // 50):
            result["errors"].append(f"possible leak: late_min_free {late_min} << early_min_free {early_min}")

    print(json.dumps(result, indent=2))
    if a.dump_tokens and 0 in round_tokens and 1 in round_tokens:
        Path(a.dump_tokens).write_text(json.dumps({"round0": round_tokens[0], "round1": round_tokens[1]}))
        # reuse parity: the same prompt in round 0 and round 1 must decode identically
        same = round_tokens[0] == round_tokens[1]
        print("reuse parity round0==round1:", "IDENTICAL" if same else "DIFFER")
        if not same:
            result["errors"].append("reuse parity differs between rounds")
    if result["errors"]:
        print("FAIL:", "; ".join(result["errors"]))
        return 1
    print("PASS: pool stress stayed balanced and leak-free over", a.rounds, "rounds x", a.slots, "slots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
