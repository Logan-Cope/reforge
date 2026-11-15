#!/usr/bin/env python3
"""Run the recreate pipeline through the LangGraph wrapper (human-review interrupt).

Same pipeline as scripts/recreate.py, orchestrated by the checkpointed StateGraph in
`reforge.graph`: deconstruct -> match -> gate -> review (interrupt) -> render. At the
interrupt the proposed slots print; approve to render, reject to stop without rendering.

Usage:
    PYTHONPATH=src python scripts/run_graph.py --max-shots 2 --auto-approve --no-persist
    PYTHONPATH=src python scripts/run_graph.py --reference reference/DISCIPLINE.mp4 \
        --out output/DISCIPLINE_graph.mp4      # prompts y/n at the review interrupt
"""
import argparse
import os
import uuid

from langgraph.types import Command

from reforge.graph import build_graph, get_vision


def main() -> None:
    ap = argparse.ArgumentParser(description="Recreate a reference reel via the LangGraph wrapper.")
    ap.add_argument("--reference", default="reference/DISCIPLINE.mp4")
    ap.add_argument("--out", default="output/DISCIPLINE_graph.mp4")
    ap.add_argument("--beat", type=float, default=None, help="force a regular cut cadence (s)")
    ap.add_argument("--max-shots", type=int, default=None, help="only build the first N shots")
    ap.add_argument("--no-persist", action="store_true", help="skip writing the reels/slots record")
    ap.add_argument("--auto-approve", action="store_true",
                    help="resume the review interrupt with approved=True (smoke tests/CI)")
    args = ap.parse_args()

    app = build_graph()
    cfg = {"configurable": {"thread_id": f"reel-{uuid.uuid4().hex[:8]}"}}
    print(f"[graph] thread {cfg['configurable']['thread_id']}: "
          f"deconstruct -> match -> gate -> review -> render")
    result = app.invoke(
        {"reference": args.reference, "out_path": args.out, "work_dir": "data/recreate_work",
         "max_shots": args.max_shots, "beat_s": args.beat, "persist": not args.no_persist},
        cfg,
    )

    interrupts = result.get("__interrupt__") or []
    if interrupts:
        print("[review] interrupt - proposed reel:")
        for row in interrupts[0].value:
            print(f"        slot {row['slot']:02d}: {row['clip']}")
            print(f"                 reason: {row['reason']}")
        if args.auto_approve:
            approved = True
            print("[review] --auto-approve: resuming with approved=True")
        else:
            approved = input("[review] approve render? [y/N] ").strip().lower() in ("y", "yes")
        result = app.invoke(Command(resume={"approved": approved}), cfg)

    if result.get("output"):
        out = result["output"]
        print(f"WROTE {out} ({os.path.getsize(out)} bytes)")
        if result.get("reel_id") is not None:
            print(f"persisted run: reels.id={result['reel_id']}")
    else:
        print("[review] rejected - run ended without rendering (re-run to try again)")

    cost = get_vision().cost_summary()
    by_model = "  ".join(f"{m.split('-', 1)[-1]}:{d['calls']}call/${d['usd']:.3f}"
                         for m, d in cost["per_model"].items())
    print(f"COST: ${cost['total_usd']:.3f} over {cost['total_calls']} calls  [{by_model}]")


if __name__ == "__main__":
    main()
