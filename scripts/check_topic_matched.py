"""Quick quality check for topic-matched generated data."""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = PROJECT_ROOT / "data" / "gen_topic_matched"


def check_file(fpath: Path) -> None:
    lines = fpath.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        print(f"{fpath.name}: EMPTY")
        return

    rows = [json.loads(line) for line in lines]
    lengths = [r["meta"]["actual_chars"] for r in rows]
    md_count = sum(
        1 for r in rows
        if "**" in r["text"] or "```" in r["text"] or re.search(r"#{1,6}\s", r["text"])
    )

    print(f"=== {fpath.name} ===")
    print(f"  Texts: {len(rows)}")
    print(f"  Chars: min={min(lengths)}, median={sorted(lengths)[len(lengths)//2]}, "
          f"max={max(lengths)}, mean={sum(lengths)//len(lengths)}")
    print(f"  Short <300: {sum(1 for x in lengths if x < 300)}")
    print(f"  Long >8000: {sum(1 for x in lengths if x > 8000)}")
    print(f"  With markdown: {md_count}")
    print(f"  Unique doc_id: {len(set(r['doc_id'] for r in rows))}/{len(rows)}")
    print(f"  Unique human_doc_id: {len(set(r['meta']['human_doc_id'] for r in rows))}/{len(rows)}")
    print(f"  topic_matched=True: {sum(1 for r in rows if r['meta'].get('topic_matched'))}/{len(rows)}")

    last = rows[-1]
    print(f"  --- Sample (last) ---")
    print(f"  topic: {last['meta']['topic'][:100]}")
    print(f"  chars: {last['meta']['actual_chars']}")
    print(f"  text: {last['text'][:200]}...")
    print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for fpath in sorted(GEN_DIR.glob("*.jsonl")):
        check_file(fpath)
