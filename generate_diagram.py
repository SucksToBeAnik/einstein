"""
Generate a diagram of the current agent graph.

Outputs:
  diagram.mmd  — Mermaid source (always generated)
  diagram.png  — PNG via Mermaid API (requires internet; skipped if unavailable)

Usage:
  python generate_diagram.py
"""

import sys
from pathlib import Path
from agents.intent_classifier import agent

MMD_PATH = Path("diagram.mmd")
PNG_PATH = Path("diagram.png")

# LangGraph's static analysis can't see Send-based edges, so we patch them in.
SEND_PATCHES = {
    # (remove this line, add these lines in its place)
    "compare_agent --> __end__;": [
        "compare_agent -.->|Send| analyze_subject;",
        "analyze_subject --> synthesize_compare;",
        "synthesize_compare --> __end__;",
    ]
}


def patch_mermaid(mermaid: str) -> str:
    lines = mermaid.splitlines()
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped in SEND_PATCHES:
            indent = line[: len(line) - len(line.lstrip())]
            for replacement in SEND_PATCHES[stripped]:
                result.append(f"{indent}{replacement}")
        else:
            result.append(line)
    return "\n".join(result)


def main():
    graph = agent.get_graph()

    # --- Mermaid text ---
    raw_mermaid = graph.draw_mermaid()
    patched = patch_mermaid(raw_mermaid)
    MMD_PATH.write_text(patched)
    print(f"Mermaid diagram saved → {MMD_PATH}")

    # --- ASCII preview ---
    try:
        print("\n" + graph.draw_ascii() + "\n")
    except ImportError:
        pass  # grandalf not installed — skip ASCII preview

    # --- PNG via Mermaid API ---
    try:
        png_bytes = graph.draw_mermaid_png()
        PNG_PATH.write_bytes(png_bytes)
        print(f"PNG diagram saved → {PNG_PATH}")
    except Exception as exc:
        print(f"PNG export skipped ({exc})", file=sys.stderr)
        print(f"To render the diagram, paste {MMD_PATH} into https://mermaid.live")


if __name__ == "__main__":
    main()
