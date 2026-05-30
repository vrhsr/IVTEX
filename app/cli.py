#!/usr/bin/env python3
"""
cli.py  —  Complaint Auto-Routing System · Command Line Interface
────────────────────────────────────────────────────────────────

Usage examples:
    # Submit a text complaint interactively
    python app/cli.py

    # Submit text directly
    python app/cli.py --text "Pothole on MG Road near hospital. Very dangerous!"

    # Submit an audio file
    python app/cli.py --audio /path/to/complaint.mp3

    # Submit a video file
    python app/cli.py --video /path/to/complaint.mp4

    # Control number of similar complaints shown
    python app/cli.py --text "Power outage in Sector 14" --top-k 3
"""

import argparse
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from inference.engine import ComplaintRoutingEngine, SAVE_DIR


# ─── ANSI colours ─────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    DIM    = "\033[2m"

PRIORITY_COLOUR = {"High": C.RED, "Medium": C.YELLOW, "Low": C.GREEN}


def print_banner():
    print(f"""
{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════╗
║     COMPLAINT AUTO-ROUTING SYSTEM  v1.0          ║
║     IVTEX Corporate Solutions Pvt. Ltd.          ║
╚══════════════════════════════════════════════════╝{C.RESET}
""")


def format_result(result: dict) -> str:
    lines = []
    sep   = "─" * 54

    # ── Officer
    o = result["officer"]
    lines.append(f"\n{C.BOLD}{'ASSIGNED OFFICER':─<54}{C.RESET}")
    lines.append(f"  {C.BOLD}Name       :{C.RESET} {o['name']}")
    lines.append(f"  {C.BOLD}Department :{C.RESET} {o['department']}")
    lines.append(f"  {C.BOLD}Officer ID :{C.RESET} {o['id']}")
    lines.append(f"  {C.BOLD}Confidence :{C.RESET} {o['confidence']}%")

    # ── Priority
    p = result["priority"]
    pc = PRIORITY_COLOUR.get(p["level"], C.RESET)
    lines.append(f"\n{C.BOLD}{'PRIORITY':─<54}{C.RESET}")
    lines.append(f"  {C.BOLD}Level      :{C.RESET} {pc}{C.BOLD}{p['level']}{C.RESET}")
    lines.append(f"  {C.BOLD}Confidence :{C.RESET} {p['confidence']}%")

    # ── ETA
    lines.append(f"\n{C.BOLD}{'ESTIMATED RESOLUTION TIME':─<54}{C.RESET}")
    lines.append(f"  {C.BOLD}ETA        :{C.RESET} {C.CYAN}{result['eta_days']} day(s){C.RESET}")

    # ── Similar
    sims = result.get("similar_complaints", [])
    lines.append(f"\n{C.BOLD}{'SIMILAR PAST COMPLAINTS':─<54}{C.RESET}")
    if not sims:
        lines.append("  (none found)")
    for i, s in enumerate(sims, 1):
        snippet = textwrap.shorten(s["text_snippet"], width=65)
        lines.append(
            f"  {i}. [{s['complaint_id']}] "
            f"Score={s['similarity_score']:.3f}  "
            f"{PRIORITY_COLOUR.get(s['priority'], '')}[{s['priority']}]{C.RESET}\n"
            f"     {C.DIM}{snippet}{C.RESET}"
        )

    if result.get("modality") and result["modality"] != "text":
        lines.append(f"\n{C.DIM}  (Input modality: {result['modality']}){C.RESET}")

    return "\n".join(lines)


def interactive_mode(engine: ComplaintRoutingEngine, top_k: int):
    print(f"{C.DIM}Type your complaint and press Enter. "
          f"Type 'quit' to exit.{C.RESET}\n")
    while True:
        try:
            text = input(f"{C.BOLD}Enter complaint:{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        result = engine.predict(text, top_k_similar=top_k)
        print(format_result(result))
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Complaint Auto-Routing System — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--text",   type=str, help="Complaint text (inline)")
    parser.add_argument("--audio",  type=str, help="Path to audio file (.wav/.mp3/…)")
    parser.add_argument("--video",  type=str, help="Path to video file (.mp4/.mkv/…)")
    parser.add_argument("--top-k",  type=int, default=5,
                        help="Number of similar complaints to retrieve (default: 5)")
    parser.add_argument("--json",   action="store_true",
                        help="Output raw JSON instead of formatted display")
    args = parser.parse_args()

    print_banner()
    engine = ComplaintRoutingEngine().load(SAVE_DIR)

    result = None

    if args.text:
        result = engine.process(text=args.text, top_k=args.top_k)
    elif args.audio:
        result = engine.process(audio_path=args.audio, top_k=args.top_k)
    elif args.video:
        result = engine.process(video_path=args.video, top_k=args.top_k)
    else:
        # Interactive mode
        interactive_mode(engine, args.top_k)
        return

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        if result.get("modality") != "text":
            print(f"{C.DIM}Transcription: {result.get('source_text', '')[:200]}…{C.RESET}\n")
        print(format_result(result))


if __name__ == "__main__":
    main()
