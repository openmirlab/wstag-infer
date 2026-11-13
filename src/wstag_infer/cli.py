"""Minimal CLI for inspecting one localization result as JSON.

Reads: the public WSTAGSession facade and LocalizationResult.to_dict().
"""

from __future__ import annotations

import argparse
import json

from .session import WSTAGSession


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate described sounds with small WSTAG")
    parser.add_argument("audio", help="WAV or FLAC readable by SoundFile")
    parser.add_argument("phrases", nargs="+", help="English descriptions, preferably lowercase")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--offline", action="store_true", help="Use only cached/local files")
    args = parser.parse_args()
    with WSTAGSession(
        device=args.device,
        cache_dir=args.cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        local_files_only=args.offline,
    ) as session:
        result = session.infer(args.audio, args.phrases, threshold=args.threshold)
    print(json.dumps(result.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
