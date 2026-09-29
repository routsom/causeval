"""causeval command-line interface.

Phase 0 ships only ``version``. The ``run``/``compare``/``gate``/``plan`` commands and the
intervention/audit/attribute commands arrive in later phases (SPEC §4).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from causeval import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="causeval", description=__doc__)
    parser.add_argument("--version", action="version", version=f"causeval {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("version", help="print the causeval version")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "version" or args.command is None:
        print(f"causeval {__version__}")
        return 0
    parser.error(f"unknown command: {args.command}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
