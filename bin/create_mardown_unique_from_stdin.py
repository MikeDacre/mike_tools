#!/usr/bin/env python3

"""
Markdown Generator Script

This script generates a markdown file with YAML frontmatter metadata. The user can
provide input from a command-line argument or STDIN. Metadata includes a title,
date, creation and update timestamps, and a unique ID based on the date/time.

Usage:
    python script.py -i "Markdown body here" -n "my_markdown_title" -d 202506081200
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Template for YAML frontmatter
FRONT = """---
title: {title}
date: {date}
created: {timestamp}
updated: {timestamp}
id: {timestamp}
---
"""

def parse_args():
    """
    Parse command-line arguments.
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a markdown file from input text and a fixed YAML frontmatter."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Input multiline string (optional; if not given, STDIN will be used)."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output.md"),
        help="Output markdown file path (default: output.md)"
    )
    parser.add_argument(
        "-d", "--datetime",
        type=str,
        help="Date-time string in format YYYYMMDDHHmm (optional). If not provided, current date/time will be used."
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        help="File title in underscore_case format. Underscores will be replaced with spaces for the title."
    )
    return parser.parse_args()

def parse_datetime(dt_str: str) -> datetime:
    """
    Parse a datetime string in the format YYYYMMDDHHmm.

    Args:
        dt_str (str): Date-time string.

    Returns:
        datetime: Parsed datetime object.
    """
    return datetime.strptime(dt_str, "%Y%m%d%H%M")

def read_input(input_arg: str) -> str:
    """
    Read input either from provided string or from STDIN.

    Args:
        input_arg (str): Input string or None.

    Returns:
        str: Input text stripped of leading/trailing whitespace.
    """
    if input_arg:
        return input_arg.strip()
    else:
        print("Enter your markdown body (Ctrl+D or Ctrl+Z to end):", file=sys.stderr)
        return sys.stdin.read().strip()


def generate_frontmatter(dt: datetime, title: str) -> str:
    """
    Generate formatted YAML frontmatter using provided datetime and title.

    Args:
        dt (datetime): Datetime for frontmatter fields.
        title (str): Title string.

    Returns:
        str: Formatted YAML frontmatter string.
    """
    date_str = dt.strftime("%Y-%m-%d")
    timestamp = dt.strftime("%Y%m%d%H%M")
    return FRONT.format(title=title, date=date_str, timestamp=timestamp)

def write_markdown(front: str, body: str, output_path: Path):
    """
    Write markdown content to file.

    Args:
        front (str): YAML frontmatter string.
        body (str): Markdown body content.
        output_path (Path): Destination path for output file.
    """
    markdown = f"{front}\n{body}\n"
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Markdown written to: {output_path}")

def main():
    """
    Main entry point for script execution.
    Parses arguments, builds frontmatter, reads input, writes markdown.
    """
    args = parse_args()
    dt = parse_datetime(args.datetime) if args.datetime else datetime.now()
    title = args.name.replace("_", " ") if args.name else "***"
    front = generate_frontmatter(dt, title)
    body = read_input(args.input)
    write_markdown(front, body, args.output)

if __name__ == "__main__":
    main()

