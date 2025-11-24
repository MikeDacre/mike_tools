from datetime import datetime

# YAML frontmatter template
FRONT = """---
title: {title}
date: {date}
created: {timestamp}
updated: {timestamp}
id: {timestamp}
---
"""

def handler(pd: "pipedream"):
    """
    Pipedream-compatible handler function using the `pd` context object.

    Expects:
        pd.steps["trigger"]["event"]["file_name"]
        pd.steps["trigger"]["event"]["content"]

    Returns:
        dict: Result with file path, title, and timestamp metadata.
    """
    file = pd.steps["trigger"]["event"]["file"]

    file = event["file"]
    file_name = file.get("file_name", "untitled").strip().replace(" ", "_")
    title = file_name.replace("_", " ").title()

    content = event.get("content", "").strip()

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y%m%d%H%M")

    frontmatter = FRONT.format(title=title, date=date_str, timestamp=timestamp)
    full_text = f"{frontmatter}\n{content}\n"

    output_path = f"/tmp/{file_name}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    return {
        "status": "success",
        "file_path": output_path,
        "title": title,
        "timestamp": timestamp
    }

