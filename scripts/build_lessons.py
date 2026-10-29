"""Execute the lesson notebooks and export their readable Markdown versions.

Run from the repository root: python scripts/build_lessons.py
Notebook text and teaching code are the source; do not edit the exports separately.
"""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def github_markdown(text: str) -> str:
    """Protect display math from Markdown parsing with GitHub's math fences.

    A bare ``=`` in a dollar-delimited derivation becomes a setext heading
    before GitHub reaches MathJax. Fenced math keeps the TeX intact. Notebook
    cells retain their dollar delimiters because Jupyter uses those directly.
    Ordinary code fences, inline math, and HTML details are left untouched.
    """
    result: list[str] = []
    math_lines: list[str] | None = None
    code_fence: tuple[str, int] | None = None

    def emit_math() -> None:
        if not math_lines or not any(line.strip() for line in math_lines):
            raise ValueError("Empty display-math block")
        if result and result[-1].strip():
            result.append("")
        result.extend(["```math", *math_lines, "```"])

    for line in text.splitlines():
        stripped = line.strip()
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if code_fence is not None:
            result.append(line)
            if (fence and fence[1][0] == code_fence[0]
                    and len(fence[1]) >= code_fence[1] and not fence[2].strip()):
                code_fence = None
            continue
        if not fence and re.search(r"(?<!\\)\\phantom\b", line):
            raise ValueError("GitHub does not allow the phantom macro; use a visible symbol instead")
        if not fence and re.search(r"(?<!\\)\\operatorname\b", line):
            raise ValueError("GitHub does not allow the operatorname macro; use mathrm instead")
        if math_lines is not None:
            if stripped.endswith("$$"):
                remainder = stripped[:-2]
                if remainder:
                    math_lines.append(remainder)
                emit_math()
                math_lines = None
            else:
                math_lines.append(line)
            continue
        if fence:
            code_fence = (fence[1][0], len(fence[1]))
            result.append(line)
        elif stripped.startswith("$$"):
            remainder = stripped[2:]
            if remainder.endswith("$$"):
                math_lines = [remainder[:-2]]
                emit_math()
                math_lines = None
            else:
                math_lines = [remainder] if remainder else []
        else:
            result.append(line)
    if math_lines is not None:
        raise ValueError("Unclosed display-math block: missing closing $$")
    return "\n".join(result) + ("\n" if text.endswith("\n") else "")


def markdown_target(language: str, number: int) -> Path:
    suffix = "-en" if language == "en" else ""
    return ROOT / "lessons" / language / f"lesson-{number:02}{suffix}.md"


def rewrite_links(text: str, source: Path, target: Path) -> str:
    """Keep local links correct when moving from notebooks to lessons."""
    import os

    def replace(match: re.Match) -> str:
        url = match.group(2)
        if re.match(r"(?:https?://|mailto:|#)", url):
            return match.group(0)
        path_part, sep, anchor = url.partition("#")
        resolved = (source.parent / path_part).resolve()
        if resolved.suffix == ".ipynb" and resolved.parent.name in ("ru", "en"):
            number = int(re.search(r"lesson-(\d+)", resolved.stem).group(1))
            resolved = markdown_target(resolved.parent.name, number)
        relative = Path(os.path.relpath(resolved, target.parent)).as_posix()
        return match.group(1) + relative + (sep + anchor if sep else "") + ")"

    return re.sub(r"(!?\[[^\]]*\]\()([^\s)]+)\)", replace, text)


def export_notebook(notebook, source: Path, target: Path, language: str, number: int) -> None:
    notebook_link = f"../../notebooks/{language}/lesson-{number:02}.ipynb"
    label = "Открыть Jupyter-ноутбук" if language == "ru" else "Open the Jupyter notebook"
    note = (
        "Здесь можно прочитать урок и посмотреть готовые графики. Чтобы двигать ползунки, открой ноутбук в Jupyter."
        if language == "ru" else
        "You can read the lesson and see the plots here. To try the sliders, open the notebook in Jupyter."
    )
    parts = [f"[{label}]({notebook_link})\n\n{note}\n"]
    image_number = 0
    image_dir = ROOT / "assets" / f"lesson{number}" / language
    image_dir.mkdir(parents=True, exist_ok=True)
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            parts.append(github_markdown(rewrite_links(cell.source, source, target)))
        elif cell.cell_type == "code":
            parts.append("```python\n" + cell.source + "\n```")
            for output in cell.get("outputs", []):
                if output.output_type == "stream":
                    value = output.text.strip()
                    if value:
                        parts.append("```text\n" + value + "\n```")
                elif output.output_type in ("display_data", "execute_result"):
                    data = output.get("data", {})
                    if "image/png" in data:
                        image_number += 1
                        name = f"figure-{image_number:02}.png"
                        (image_dir / name).write_bytes(base64.b64decode(data["image/png"]))
                        alt = f"Рисунок {image_number}" if language == "ru" else f"Figure {image_number}"
                        parts.append(f"![{alt}](../../assets/lesson{number}/{language}/{name})")
                    elif "text/markdown" in data:
                        parts.append(github_markdown(data["text/markdown"]))
                    elif "text/plain" in data:
                        value = data["text/plain"].strip()
                        if value and not value.startswith(("<", "VBox(", "HBox(")):
                            parts.append("```text\n" + value + "\n```")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n\n".join(parts).rstrip() + "\n", encoding="utf-8")


def build(execute: bool = True, numbers=(1, 2, 3, 4, 5), languages=("ru", "en")) -> None:
    for language in languages:
        for number in numbers:
            source = ROOT / "notebooks" / language / f"lesson-{number:02}.ipynb"
            notebook = nbformat.read(source, as_version=4)
            if execute:
                notebook = NotebookClient(
                    notebook, timeout=180, kernel_name="python3", store_widget_state=True,
                    resources={"metadata": {"path": str(source.parent)}},
                ).execute()
            notebook.metadata.pop("widgets", None)
            for cell in notebook.cells:
                cell.metadata.pop("execution", None)
                if cell.cell_type == "code":
                    cell.outputs = [
                        output for output in cell.get("outputs", [])
                        if "application/vnd.jupyter.widget-view+json" not in output.get("data", {})
                    ]
            nbformat.validate(notebook)
            nbformat.write(notebook, source)
            export_notebook(notebook, source, markdown_target(language, number), language, number)
            print(f"Built {language}/lesson-{number:02}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-execute", action="store_true", help="Export existing notebook outputs")
    parser.add_argument("--lessons", type=int, nargs="+", choices=range(1, 6),
                        default=[1, 2, 3, 4, 5], help="Lesson numbers to build (default: all)")
    parser.add_argument("--languages", nargs="+", choices=["ru", "en"],
                        default=["ru", "en"], help="Languages to build (default: both)")
    args = parser.parse_args()
    build(execute=not args.no_execute, numbers=args.lessons, languages=args.languages)
