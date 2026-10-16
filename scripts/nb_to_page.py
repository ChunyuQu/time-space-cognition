"""Execute a notebook and publish it as a Hugo page bundle.

    python scripts/nb_to_page.py notebooks/foundations/01_behavioral_data.ipynb

The notebook is executed in its own directory and saved back with outputs, so
the copy on GitHub shows the same figures as the website. Page front matter
comes from the notebook metadata under "page":

    {"page": {"title": ..., "summary": ..., "weight": 10, "slug": "...",
              "section": "modeling/foundations", "date": "2026-10-15"}}

An optional "cover_fig": N turns figure N into the card cover
static/images/foundations/<slug>.webp (1200 x 500, white) and adds it to the
front matter.

Cells tagged "skip-page" are left out of the page; cells tagged "hide-input"
show only their outputs. Inline maths written as $...$ becomes \\(...\\) so that
Goldmark's passthrough extension hands it to KaTeX untouched.
"""

import argparse
import base64
import json
import re
import shutil
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/ChunyuQu/time-space-cognition/blob/main"
MAX_TEXT_LINES = 40

# Inline maths may wrap onto the next line, but never contains a dollar sign itself.
INLINE_MATH = re.compile(r"(?<![\\$])\$(?!\$)([^$]+?)(?<![\\$])\$(?!\$)")
FENCE = re.compile(r"(```.*?```)", re.S)


def convert_math(md):
    # Leave fenced code untouched; only prose gets its $...$ rewritten.
    parts = FENCE.split(md)
    for i in range(0, len(parts), 2):
        parts[i] = INLINE_MATH.sub(lambda m: r"\(" + m.group(1) + r"\)", parts[i])
    return "".join(parts)


def text_block(text):
    lines = text.rstrip("\n").splitlines()
    if len(lines) > MAX_TEXT_LINES:
        lines = lines[:MAX_TEXT_LINES] + ["..."]
    return "```text\n" + "\n".join(lines) + "\n```"


def strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def render_outputs(outputs, bundle, counter):
    blocks = []
    for out in outputs:
        kind = out.get("output_type")
        if kind == "stream" and out.get("name") == "stdout":
            blocks.append(text_block(strip_ansi(out["text"])))
        elif kind in ("display_data", "execute_result"):
            data = out.get("data", {})
            if "image/png" in data:
                counter[0] += 1
                name = f"fig-{counter[0]:02d}.png"
                (bundle / name).write_bytes(base64.b64decode(data["image/png"]))
                blocks.append(f"![Figure {counter[0]}]({name})")
            elif "text/markdown" in data:
                blocks.append(data["text/markdown"])
            elif "text/plain" in data:
                blocks.append(text_block(data["text/plain"]))
        elif kind == "error":
            raise RuntimeError("notebook raised: " + out.get("ename", "") + ": " + out.get("evalue", ""))
    return blocks


def front_matter(meta, nb_rel):
    # JSON strings are valid double-quoted YAML, so apostrophes and colons are safe.
    lines = ["---"]
    lines.append(f"title: {json.dumps(meta['title'], ensure_ascii=False)}")
    lines.append(f"date: {meta.get('date', '2026-10-15')}")
    lines.append(f"weight: {meta.get('weight', 100)}")
    if meta.get("cover_fig"):
        lines.append(f"cover: /images/foundations/{meta['slug']}.webp")
    lines.append(f"summary: {json.dumps(meta['summary'], ensure_ascii=False)}")
    lines.append("math: true")
    lines.append(f"notebook: {json.dumps(REPO_URL + '/' + nb_rel)}")
    lines.append("---")
    return "\n".join(lines)


def make_cover(png, out, size=(1200, 500)):
    from PIL import Image

    im = Image.open(png).convert("RGB")
    scale = min(size[0] / im.width, size[1] / im.height) * 0.96
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, "WEBP", quality=85)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()

    nb_path = args.notebook.resolve()
    nb = nbformat.read(nb_path, as_version=4)
    if not args.no_execute:
        NotebookClient(nb, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(nb_path.parent)}}).execute()
        nbformat.write(nb, nb_path)

    meta = nb.metadata["page"]
    bundle = ROOT / "content" / meta.get("section", "modeling/foundations") / meta["slug"]
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    nb_rel = nb_path.relative_to(ROOT).as_posix()
    body = [front_matter(meta, nb_rel)]
    counter = [0]
    for cell in nb.cells:
        tags = cell.metadata.get("tags", [])
        if "skip-page" in tags:
            continue
        if cell.cell_type == "markdown":
            src = cell.source
            # The notebook's own H1 duplicates the page title.
            src = re.sub(r"\A# .*\n?", "", src).strip()
            if src:
                body.append(convert_math(src))
        elif cell.cell_type == "code":
            if "hide-input" not in tags and cell.source.strip():
                body.append("```python\n" + cell.source.rstrip() + "\n```")
            body.extend(render_outputs(cell.get("outputs", []), bundle, counter))

    (bundle / "index.md").write_text("\n\n".join(body) + "\n", encoding="utf-8")
    if meta.get("cover_fig"):
        make_cover(bundle / f"fig-{meta['cover_fig']:02d}.png",
                   ROOT / "static" / "images" / "foundations" / f"{meta['slug']}.webp")
    print(f"{nb_rel} -> {bundle.relative_to(ROOT).as_posix()} ({counter[0]} figures)")


if __name__ == "__main__":
    main()
