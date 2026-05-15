from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path


DOC_DIR = Path(__file__).resolve().parent
HTML_PATH = DOC_DIR / "HONG-2016.html"
MARKDOWN_PATH = DOC_DIR / "HONG-2016.md"
IMAGE_DIR = DOC_DIR / "HONG-2016_images"

IMG_TAG_PATTERN = re.compile(
    r'<img\b[^>]*\bsrc="data:image/(?P<kind>[a-zA-Z0-9.+-]+);base64,(?P<data>[^"]+)"',
    re.IGNORECASE,
)
MARKDOWN_IMAGE_PATTERN = re.compile(
    r'(\!\[[^\]]*\]\(\s*)([^)\s]+_img\.(?:jpg|jpeg|png|gif|webp))(\s*\))'
)


def normalize_extension(kind: str) -> str:
    kind = kind.lower()
    if kind == "jpeg":
        return "jpg"
    return kind


def extract_images() -> tuple[int, set[str]]:
    html = HTML_PATH.read_text(encoding="utf-8")
    IMAGE_DIR.mkdir(exist_ok=True)

    written: set[str] = set()
    for match in IMG_TAG_PATTERN.finditer(html):
        extension = normalize_extension(match.group("kind"))
        raw = base64.b64decode(match.group("data"))
        digest = hashlib.md5(raw).hexdigest()
        filename = f"{digest}_img.{extension}"
        path = IMAGE_DIR / filename
        if not path.exists():
            path.write_bytes(raw)
        written.add(filename)

    return len(written), written


def rewrite_markdown(available_images: set[str]) -> int:
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    rewritten_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal rewritten_count
        image_name = match.group(2).strip()
        if image_name not in available_images:
            return match.group(0)
        replacement = f"{match.group(1)}HONG-2016_images/{image_name}{match.group(3)}"
        if replacement != match.group(0):
            rewritten_count += 1
        return replacement

    updated = MARKDOWN_IMAGE_PATTERN.sub(replace, markdown)
    if updated == markdown:
        return 0

    MARKDOWN_PATH.write_text(updated, encoding="utf-8")
    return rewritten_count


def main() -> None:
    extracted_count, available_images = extract_images()
    rewritten_count = rewrite_markdown(available_images)
    print(f"extracted {extracted_count} unique images into {IMAGE_DIR.name}/")
    print(f"rewrote {rewritten_count} markdown image links in {MARKDOWN_PATH.name}")


if __name__ == "__main__":
    main()