import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from openai import OpenAI
from pypdf import PdfReader


DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()


def extract_pdf_to_markdown(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: List[str] = [f"# {pdf_path.stem}\n"]
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        parts.append(f"\n## Page {idx}\n\n{text.strip()}\n")
    return "\n".join(parts).strip() + "\n"


def analyze_markdown(md_text: str) -> Dict:
    if not API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in environment.")

    client = OpenAI(api_key=API_KEY, base_url=DEEPSEEK_BASE_URL)
    prompt = f"""
你是量化研究助手。请阅读下面 Markdown，并输出严格合法 JSON：
{{
  "title": "论文标题",
  "category": "因子类|策略类|工具类",
  "summary_cn": "300-800字中文摘要",
  "key_points": ["要点1","要点2","要点3"],
  "formulas_latex": ["公式1","公式2"],
  "actionable_ideas": ["可落地想法1","可落地想法2"]
}}

只输出 JSON，不要额外文字。

---BEGIN MARKDOWN---
{md_text[:120000]}
---END MARKDOWN---
""".strip()

    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        temperature=0.1,
        max_tokens=3500,
        messages=[{"role": "user", "content": prompt}],
    )
    content = (resp.choices[0].message.content or "").strip()

    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise RuntimeError("LLM output is not valid JSON.")


def render_report(data: Dict) -> str:
    lines = [
        f"# {data.get('title', 'Untitled')}",
        "",
        f"- 类别：{data.get('category', '')}",
        "",
        "## 摘要",
        "",
        data.get("summary_cn", ""),
        "",
        "## 关键要点",
        "",
    ]
    for x in data.get("key_points", []) or []:
        lines.append(f"- {x}")
    lines.extend(["", "## 公式（LaTeX）", ""])
    for f in data.get("formulas_latex", []) or []:
        lines.append(f"- `{f}`")
    lines.extend(["", "## 可落地想法", ""])
    for x in data.get("actionable_ideas", []) or []:
        lines.append(f"- {x}")
    lines.append("")
    return "\n".join(lines)


def process_one_pdf(pdf_path: Path, out_root: Path) -> None:
    out_dir = out_root / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    md_text = extract_pdf_to_markdown(pdf_path)
    md_file = out_dir / f"{pdf_path.stem}.md"
    md_file.write_text(md_text, encoding="utf-8")

    data = analyze_markdown(md_text)
    json_file = out_dir / f"{pdf_path.stem}.json"
    json_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    report_file = out_dir / f"{pdf_path.stem}_report.md"
    report_file.write_text(render_report(data), encoding="utf-8")

    print(f"OK: {pdf_path}")
    print(f"  - {md_file}")
    print(f"  - {json_file}")
    print(f"  - {report_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, default=None, help="single pdf path")
    parser.add_argument("--out-dir", type=str, default="parsed_results")
    parser.add_argument("--pdf-list-file", type=str, default=None, help="text file with one pdf path per line")
    args = parser.parse_args()

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    pdfs: List[Path] = []
    if args.pdf:
        pdfs.append(Path(args.pdf).resolve())
    if args.pdf_list_file:
        for line in Path(args.pdf_list_file).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                pdfs.append((Path.cwd() / s).resolve())

    if not pdfs:
        raise RuntimeError("No PDF input found.")

    for p in pdfs:
        if not p.is_file():
            raise FileNotFoundError(f"PDF not found: {p}")
        process_one_pdf(p, out_root)


if __name__ == "__main__":
    main()
