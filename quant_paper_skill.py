# -*- coding: utf-8 -*-
"""
quant_paper_skill.py

本地可复用 Skill：
1) PDF -> Markdown（MinerU / magic-pdf，默认 -m auto，利于版式/图表/公式）
2) Markdown -> 结构化 JSON（兼容 Anthropic API 的网关）
3) 标准化 Markdown 报告（单输入 PDF -> 单套输出：json + report）

用法：
  python quant_paper_skill.py --pdf "C:\\path\\to\\paper.pdf"
  python quant_paper_skill.py   # 从 QUANT_SHARED_DIR 随机抽 2 篇 PDF 批量测试
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
from openai import OpenAI


BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "http://1.95.142.151:3000/").strip()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
QUANT_SHARED_DIR = Path(r"C:\Users\19216\Desktop\quant_shared")
OUTPUT_DIR = Path(__file__).resolve().parent / "parsed_results"
DEFAULT_MAGIC_PDF_METHOD = os.getenv("MAGIC_PDF_METHOD", "auto").strip() or "auto"
LLM_RETRY_TIMES = int(os.getenv("LLM_RETRY_TIMES", "3"))
LLM_RETRY_DELAY_SEC = float(os.getenv("LLM_RETRY_DELAY_SEC", "1.5"))
PDF_PARSE_RETRY_TIMES = int(os.getenv("PDF_PARSE_RETRY_TIMES", "2"))


def _safe_err_text(err: Exception) -> str:
    """避免 Windows 控制台编码导致报错信息打印失败。"""
    text = str(err)
    return text.encode("gbk", errors="replace").decode("gbk", errors="replace")


def _run_with_retries(func, attempts: int, delay_sec: float, op_name: str):
    """
    通用重试封装（指数退避），最后一次失败时抛出原始异常。
    """
    if attempts <= 1:
        return func()

    last_err: Optional[Exception] = None
    for i in range(1, attempts + 1):
        try:
            return func()
        except Exception as e:
            last_err = e
            if i >= attempts:
                break
            time.sleep(delay_sec * (2 ** (i - 1)))
    raise RuntimeError(f"{op_name} 重试失败: {last_err}") from last_err


def _load_auth_token() -> str:
    """优先环境变量，其次 .env 文件。"""
    token = os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip()
    if token:
        return token

    env_candidates = [
        Path(".env"),
        Path(__file__).resolve().parent / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in env_candidates:
        if not env_path.is_file():
            continue
        try:
            text = env_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            if k.strip() == "ANTHROPIC_AUTH_TOKEN":
                return v.strip().strip('"').strip("'")

    return ""


def _pymupdf_to_plain_md(pdf: Path) -> str:
    """
    magic-pdf 不可用时的后备：PyMuPDF 逐页抽取纯文本，结构弱于 MinerU，但无需 detectron2。
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError("未安装 PyMuPDF（import fitz）。请 pip install pymupdf") from e

    doc = fitz.open(pdf)
    try:
        parts: List[str] = [
            "<!-- parsed_by: pymupdf_fallback -->\n\n",
            "> 说明：magic-pdf 未成功产出 Markdown（常见原因：缺少 detectron2 等依赖）。",
            "以下为 PyMuPDF 纯文本抽取，公式与复杂图表可能不完整。\n\n",
        ]
        for i in range(len(doc)):
            page = doc[i]
            parts.append(f"\n\n## 第 {i + 1} 页\n\n")
            parts.append(page.get_text(sort=True) or "")
        return "".join(parts)
    finally:
        doc.close()


def _docling_to_md(pdf: Path) -> str:
    """
    Docling 后备：尽量保留结构、表格，并可输出公式的 LaTeX（效果取决于文档质量与模型）。
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as e:
        raise RuntimeError("未安装 docling。请先 pip install docling") from e

    converter = DocumentConverter()
    result = converter.convert(str(pdf))
    # Docling API: DocumentConversionResult.document.render_as_markdown()
    doc = getattr(result, "document", None)
    if not doc:
        raise RuntimeError("docling 未返回 document。")
    md = doc.render_as_markdown()
    md = (md or "").strip()
    if not md:
        raise RuntimeError("docling 转换结果为空。")
    return "<!-- parsed_by: docling_fallback -->\n\n" + md + "\n"


def _magic_pdf_to_md(pdf: Path, out: Path, method: str) -> str:
    cmd = ["magic-pdf", "-p", str(pdf), "-o", str(out), "-m", method]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except FileNotFoundError as e:
        raise RuntimeError("未找到 magic-pdf 命令，请先安装并确保在 PATH 中。") from e
    except Exception as e:
        raise RuntimeError(f"运行 magic-pdf 失败: {e}") from e

    stderr_text = (proc.stderr or "").strip()
    stdout_text = (proc.stdout or "").strip()
    merged_log = "\n".join([x for x in [stdout_text, stderr_text] if x]).strip()

    if proc.returncode != 0:
        raise RuntimeError(f"magic-pdf 运行失败（returncode={proc.returncode}）。日志:\n{merged_log}")

    # 仅选择解析器产出的 markdown，排除流程自身生成的报告文件，避免“拿旧错误报告当论文正文”。
    md_files = [
        p
        for p in out.rglob("*.md")
        if not p.name.endswith("_error_report.md")
        and not p.name.endswith("_report.md")
        and p.name != f"{pdf.stem}.md"
    ]
    if not md_files:
        lower_log = merged_log.lower()
        if "traceback" in lower_log or "error" in lower_log or "modulenotfounderror" in lower_log:
            raise RuntimeError(f"magic-pdf 运行异常且未产出 Markdown。日志:\n{merged_log}")
        raise RuntimeError(f"magic-pdf 运行成功，但在 {out} 下未找到 .md 文件。日志:\n{merged_log}")

    stem_lower = pdf.stem.lower()
    matched = [p for p in md_files if stem_lower in p.stem.lower() or stem_lower in str(p).lower()]
    candidates = matched if matched else md_files

    # 优先使用 MinerU/magic-pdf 的 auto 子目录下产物
    auto_candidates = [p for p in candidates if "auto" in [x.lower() for x in p.parts]]
    if auto_candidates:
        candidates = auto_candidates
    chosen = max(candidates, key=lambda p: p.stat().st_mtime)

    try:
        return chosen.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"读取 Markdown 失败: {chosen} -> {e}") from e


def parse_pdf_to_md(pdf_path: str, output_dir: str, method: Optional[str] = None) -> str:
    """
    优先 MinerU magic-pdf；失败且未设置 PDF_PARSE_NO_FALLBACK=1 时，退回 PyMuPDF 纯文本。
    method: auto | txt | ocr
    """
    pdf = Path(pdf_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF 不存在: {pdf}")

    m = (method or DEFAULT_MAGIC_PDF_METHOD).strip().lower()
    if m not in ("auto", "txt", "ocr"):
        m = "auto"

    try:
        return _run_with_retries(
            lambda: _magic_pdf_to_md(pdf, out, m),
            attempts=max(1, PDF_PARSE_RETRY_TIMES),
            delay_sec=1.0,
            op_name="PDF 解析（magic-pdf）",
        )
    except RuntimeError as magic_err:
        no_fb = os.getenv("PDF_PARSE_NO_FALLBACK", "").strip().lower() in ("1", "true", "yes")
        if no_fb:
            raise
        try:
            return _docling_to_md(pdf)
        except Exception as docling_err:
            try:
                return _pymupdf_to_plain_md(pdf)
            except Exception as fb_err:
                raise RuntimeError(
                    f"magic-pdf 失败: {magic_err}\n"
                    f"docling 后备失败: {docling_err}\n"
                    f"PyMuPDF 后备失败: {fb_err}"
                ) from fb_err


def _build_extract_prompt(md_content: str) -> str:
    return f"""
你是一名顶级量化研究助手。请阅读下方由 PDF 转换得到的 Markdown 文档，完成分类与信息抽取。

重要约束（为保证可解析性）：
- 你必须输出**严格合法 JSON**，不要输出任何额外解释文字。
- 所有字符串字段都必须正确闭合引号；反斜杠 `\\` 必须按 JSON 规则转义（例如 LaTeX 里出现 `\\alpha` 也要保证作为 JSON 字符串合法）。
- `Shared.Formulas_LaTeX` 如过长，请**只保留最核心的 5–15 条公式**（优先保留定义式与目标函数），不要为了“全量”导致 JSON 断裂或截断。

## 第一步：分类（三选一）
只能选其一，填在 Meta_Info.Category：
- **因子类**：主要讨论 alpha 因子、风险因子、因子收益、因子构造与检验等。
- **策略类**：主要讨论可交易策略、规则体系、回测框架、进出场与仓位规则等。
- **工具类**：主要讨论数据处理工具、统计检验方法、优化器、回测库用法、工程实现方法等。

## 第二步：输出
1) **必须**输出严格合法 JSON（不要 Markdown 代码块外壳以外的任何解释文字）。
2) 解析务必详细，不省略关键公式、参数阈值、数据清洗步骤。
3) **Factor_Block / Strategy_Block / Tool_Block** 中与当前类别无关的必须填 null；**仅当前类别对应块为非 null**。
4) 公式尽量用 LaTeX 源码；多个公式用换行拼接。
5) Meta_Info.Summary：**500–1000 个汉字**的中文详细摘要（不足或超长都不合格），需包含：研究问题、方法、数据、主要结论、局限、可落地启发。

JSON 模板（字段名必须完全一致）：
{{
  "Meta_Info": {{
    "Title": "文献标题",
    "Category": "因子类 | 策略类 | 工具类",
    "Summary": "500-1000字中文摘要"
  }},
  "Shared": {{
    "Overview": "大致内容：背景、问题、方法概要、数据与结论",
    "Core_Theme": "核心策略/因子/方法思路（精炼）",
    "Formulas_LaTeX": "核心公式 LaTeX 汇总"
  }},
  "Factor_Block": null,
  "Strategy_Block": null,
  "Tool_Block": null
}}

当 Category 为 **因子类** 时，将 Factor_Block 设为如下结构（另两类块为 null）：
{{
  "Factors": [
    {{
      "Name": "因子名称或简称",
      "Definition": "定义与直觉",
      "Construction": "构造步骤、计算窗口、截面/时序处理",
      "Economic_Logic": "经济学或行为金融学解释"
    }}
  ],
  "Data_Fields": ["所需字段，如 Open, Close, VWAP"],
  "Data_Processing": ["中性化、去极值、标准化等详细说明"]
}}

当 Category 为 **策略类** 时，将 Strategy_Block 设为如下结构：
{{
  "How_Strategy_Is_Built": "策略如何从数据与指标一步步得到交易信号与仓位",
  "Logic_Chain": "完整逻辑链条（因果与顺序）",
  "Key_Rules": ["进出场、过滤、持仓、关键参数阈值"],
  "Risk_Management": "止损、仓位、分散化等"
}}

当 Category 为 **工具类** 时，将 Tool_Block 设为如下结构：
{{
  "Tool_Or_Method_Name": "",
  "Purpose": "解决什么问题",
  "Implementation_Detail": "尽可能详细：算法、数据结构、与量化流程的结合方式",
  "Steps": ["步骤1", "步骤2"],
  "Parameters_And_Caveats": "参数、假设、适用边界与坑"
}}

以下是 Markdown 原文：
-----BEGIN MARKDOWN-----
{md_content}
-----END MARKDOWN-----
""".strip()


def _truncate_md_for_llm(md: str, max_chars: int = 80000) -> str:
    """
    防止输入过长导致模型输出被截断，从而产生不合法 JSON。
    保留开头+结尾，中间用占位提示。
    """
    s = (md or "").strip()
    if len(s) <= max_chars:
        return s
    head = s[: max_chars // 2]
    tail = s[-max_chars // 2 :]
    return (
        head
        + "\n\n...[中间内容过长已截断，用于保证结构化抽取稳定性]...\n\n"
        + tail
    )


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return json.loads(m.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("模型返回中未找到可解析 JSON。")


def _zh_char_len(s: str) -> int:
    """粗略按「汉字 + 字母数字」计长度，用于摘要区间校验。"""
    return len(s.strip())


def _clamp_summary_length(summary: str, min_len: int = 500, max_len: int = 1000) -> str:
    s = summary.strip()
    if len(s) <= max_len:
        return s
    clipped = s[:max_len]
    cut_points = [clipped.rfind("。"), clipped.rfind("！"), clipped.rfind("？")]
    cut = max(cut_points)
    if cut >= min_len:
        return clipped[: cut + 1].strip()
    return clipped.strip()


def _validate_category_payload(data: Dict[str, Any]) -> None:
    cat = str(data.get("Meta_Info", {}).get("Category", "")).strip()
    allowed = ("因子类", "策略类", "工具类")
    if cat not in allowed:
        raise ValueError(f"Category 必须是 {allowed} 之一，当前: {cat!r}")

    fb = data.get("Factor_Block")
    sb = data.get("Strategy_Block")
    tb = data.get("Tool_Block")

    if cat == "因子类" and fb in (None, {}):
        raise ValueError("因子类文献 Factor_Block 不能为空。")
    if cat == "策略类" and sb in (None, {}):
        raise ValueError("策略类文献 Strategy_Block 不能为空。")
    if cat == "工具类" and tb in (None, {}):
        raise ValueError("工具类文献 Tool_Block 不能为空。")

    if cat == "因子类":
        data["Strategy_Block"] = None
        data["Tool_Block"] = None
    elif cat == "策略类":
        data["Factor_Block"] = None
        data["Tool_Block"] = None
    else:
        data["Factor_Block"] = None
        data["Strategy_Block"] = None


def extract_knowledge(md_content: str) -> Dict[str, Any]:
    """调用大模型提取结构化知识 JSON。"""
    token = _load_auth_token()
    if not token:
        raise RuntimeError(
            "未找到 ANTHROPIC_AUTH_TOKEN。请在环境变量或 .env 中配置该 token。"
        )

    provider = LLM_PROVIDER
    if provider not in ("anthropic", "deepseek"):
        provider = "anthropic"

    anth_client = anthropic.Anthropic(base_url=BASE_URL, api_key=token) if provider == "anthropic" else None
    ds_client = OpenAI(api_key=token, base_url=DEEPSEEK_BASE_URL) if provider == "deepseek" else None
    prompt = _build_extract_prompt(_truncate_md_for_llm(md_content))

    def _call_llm(user_prompt: str, max_tokens: int = 7000) -> str:
        def _once():
            if provider == "deepseek":
                return ds_client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    temperature=0.1,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": user_prompt}],
                )
            return anth_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                temperature=0.1,
                messages=[{"role": "user", "content": user_prompt}],
            )

        try:
            resp = _run_with_retries(
                _once,
                attempts=max(1, LLM_RETRY_TIMES),
                delay_sec=max(0.1, LLM_RETRY_DELAY_SEC),
                op_name="LLM 调用",
            )
        except Exception as e:
            raise RuntimeError(f"调用 Anthropic 失败: {e}") from e

        content_text = ""
        if provider == "deepseek":
            choices = getattr(resp, "choices", None) or []
            if choices:
                msg = getattr(choices[0], "message", None)
                content_text = (getattr(msg, "content", "") or "").strip()
        else:
            if hasattr(resp, "content") and resp.content:
                parts: List[str] = []
                for blk in resp.content:
                    txt = getattr(blk, "text", "")
                    if txt:
                        parts.append(txt)
                content_text = "\n".join(parts).strip()
        if not content_text:
            raise RuntimeError("模型返回为空。")
        return content_text

    content_text = _call_llm(prompt, max_tokens=7000)

    try:
        data = _extract_json_from_text(content_text)
    except Exception:
        # 轻量修复重试：把原始输出喂回去，要求只修复为合法 JSON，可适度删减公式字段
        repair_prompt = (
            "你上一次的输出 JSON 不合法或被截断。\n"
            "请基于下方“原始输出草稿”，只输出一份**严格合法 JSON**，字段必须与模板一致。\n"
            "如果 Shared.Formulas_LaTeX 过长或导致截断，请删减到最关键的 5–15 条。\n\n"
            "-----BEGIN DRAFT-----\n"
            f"{content_text}\n"
            "-----END DRAFT-----\n"
        )
        content_text2 = _call_llm(repair_prompt, max_tokens=3500)
        try:
            data = _extract_json_from_text(content_text2)
        except Exception as e2:
            raise RuntimeError(
                f"模型输出不是合法 JSON: {e2}\n原始输出(前2000):\n{content_text[:2000]}\n\n修复输出(前2000):\n{content_text2[:2000]}"
            ) from e2

    data.setdefault("Meta_Info", {})
    summary = str(data["Meta_Info"].get("Summary", "")).strip()
    summary = _clamp_summary_length(summary, min_len=500, max_len=1000)
    data["Meta_Info"]["Summary"] = summary

    n = _zh_char_len(summary)
    if n < 500:
        raise RuntimeError(f"摘要长度不足（当前约 {n} 字），要求 500–1000 字。")
    if n > 1000:
        data["Meta_Info"]["Summary"] = _clamp_summary_length(summary, min_len=500, max_len=1000)

    _validate_category_payload(data)
    return data


def render_standard_report(data: Dict[str, Any]) -> str:
    """
    将 JSON 转为固定版式的中文 Markdown 报告（一篇 PDF 对应一份可读总结）。
    """
    meta = data.get("Meta_Info") or {}
    shared = data.get("Shared") or {}
    title = str(meta.get("Title", "未命名文献")).strip()
    cat = str(meta.get("Category", "")).strip()
    summary = str(meta.get("Summary", "")).strip()

    lines: List[str] = [
        f"# {title}",
        "",
        "## 元信息",
        "",
        f"- **类别**：{cat}",
        "",
        "## 摘要（500–1000 字）",
        "",
        summary,
        "",
        "## 大致内容",
        "",
        str(shared.get("Overview", "")).strip(),
        "",
        "## 核心策略 / 因子 / 方法思路",
        "",
        str(shared.get("Core_Theme", "")).strip(),
        "",
        "## 公式（LaTeX）",
        "",
        str(shared.get("Formulas_LaTeX", "")).strip() or "（文中未提取到独立公式条目）",
        "",
    ]

    fb = data.get("Factor_Block")
    if fb and cat == "因子类":
        lines.extend(["## 因子类专节：因子清单与构造", ""])
        factors = fb.get("Factors") or []
        if isinstance(factors, list):
            for i, f in enumerate(factors, 1):
                if not isinstance(f, dict):
                    continue
                lines.append(f"### 因子 {i}：{f.get('Name', '')}")
                lines.append("")
                lines.append(f"- **定义**：{f.get('Definition', '')}")
                lines.append(f"- **构造**：{f.get('Construction', '')}")
                lines.append(f"- **经济逻辑**：{f.get('Economic_Logic', '')}")
                lines.append("")
        df = fb.get("Data_Fields")
        if df:
            lines.append("**数据字段**：")
            lines.append("")
            lines.append(", ".join(str(x) for x in df) if isinstance(df, list) else str(df))
            lines.append("")
        dp = fb.get("Data_Processing")
        if dp:
            lines.append("**数据处理**：")
            lines.append("")
            if isinstance(dp, list):
                for x in dp:
                    lines.append(f"- {x}")
            else:
                lines.append(str(dp))
            lines.append("")

    sb = data.get("Strategy_Block")
    if sb and cat == "策略类":
        lines.extend(["## 策略类专节：如何构建与逻辑", ""])
        lines.append(f"### 策略如何做出\n\n{str(sb.get('How_Strategy_Is_Built', '')).strip()}")
        lines.append("")
        lines.append(f"### 逻辑链条\n\n{str(sb.get('Logic_Chain', '')).strip()}")
        lines.append("")
        kr = sb.get("Key_Rules") or []
        lines.append("### 关键规则")
        lines.append("")
        if isinstance(kr, list):
            for x in kr:
                lines.append(f"- {x}")
        else:
            lines.append(str(kr))
        lines.append("")
        lines.append(f"### 风险管理\n\n{str(sb.get('Risk_Management', '')).strip()}")
        lines.append("")

    tb = data.get("Tool_Block")
    if tb and cat == "工具类":
        lines.extend(["## 工具类专节：方法实现细节", ""])
        lines.append(f"- **名称**：{str(tb.get('Tool_Or_Method_Name', '')).strip()}")
        lines.append(f"- **用途**：{str(tb.get('Purpose', '')).strip()}")
        lines.append("")
        lines.append("### 实现细节（尽可能详尽）")
        lines.append("")
        lines.append(str(tb.get("Implementation_Detail", "")).strip())
        lines.append("")
        steps = tb.get("Steps") or []
        lines.append("### 步骤分解")
        lines.append("")
        if isinstance(steps, list):
            for i, x in enumerate(steps, 1):
                lines.append(f"{i}. {x}")
        else:
            lines.append(str(steps))
        lines.append("")
        lines.append(f"### 参数与注意\n\n{str(tb.get('Parameters_And_Caveats', '')).strip()}")
        lines.append("")

    lines.append("---")
    lines.append("*本报告由 quant_paper_skill 由 PDF→Markdown→LLM 自动生成，请核对公式与参数。*")
    lines.append("")
    return "\n".join(lines)


def process_one_pdf(
    pdf_path: str,
    out_root: Optional[Path] = None,
    method: Optional[str] = None,
    output_format: str = "both",
) -> Dict[str, Any]:
    """
    单篇 PDF 完整流水线：解析 → 抽取 → 写 md / json / *_report.md。
    返回 paths 与 data。
    """
    pdf = Path(pdf_path).resolve()
    root = Path(out_root) if out_root else OUTPUT_DIR
    single_out = root / pdf.stem
    single_out.mkdir(parents=True, exist_ok=True)

    md_saved = single_out / f"{pdf.stem}.md"
    json_path = single_out / f"{pdf.stem}.json"
    report_path = single_out / f"{pdf.stem}_report.md"
    error_report_path = single_out / f"{pdf.stem}_error_report.md"

    try:
        md_text = parse_pdf_to_md(str(pdf), str(single_out), method=method)
        md_saved.write_text(md_text, encoding="utf-8")

        data = extract_knowledge(md_text)

        fmt = (output_format or "both").strip().lower()
        if fmt not in ("json", "md", "both"):
            fmt = "both"

        if fmt in ("json", "both"):
            json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if fmt in ("md", "both"):
            report_path.write_text(render_standard_report(data), encoding="utf-8")

        return {
            "status": "success",
            "pdf": str(pdf),
            "md_path": str(md_saved),
            "json_path": str(json_path) if fmt in ("json", "both") else None,
            "report_path": str(report_path) if fmt in ("md", "both") else None,
            "error_report_path": None,
            "data": data,
        }
    except Exception as e:
        tb = traceback.format_exc()
        error_report = (
            f"# PDF 处理失败报告\n\n"
            f"- PDF: `{pdf}`\n"
            f"- 输出目录: `{single_out}`\n"
            f"- 错误: `{_safe_err_text(e)}`\n\n"
            f"## Traceback\n\n```text\n{tb}\n```\n"
        )
        error_report_path.write_text(error_report, encoding="utf-8")
        raise RuntimeError(f"{_safe_err_text(e)}（错误报告: {error_report_path}）") from e


def _pick_two_pdfs(pdf_dir: Path) -> List[Path]:
    pdfs = [p for p in pdf_dir.rglob("*.pdf") if p.is_file()]
    if not pdfs:
        raise FileNotFoundError(f"目录下未找到 PDF: {pdf_dir}")
    if len(pdfs) >= 2:
        return random.sample(pdfs, 2)
    return [pdfs[0], pdfs[0]]


def _run_batch_test(out_root: Path, method: Optional[str] = None) -> None:
    random.seed()
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("quant_paper_skill: PDF -> Markdown -> JSON -> Report")
    print("=" * 80)
    print(f"PDF 目录: {QUANT_SHARED_DIR}")
    print(f"输出目录: {out_root}")
    print(f"magic-pdf 模式: {method or DEFAULT_MAGIC_PDF_METHOD}")

    try:
        pdf_list = _pick_two_pdfs(QUANT_SHARED_DIR)
    except Exception as e:
        print(f"[ERROR] 选取 PDF 失败: {e}")
        sys.exit(1)

    print(f"本次测试 PDF 数量: {len(pdf_list)}")
    for i, pdf in enumerate(pdf_list, 1):
        print(f"{i}. {pdf}")
    print("-" * 80)

    success = 0
    for idx, pdf in enumerate(pdf_list, 1):
        print(f"[{idx}/{len(pdf_list)}] 处理: {pdf.name}")
        try:
            result = process_one_pdf(str(pdf), out_root, method=method)
            print(f"  - Markdown: {result['md_path']}")
            if result.get("json_path"):
                print(f"  - JSON: {result['json_path']}")
            if result.get("report_path"):
                print(f"  - 报告: {result['report_path']}")
            success += 1
        except Exception as e:
            print(f"  - [ERROR] {_safe_err_text(e)}")

    print("-" * 80)
    print(f"完成: {success}/{len(pdf_list)} 成功")
    print(f"结果目录: {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="量化 PDF：magic-pdf + LLM 分类与摘要")
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="单篇 PDF 的绝对或相对路径；省略则批量测试 QUANT_SHARED_DIR",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=f"输出根目录，默认 {OUTPUT_DIR}",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=("auto", "txt", "ocr"),
        help="magic-pdf -m 参数；默认读环境变量 MAGIC_PDF_METHOD 或 auto",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=("json", "md", "both"),
        help="最终输出格式：json / md / both（默认 both）",
    )
    args = parser.parse_args()

    out = Path(args.out_dir).resolve() if args.out_dir else OUTPUT_DIR

    if args.pdf:
        out.mkdir(parents=True, exist_ok=True)
        try:
            result = process_one_pdf(args.pdf, out, method=args.method, output_format=args.format)
            print("处理完成。")
            print(f"Markdown: {result['md_path']}")
            if result.get("json_path"):
                print(f"JSON:     {result['json_path']}")
            if result.get("report_path"):
                print(f"报告:     {result['report_path']}")
        except Exception as e:
            print(f"[ERROR] {_safe_err_text(e)}", file=sys.stderr)
            sys.exit(1)
        return

    _run_batch_test(out, method=args.method)


if __name__ == "__main__":
    main()
