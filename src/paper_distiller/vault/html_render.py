"""Markdown → HTML rendering for vault entries.

Converts an article's markdown body into a standalone HTML page with:
- Basic CSS for readable typography
- MathJax CDN link so LaTeX `$...$` / `$$...$$` math renders
- [[wikilink]] / [[slug|display]] expansion into <a> tags pointing at
  sibling .html files in the same directory
"""

from __future__ import annotations

import re

import markdown as _md


_WIKILINK_RE = re.compile(r"\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]")


def _expand_wikilinks(md_text: str) -> str:
    """Replace [[slug]] / [[slug|display]] with <a href="slug.html">display</a>.

    NOTE: This is a pre-markdown text substitution, so wikilinks INSIDE code
    blocks would also get expanded. In practice, distilled articles don't
    embed wikilinks inside code, so this is fine.
    """
    def repl(m):
        slug = m.group(1).strip()
        display = (m.group(2) or slug).strip()
        return f'<a href="{slug}.html">{display}</a>'
    return _WIKILINK_RE.sub(repl, md_text)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif;
    max-width: 820px;
    margin: 2em auto;
    padding: 0 1em;
    line-height: 1.65;
    color: #222;
  }}
  h1, h2, h3, h4 {{ color: #111; margin-top: 1.6em; }}
  h1 {{ border-bottom: 1px solid #ddd; padding-bottom: 0.3em; }}
  h2 {{ border-bottom: 1px solid #eee; padding-bottom: 0.2em; }}
  blockquote {{
    border-left: 3px solid #ddd;
    padding-left: 1em;
    color: #555;
    margin-left: 0;
  }}
  code {{
    background: #f4f4f4;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: "JetBrains Mono", Consolas, "Source Code Pro", monospace;
    font-size: 0.9em;
  }}
  pre {{
    background: #f4f4f4;
    padding: 1em;
    overflow: auto;
    border-radius: 4px;
  }}
  pre code {{ background: transparent; padding: 0; }}
  table {{ border-collapse: collapse; margin: 1em 0; }}
  th, td {{ border: 1px solid #ddd; padding: 4px 10px; }}
  th {{ background: #fafafa; }}
  a {{ color: #0366d6; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
<script>
  window.MathJax = {{
    tex: {{
      inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
      displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
    }}
  }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
</head>
<body>
{body}
</body>
</html>
"""


def render_html(title: str, body_md: str) -> str:
    """Convert (title, markdown body) → standalone HTML page string."""
    body_md_with_links = _expand_wikilinks(body_md)
    body_html = _md.markdown(
        body_md_with_links,
        extensions=["extra", "sane_lists", "smarty"],
        output_format="html5",
    )
    # Escape only literal `{` / `}` in title to avoid str.format conflicts
    safe_title = title.replace("{", "{{").replace("}", "}}")
    return _HTML_TEMPLATE.format(title=safe_title, body=body_html)
