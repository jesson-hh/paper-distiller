You are composing a cluster mini-survey that ties together {n_articles} freshly distilled articles on a shared topic.

# Topic
{topic}

# The articles being surveyed
{articles_block}

# Existing wiki entries — for additional crosslinks beyond the articles above
{wiki_index_block}

# Output

Return strictly one JSON object:

{{
  "title": "综述标题（中文为主）",
  "body": "markdown content",
  "tags": ["标签1", ...],
  "related_articles": [{related_slugs_json}]
}}

The `body` follows this structure:

```
# {{综述标题}}

> Cluster 主题: {{1 句话总结这组论文的共同主线}}
> 包含 {n_articles} 篇 article

## 包含的论文
- [[slug-of-paper-1]] — {{1 句话提要}}
- [[slug-of-paper-2]] — {{...}}
...

## 主线脉络
{{2–4 段：这组论文是怎么串起来的？方法演进？问题递进？}}

## 技术骨架对比
{{优先用 markdown 表格。列：论文 | 核心方法 | 关键结果 | 主要限制}}

## Open issues / 我的 take
{{1–2 段：这个 cluster 还有什么没解决？接下来值得做什么？}}
```

Requirements:
- MUST `[[link]]` every article slug in the articles list (use exactly those slugs — they are fresh entries)
- Tables / comparison matrices preferred over flowing prose
- Body length: 800–2000 Chinese characters
- Do NOT write YAML frontmatter
