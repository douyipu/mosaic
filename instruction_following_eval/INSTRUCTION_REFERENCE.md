# IFEval 指令类型与参数参考

本文档整理了 `input_data.jsonl` 中所有 `instruction_id_list` 及其对应的 `kwargs` 参数说明。

---

## 1. change_case（大小写相关）

### change_case:capital_word_frequency
全大写单词出现频率约束。

| 参数 | 类型 | 说明 |
|------|------|------|
| capital_relation | str | `"at least"` 或 `"less than"` |
| capital_frequency | int | 次数阈值 |

**kwargs 示例**：`{"capital_relation": "less than", "capital_frequency": 10}`

---

### change_case:english_capital
回复必须全大写（英文）。

**kwargs**：`{}`（无参数）

---

### change_case:english_lowercase
回复必须全小写（英文）。

**kwargs**：`{}`（无参数）

---

## 2. combination（组合指令）

### combination:repeat_prompt
先逐字复述 prompt，再给出回答。

| 参数 | 类型 | 说明 |
|------|------|------|
| prompt_to_repeat | str | 需要复述的 prompt 原文 |

**kwargs 示例**：`{"prompt_to_repeat": "Write an email to my boss..."}`

---

### combination:two_responses
回复需包含两个独立回答，用 `******`（6 个星号）分隔。

**kwargs**：`{}`（无参数）

---

## 3. detectable_content（可检测内容）

### detectable_content:number_placeholders
回复中需包含指定数量的占位符（如 `[address]`、`[name]`）。

| 参数 | 类型 | 说明 |
|------|------|------|
| num_placeholders | int | 占位符数量（1–20） |

**kwargs 示例**：`{"num_placeholders": 12}`

---

### detectable_content:postscript
回复需包含后记（P.S. 或 P.P.S）。

| 参数 | 类型 | 说明 |
|------|------|------|
| postscript_marker | str | `"P.S."` 或 `"P.P.S"` |

---

## 4. detectable_format（可检测格式）

### detectable_format:constrained_response
回复需符合特定格式约束（具体由 prompt 定义）。

**kwargs**：`{}`（无参数）

---

### detectable_format:json_format
回复需为合法 JSON 格式。

**kwargs**：`{}`（无参数）

---

### detectable_format:multiple_sections
回复需包含多个带标题的段落。

| 参数 | 类型 | 说明 |
|------|------|------|
| section_spliter | str | 段落标题前缀，如 `"PARAGRAPH"`、`"SECTION"`、`"Day"` |
| num_sections | int | 段落数量（2–7） |

**kwargs 示例**：`{"section_spliter": "PARAGRAPH", "num_sections": 2}`

---

### detectable_format:number_bullet_lists
回复需包含指定数量的 Markdown 列表项（`* ` 开头）。

| 参数 | 类型 | 说明 |
|------|------|------|
| num_bullets | int | 列表项数量（1–10） |

---

### detectable_format:number_highlighted_sections
回复需包含指定数量的高亮段落（`*...*` 包裹）。

| 参数 | 类型 | 说明 |
|------|------|------|
| num_highlights | int | 高亮段落数量（1–15） |

---

### detectable_format:title
回复需包含用双尖括号包裹的标题，如 `<<title>>`。

**kwargs**：`{}`（无参数）

---

## 5. keywords（关键词相关）

### keywords:existence
回复需包含指定关键词（全部出现）。

| 参数 | 类型 | 说明 |
|------|------|------|
| keywords | list[str] | 必须出现的词列表 |

**kwargs 示例**：`{"keywords": ["correlated", "experiencing"]}`

---

### keywords:forbidden_words
回复不得包含指定词。

| 参数 | 类型 | 说明 |
|------|------|------|
| forbidden_words | list[str] | 禁止出现的词列表 |

---

### keywords:frequency
某关键词出现次数需满足约束。

| 参数 | 类型 | 说明 |
|------|------|------|
| relation | str | `"at least"` 或 `"less than"` |
| keyword | str | 目标词 |
| frequency | int | 次数阈值 |

**kwargs 示例**：`{"relation": "at least", "keyword": "brand", "frequency": 2}`

---

### keywords:letter_frequency
某字符（如 `#`、`!`、字母）出现次数需满足约束。

| 参数 | 类型 | 说明 |
|------|------|------|
| let_relation | str | `"at least"` 或 `"less than"` |
| letter | str | 目标字符 |
| let_frequency | int | 次数阈值 |

**kwargs 示例**：`{"let_relation": "at least", "letter": "#", "let_frequency": 4}`

---

## 6. language（语言）

### language:response_language
回复需使用指定语言。

| 参数 | 类型 | 说明 |
|------|------|------|
| language | str | 语言代码：`ar`、`bn`、`bg`、`de`、`fa`、`fi`、`gu`、`hi`、`it`、`kn`、`ko`、`mr`、`ne`、`pa`、`pt`、`ru`、`sw`、`ta`、`te`、`th`、`ur`、`vi` 等 |

---

## 7. length_constraints（长度约束）

### length_constraints:number_paragraphs
段落数量约束。

| 参数 | 类型 | 说明 |
|------|------|------|
| num_paragraphs | int | 段落数量（2–6） |

---

### length_constraints:number_sentences
句子数量约束。

| 参数 | 类型 | 说明 |
|------|------|------|
| relation | str | `"at least"` 或 `"less than"` |
| num_sentences | int | 句子数量 |

**kwargs 示例**：`{"relation": "less than", "num_sentences": 45}`

---

### length_constraints:number_words
词数约束。

| 参数 | 类型 | 说明 |
|------|------|------|
| relation | str | `"at least"` 或 `"less than"` |
| num_words | int | 词数 |

**kwargs 示例**：`{"relation": "at least", "num_words": 300}`

---

### length_constraints:nth_paragraph_first_word
第 N 段的首词需为指定词。

| 参数 | 类型 | 说明 |
|------|------|------|
| first_word | str | 期望的首词 |
| num_paragraphs | int | 总段落数 |
| nth_paragraph | int | 第几段（1-based） |

**kwargs 示例**：`{"first_word": "weekend", "num_paragraphs": 4, "nth_paragraph": 1}`

---

## 8. punctuation（标点）

### punctuation:no_comma
回复不得包含逗号。

**kwargs**：`{}`（无参数）

---

## 9. startend（起止）

### startend:end_checker
回复需以指定短语结尾。

| 参数 | 类型 | 说明 |
|------|------|------|
| end_phrase | str | 必须出现的结尾短语 |

**kwargs 示例**：`{"end_phrase": "Can I get my money back for the classes I missed?"}`

---

### startend:quotation
回复需用双引号包裹（首尾各一个 `"`）。

**kwargs**：`{}`（无参数）

---

## 参数取值汇总

| 参数名 | 常见取值 |
|--------|----------|
| relation / capital_relation / let_relation | `"at least"`, `"less than"` |
| num_words | 20–1200 |
| num_sentences | 1–100 |
| num_paragraphs | 2–7 |
| num_bullets | 1–10 |
| num_highlights | 1–15 |
| num_placeholders | 1–20 |
| num_sections | 2–7 |
| section_spliter | `"PARAGRAPH"`, `"SECTION"`, `"Section"`, `"Day"`, `"Audience"` |
| postscript_marker | `"P.S."`, `"P.P.S"` |
| language | ISO 639-1 代码（如 `hi`、`ko`、`ar`） |
