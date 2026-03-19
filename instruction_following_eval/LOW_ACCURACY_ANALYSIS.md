# 三个低准确率指令的根因分析

## 1. startend:end_checker（1.7%）→ **主要是 kwargs 抽错**

### 数据
- 总样本：2486
- **成功抽取 end_phrase：56（2.3%）**
- **kwargs 为空：2430（97.7%）**

### 原因
当 `kwargs` 为空时，IFEval 的 `EndChecker` 使用**随机默认值**：
```python
# instructions.py
_ENDING_OPTIONS = ("Any other questions?", "Is there anything else I can help with?")
# 当 end_phrase 为 None 时：self._end_phrase = random.choice(_ENDING_OPTIONS)
```

也就是说，97.7% 的样本在用「Any other questions?」或「Is there anything else I can help with?」去检查结尾，而 Tulu 的 prompt 要求的结尾句各不相同，回复几乎不可能以这两句结尾。

### 抽样验证
| 样本 | prompt 要求的结尾 | 实际 response 结尾 | kwargs 抽取 |
|------|-------------------|--------------------|-------------|
| 1 | "Together, we can secure a sustainable future for all." | ✅ 一致（在 JSON 内） | ❌ 空 |
| 2 | "Each summary should end with a quote"（无固定句） | 模型拒绝任务 | ❌ 空 |
| 3 | "And in the flickering candlelight, he whispered, 'Is this the path to redemption or ruin?'" | ✅ 完全一致 | ❌ 空 |

**结论**：样本 1、3 中模型都按要求结尾，但因 `end_phrase` 未抽取，被错误判为失败。**根因是 kwargs 抽取失败，不是模型未遵守。**

---

## 2. combination:repeat_prompt（7.1%）→ **主要是模型没遵守**

### 抽样验证
| 样本 | prompt_to_repeat（应逐字复述） | response 开头 | 是否遵守 |
|------|-------------------------------|---------------|----------|
| 1 | "Please integrate... How has the prison education system..." | "The question at hand is: How has the prison education system..." | ❌ 改写，非逐字 |
| 2 | "Imagine you are a game developer... What would a dragon say..." | "What would a dragon say... Imagine you are a game developer..." | ❌ 顺序错，且非逐字 |
| 3 | "Explain the benefits of COVID-19 vaccinations..." | "The COVID-19 vaccinations have been instrumental..." | ❌ 直接回答，未复述 |

### 结论
模型普遍**没有逐字复述** prompt，而是改写、调序或直接回答。**根因是模型未遵守，不是 kwargs 抽错。**

---

## 3. detectable_format:constrained_response（0%）→ **约束映射错误**

### 原因
IFEval 的 `ConstrainedResponseChecker` **写死**了三个选项：
```python
_CONSTRAINED_RESPONSE_OPTIONS = ("My answer is yes.", "My answer is no.", "My answer is maybe.")
```

而 Tulu 的 `format:choose one from options` 中，**每个 prompt 的选项都不同**，例如：
- "Choose from: ('500', '1,000', '1,500')"
- "Your answer must contain one of: 'Absolutely, without a doubt', 'Not at all likely', or 'Possibly, but not certain'"

### 抽样验证
| 样本 | prompt 的选项 | response | IFEval 检查的选项 |
|------|---------------|----------|--------------------|
| 1 | 要求结尾 "Amen to that!"（非选项类） | ✅ 以 "Amen to that!" 结尾 | "My answer is yes/no/maybe" |
| 2 | ('500', '1,000', '1,500') | 选了 1,500 | "My answer is yes/no/maybe" |
| 3 | "Absolutely, without a doubt" 等 | 以 "Absolutely, without a doubt" 开头 | "My answer is yes/no/maybe" |

### 结论
Tulu 的「从给定选项中选择」与 IFEval 的「固定 yes/no/maybe」**语义不兼容**。模型在样本 2、3 中其实遵守了 prompt，但被 IFEval 的固定选项判为失败。**根因是约束映射错误，不应把 Tulu 的 format:choose one from options 映射到 constrained_response。**

---

## 总结

| 指令 | 根因 | 建议 |
|------|------|------|
| **startend:end_checker** | kwargs 抽取失败（97.7% 为空） | 改进 end_phrase 的正则抽取 |
| **combination:repeat_prompt** | 模型未逐字复述 | 无需改评估逻辑，属模型能力问题 |
| **detectable_format:constrained_response** | 约束映射不匹配 | 不再映射到 constrained_response，或实现支持自定义选项的 checker |
