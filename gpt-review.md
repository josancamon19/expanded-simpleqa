Here are the most likely “red flags” a reviewer / lab PI will point out, and why they matter.

### 1) **Automatic KGs can be wrong → your QA becomes wrong**

Your whole pipeline inherits errors from graph extraction. KGGen explicitly motivates itself by noting that *automatically extracted KGs can be of questionable quality*, even if KGGen improves sparsity via clustering. If the KG invents/merges entities or relations, you’ll synthesize clean-looking but false QAs. ([arXiv][1])

### 2) **“Single indisputable answer” is *hard* to guarantee at scale**

SimpleQA’s core design constraint is **one indisputable answer** and easy grading; achieving that reliably required careful dataset construction choices (and later “Verified” exists because reliability is fragile). At 50k, ambiguity (aliases, date versions, “current” vs “former”, unit conventions, alternate spellings) will explode unless the verification protocol is extremely strict. ([arXiv][2])

### 3) **LLM-as-judge is not a free lunch (biases, inconsistency, self-preference)**

A common criticism: using an LLM to “verify” QAs can create **systematic judge bias** (including favoring its own style/answers), inconsistencies, and evaluation leakage. There’s substantial literature warning that LLM judges can be biased and brittle, and offering “principles/guidelines” precisely because naive LLM-judging can be misleading. ([arXiv][3])

### 4) **Self-referential evaluation risk**

If the *same* family of models (or prompts) is used to:

* extract the KG,
* generate questions,
* judge validity,
* and then evaluate candidate systems,
  you can get **self-reinforcing** benchmarks where models look good for the wrong reasons (judge agrees with generator’s artifacts). This is a known failure mode in LLM-judge setups. ([Computer Science][4])

### 5) **Novelty / positioning: “Isn’t this STaRK + Wikipedia?”**

STaRK already contributes a **pipeline to synthesize realistic queries** that integrate relational + textual constraints, and includes human evaluation / human-authored queries to validate quality. A skeptical reviewer may say your work is primarily a domain swap (Wikipedia) and a different graph builder (KGGen), unless you isolate what is fundamentally new (e.g., reliability protocol, error analysis, or a new task definition). ([arXiv][5])

### 6) **Question distribution might be “templated” and not challenging**

SimpleQA is intentionally **adversarially collected** (against GPT-4) to be challenging, not just large. A graph-driven generator can overproduce easy relation-lookup questions (“X’s birthplace”, “Y’s spouse”), inflating size without increasing evaluative value. ([arXiv][2])

### 7) **Wikipedia grounding can be time-variant and version-sensitive**

Wikipedia changes. If you don’t lock dumps + timestamps, questions like “current CEO”, “most recent”, or updated numbers become invalid. This becomes a reproducibility and dataset “half-life” issue at 50k scale.

### 8) **Human review cost may be larger than it sounds**

“Light human review” over 50k items is still serious labor if reliability targets are high (SimpleQA Verified exists for a reason). Reviewers may challenge the realism of the promised cost/time unless you specify sampling strategies, double-annotation rates, and measured inter-annotator agreement. ([ResearchGate][6])

### 9) **The two proposed experiments may look bolted-on**

Direct prompting vs graph-mediated is clean. But “SFT vs KL memorization approaches” might read as a *separate* research thread unless you argue clearly why this dataset uniquely enables that comparison (and why existing factuality sets aren’t enough).

### 10) **Metric / evaluation protocol ambiguity**

“How do we score?” If you rely on exact-match short answers, you’ll get alias issues; if you allow soft matching, you risk judge artifacts. SimpleQA leaned on discrete grading (“correct/incorrect/not attempted”) precisely to avoid fuzzy scoring debates. You’ll be expected to specify scoring rules tightly. ([arXiv][2])

---

If you want, paste your current “1-page pitch” (or just bullets), and I’ll rewrite it to **pre-empt these critiques** (i.e., make the novelty + reliability story reviewer-proof) while keeping it short enough for a lab approval decision.

[1]: https://arxiv.org/pdf/2502.09956?utm_source=chatgpt.com "KGGen: Extracting Knowledge Graphs from Plain Text with ..."
[2]: https://arxiv.org/abs/2411.04368?utm_source=chatgpt.com "Measuring short-form factuality in large language models"
[3]: https://arxiv.org/abs/2411.15594?utm_source=chatgpt.com "[2411.15594] A Survey on LLM-as-a-Judge"
[4]: https://www.cs.unh.edu/~dietz/papers/dietz2025principles.pdf?utm_source=chatgpt.com "Principles and Guidelines for the Use of LLM Judges"
[5]: https://arxiv.org/abs/2404.13207?utm_source=chatgpt.com "STaRK: Benchmarking LLM Retrieval on Textual and ..."
[6]: https://www.researchgate.net/publication/395387806_SimpleQA_Verified_A_Reliable_Factuality_Benchmark_to_Measure_Parametric_Knowledge?utm_source=chatgpt.com "(PDF) SimpleQA Verified: A Reliable Factuality Benchmark ..."
