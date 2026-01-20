<details>
<summary><strong>Background: SimpleQA & SimpleQA Verified</strong></summary>

>**SimpleQA** is a 4,326-question factual accuracy benchmark created by OpenAI where questions are adversarially collected against GPT-4, primarily sourced from Wikipedia and other web sources
>
>**SimpleQA Verified** refines this through a rigorous multi-stage filtering pipeline that reduces the dataset to 1,000 high-quality questions by removing duplicate sources, applying semantic and TF-IDF de-duplication (using 0.77 cosine similarity cutoff), balancing topic and answer-type distributions, filtering for maximum difficulty (keeping only questions that all frontier models answer incorrectly), and conducting manual review for URL cleaning, source quality verification, date precision, and metadata enrichment (identifying 3.7% requiring reasoning and 7.3% being multi-step)
>
>Both benchmarks evaluate models using a three-tier grading system (correct/incorrect/not attempted) with LLM-as-judge evaluation, revealing that even state-of-the-art models are close to saturation [SimpleQA-Verified Leaderboard](https://www.kaggle.com/benchmarks/deepmind/simpleqa-verified) with Gemini-3-Pro at 72%.
>
>**Issues**
>1. Limited multi-hop reasoning: Only 7.3% are multi-step questions identified post-hoc by classifier, with no principled control over reasoning depth. [data](https://www.kaggle.com/datasets/deepmind/simpleqa-verified/data)
>2. Static and non-updatable: Manual curation makes updates expensive and slow, causing temporal facts to become stale over time
>3. Small scale after filtering: SimpleQA Verified's rigorous quality control reduces dataset to only 1,000 questions, limiting statistical power and coverage
>4. No diagnostic capabilities: Provides aggregate accuracy scores without interpretable failure analysis across knowledge types or reasoning patterns
>5. Evals are close to saturation

</details>

<details>
<summary><strong>KG to Factual QA Generation</strong></summary>

>The following are some of the available methods for creating factual QA Pairs from knowledge graphs
>
>- **Knowledge Questions from Knowledge Graphs** (Oct 2016) - [arXiv](https://arxiv.org/pdf/1610.09935)
>  *KG Input:* DBpedia triples (entity + relation + entity). Selects named entity as answer, constructs triple-pattern query, uses templates to verbalize into natural language.
>  *Focus:* Quiz-style multiple-choice generation with distractor selection and difficulty estimation via Jeopardy! data.
>
>- **KGLens** (Dec 2023) - [arXiv](https://arxiv.org/pdf/2312.11539)
>  *KG Input:* Domain-specific KG edges (19K+ edges across 3 KBs). Uses graph-guided question generator with importance sampling based on graph structure.
>  *Focus:* LLM knowledge probing and factuality assessment—achieves 95.7% accuracy vs human annotators.
>
>- **FactChecker: The Earth is Flat?** (Jan 2024) - [arXiv](https://arxiv.org/pdf/2401.00761)
>  *KG Input:* Fact triplets from large-scale knowledge databases. Rule-based generation of Yes-No, Multiple-Choice, and WH questions for single-hop and multi-hop relations.
>  *Focus:* Detecting LLM factual errors (up to 45% error rates found); test cases can fine-tune models to improve accuracy.
>
>- **STaRK** (Apr 2024, NeurIPS 2024) - [arXiv](https://arxiv.org/pdf/2404.13207)
>  *KG Input:* Semi-structured KBs combining textual properties + relational structure. Synthesizes queries integrating relational info with complex text properties.
>  *Focus:* LLM retrieval benchmark across product search, academic papers, and precision medicine domains.
>
>- **ECKGBench** (Mar 2025) - [arXiv](https://arxiv.org/pdf/2503.15990)
>  *KG Input:* Large-scale e-commerce KG with standardized automated question generation workflow.
>  *Focus:* E-commerce LLM evaluation with emphasis on hallucination detection; includes human annotation and verification.
>
>- **KGQuest** (Nov 2025) - [arXiv](https://arxiv.org/pdf/2511.11258)
>  *KG Input:* KG triplets clustered by relation type → template derivation using entity/relation type rules → LLM refinement for linguistic quality.
>  *Focus:* Scalable, deterministic pipeline balancing factual accuracy with natural language fluency via LLM polishing.

</details>

### Proposed Method

1. Create an article selection pipeline based on topic, popularity, length, etc.
2. Use [KGGen](https://github.com/stair-lab/kg-gen) to generate knowledge graphs from those articles
3. Potentially include human sampling verification here
4. Use KGQuest or FactChecker for QA generation across multiple categories and n-hop

### Advantages Over SimpleQA

| Problem | Our Solution |
|---------|--------------|
| Limited multi-hop reasoning | Graph traversal naturally generates questions of varying reasoning depth (1-hop, 2-hop, n-hop) by design |
| Static and non-updatable | Automated pipeline allows continuous updates as source articles change |
| Small scale (1,000 questions) | Generate 10K+ questions with comparable quality through automated validation |
| No diagnostic capabilities | KG structure enables stratified sampling across entity types, relation types, and knowledge domains |
| Close to saturation (72%) | Target out-of-distribution results significantly different from current benchmarks |
