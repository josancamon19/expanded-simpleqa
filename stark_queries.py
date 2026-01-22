"""
STARK-style query generation for knowledge graphs.

Generates fact-seeking queries following the STARK benchmark methodology:
1. Sample a target entity (gold answer) from the graph
2. Extract relational constraints involving that entity
3. Use LLM to synthesize natural queries combining relational + textual requirements
4. Output SimpleQA-style questions with single, verifiable answers

Reference: STaRK: Benchmarking LLM Retrieval on Textual and Relational Knowledge Bases
(https://arxiv.org/abs/2404.13207)
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion

load_dotenv()

GRAPHS_DIR = Path("graphs")


@dataclass
class Query:
    """A generated query with its gold answer."""
    question: str
    answer: str
    answer_entity: str
    supporting_relations: list[tuple[str, str, str]]
    query_type: str
    graph_idx: int


def load_graph(graph_idx: int) -> dict:
    """Load a knowledge graph by index."""
    graph_path = GRAPHS_DIR / f"graph_{graph_idx}.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    with open(graph_path) as f:
        return json.load(f)


def load_article(graph_idx: int) -> str | None:
    """Load the source article text for a graph."""
    article_path = GRAPHS_DIR / f"article_{graph_idx}.txt"
    if article_path.exists():
        with open(article_path) as f:
            return f.read()
    return None


def get_entity_relations(graph: dict, entity: str) -> dict:
    """Get all relations where entity appears as subject or object."""
    as_subject = []
    as_object = []

    for rel in graph["relations"]:
        subj, pred, obj = rel
        if subj == entity:
            as_subject.append((subj, pred, obj))
        if obj == entity:
            as_object.append((subj, pred, obj))

    return {"as_subject": as_subject, "as_object": as_object}


def select_answer_entity(graph: dict, min_relations: int = 3) -> tuple[str, dict]:
    """
    Select a candidate answer entity that has sufficient relational context.
    Following STARK: choose entities with rich relational information.
    Requires more relations (3+) to ensure specificity.
    """
    entities = graph["entities"]
    random.shuffle(entities)

    for entity in entities:
        relations = get_entity_relations(graph, entity)
        total_rels = len(relations["as_subject"]) + len(relations["as_object"])
        if total_rels >= min_relations:
            return entity, relations

    # Fallback: return entity with most relations
    best_entity = None
    best_count = 0
    best_relations = {"as_subject": [], "as_object": []}
    for entity in entities:
        relations = get_entity_relations(graph, entity)
        count = len(relations["as_subject"]) + len(relations["as_object"])
        if count > best_count:
            best_count = count
            best_entity = entity
            best_relations = relations

    return best_entity, best_relations


# SimpleQA-style few-shot examples for the prompt
SIMPLEQA_EXAMPLES = """
Example SimpleQA questions (these are the QUALITY we want):

Q: Who received the IEEE Frank Rosenblatt Award in 2010?
A: Michio Sugeno

Q: What is the name of the former Prime Minister of Iceland who worked as a cabin crew member until 1971?
A: Jóhanna Sigurðardóttir

Q: Who won the 1991 Ig Nobel Prize for Peace?
A: Edward Teller

Q: In which year did Frida Kahlo's first solo exhibit open?
A: 1938

Q: Who was the leader of the woman's suffrage movement who was born in 1847?
A: Victoria Woodhull

Q: What signature piece did Scott Wilson discover on the curb that became part of MOBA's collection?
A: Lucy in the Field with Flowers

Notice how each question contains ENOUGH SPECIFIC DETAILS that only ONE answer is possible.
Bad example: "Who was born in 1847?" - TOO VAGUE, many people were born then.
Good example: "Who was born in 1847 and led the woman's suffrage movement?" - SPECIFIC enough.
"""


QUERY_GENERATION_PROMPT = """You are generating fact-seeking questions for a QA benchmark similar to SimpleQA.
Each question MUST have exactly ONE correct, unambiguous answer.

{simpleqa_examples}

---

SOURCE ARTICLE (for context):
{article_excerpt}

---

TARGET ANSWER: {answer_entity}

RELATIONAL FACTS about this entity from the knowledge graph:
{relations_text}

---

Generate {num_queries} HIGH-QUALITY questions where the answer is exactly "{answer_entity}".

CRITICAL REQUIREMENTS:
1. Each question must contain ENOUGH SPECIFIC CONSTRAINTS that ONLY the target answer fits
2. Combine multiple facts: birth year + profession + achievement, OR role + organization + time period, etc.
3. Questions should sound natural, like a curious person asking
4. The answer should be a short entity name, not a full sentence
5. Do NOT mention the answer in the question
6. Avoid vague questions - if someone knowledgeable couldn't uniquely identify the answer, the question is BAD
7. Use specific years (not "mid-19th century"), specific roles, specific achievements

Output ONLY a JSON array, no other text:
[
  {{"question": "...", "specificity_check": "why this question uniquely identifies the answer"}}
]
"""


def generate_queries_for_entity(
    answer_entity: str,
    relations: list[tuple],
    article_text: str | None,
    num_queries: int = 1,
    model: str = "gpt-4o-mini"
) -> list[dict]:
    """
    Use LLM to generate natural language queries for a target entity.
    Includes article context and SimpleQA examples for better quality.
    """
    # Format relations as readable text
    relations_text = "\n".join([
        f"- {subj} --[{pred}]--> {obj}"
        for subj, pred, obj in relations
    ])

    # Use article excerpt if available (first 2000 chars for context)
    if article_text:
        article_excerpt = article_text[:2000] + "..." if len(article_text) > 2000 else article_text
    else:
        article_excerpt = "(Article text not available)"

    prompt = QUERY_GENERATION_PROMPT.format(
        simpleqa_examples=SIMPLEQA_EXAMPLES,
        article_excerpt=article_excerpt,
        answer_entity=answer_entity,
        relations_text=relations_text,
        num_queries=num_queries
    )

    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    content = response.choices[0].message.content

    # Parse JSON from response
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    try:
        queries = json.loads(content.strip())
    except json.JSONDecodeError:
        import re
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            queries = json.loads(match.group())
        else:
            queries = []

    return queries


def generate_queries_for_graph(
    graph_idx: int,
    num_queries: int = 10,
    model: str = "gpt-4o-mini"
) -> list[Query]:
    """
    Generate STARK-style queries for a knowledge graph.

    Args:
        graph_idx: Index of the graph to use (e.g., 0 for graph_0.json)
        num_queries: Target number of queries to generate (~10)
        model: LiteLLM model identifier for query generation

    Returns:
        List of Query objects with questions and gold answers
    """
    graph = load_graph(graph_idx)
    article_text = load_article(graph_idx)

    if article_text:
        print(f"  Loaded article text ({len(article_text)} chars)")
    else:
        print("  Warning: No article text available")

    queries = []
    used_entities = set()  # Track used entities to ensure diversity
    attempts = 0
    max_attempts = num_queries * 5

    while len(queries) < num_queries and attempts < max_attempts:
        attempts += 1

        # Step 1: Sample answer entity (STARK methodology)
        answer_entity, relations = select_answer_entity(graph, min_relations=3)

        if not answer_entity or answer_entity in used_entities:
            continue

        # Step 2: Get all relations for this entity
        all_relations = relations["as_subject"] + relations["as_object"]

        if len(all_relations) < 2:
            continue

        # Step 3: Generate ONE query per entity for diversity
        try:
            generated = generate_queries_for_entity(
                answer_entity=answer_entity,
                relations=all_relations,
                article_text=article_text,
                num_queries=1,  # One query per entity
                model=model
            )

            for q in generated:
                if len(queries) >= num_queries:
                    break

                # Determine query type
                predicates = [r[1] for r in all_relations]
                if any("born" in p.lower() for p in predicates):
                    query_type = "biographical"
                elif any("member" in p.lower() or "held by" in p.lower() for p in predicates):
                    query_type = "affiliation"
                elif any("located" in p.lower() for p in predicates):
                    query_type = "geographical"
                else:
                    query_type = "factual"

                queries.append(Query(
                    question=q["question"],
                    answer=answer_entity,
                    answer_entity=answer_entity,
                    supporting_relations=all_relations,
                    query_type=query_type,
                    graph_idx=graph_idx
                ))
                used_entities.add(answer_entity)
                print(f"  [{len(queries)}/{num_queries}] {answer_entity}")

        except Exception as e:
            print(f"  Warning: Query generation failed for {answer_entity}: {e}")
            continue

    return queries


def queries_to_simpleqa_format(queries: list[Query]) -> list[dict]:
    """Convert queries to SimpleQA-compatible format."""
    return [
        {
            "question": q.question,
            "answer": q.answer,
            "metadata": {
                "query_type": q.query_type,
                "graph_idx": q.graph_idx,
                "supporting_facts": [
                    {"subject": s, "predicate": p, "object": o}
                    for s, p, o in q.supporting_relations
                ]
            }
        }
        for q in queries
    ]


def main():
    """Generate queries for all available graphs."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate STARK-style queries from knowledge graphs")
    parser.add_argument("--graph-idx", type=int, help="Specific graph index to process")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of queries per graph")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LiteLLM model for generation")
    parser.add_argument("--output", type=str, default="queries.json", help="Output file path")
    args = parser.parse_args()

    all_queries = []

    if args.graph_idx is not None:
        graph_indices = [args.graph_idx]
    else:
        graph_files = list(GRAPHS_DIR.glob("graph_*.json"))
        graph_indices = sorted([
            int(f.stem.split("_")[1])
            for f in graph_files
            if f.stem != "combined_graph"
        ])

    for idx in graph_indices:
        print(f"\nGenerating queries for graph_{idx}...")
        try:
            queries = generate_queries_for_graph(
                graph_idx=idx,
                num_queries=args.num_queries,
                model=args.model
            )
            all_queries.extend(queries)
            print(f"  Total: {len(queries)} queries")
        except Exception as e:
            print(f"  Error processing graph_{idx}: {e}")

    # Convert to SimpleQA format and save
    output_data = queries_to_simpleqa_format(all_queries)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved {len(output_data)} queries to {output_path}")
    print(f"{'='*60}")

    # Print all queries
    if output_data:
        print("\nGenerated queries:")
        for i, q in enumerate(output_data, 1):
            print(f"\n{i}. Q: {q['question']}")
            print(f"   A: {q['answer']}")


if __name__ == "__main__":
    main()
