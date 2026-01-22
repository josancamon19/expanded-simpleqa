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
import os
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


def select_answer_entity(graph: dict, min_relations: int = 2) -> tuple[str, dict]:
    """
    Select a candidate answer entity that has sufficient relational context.
    Following STARK: choose entities with rich relational information.
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
    for entity in entities:
        relations = get_entity_relations(graph, entity)
        count = len(relations["as_subject"]) + len(relations["as_object"])
        if count > best_count:
            best_count = count
            best_entity = entity
            best_relations = relations

    return best_entity, best_relations


def build_context_for_query(
    answer_entity: str,
    relations: dict,
    max_relations: int = 5
) -> tuple[list[tuple], str]:
    """
    Build relational context for query generation.
    Returns selected relations and a query type hint.
    """
    all_rels = relations["as_subject"] + relations["as_object"]

    if len(all_rels) > max_relations:
        selected = random.sample(all_rels, max_relations)
    else:
        selected = all_rels

    # Determine query type based on relation patterns
    predicates = [r[1] for r in selected]

    if any("born" in p.lower() for p in predicates):
        query_type = "biographical"
    elif any("member" in p.lower() or "held by" in p.lower() for p in predicates):
        query_type = "affiliation"
    elif any("located" in p.lower() or "in" == p.lower() for p in predicates):
        query_type = "geographical"
    elif any("founded" in p.lower() or "established" in p.lower() for p in predicates):
        query_type = "historical"
    else:
        query_type = "factual"

    return selected, query_type


QUERY_GENERATION_PROMPT = """You are generating fact-seeking questions for a knowledge graph QA benchmark,
similar to SimpleQA. Each question must have exactly ONE correct, unambiguous answer.

Target Answer Entity: {answer_entity}

Relational Facts about this entity:
{relations_text}

Generate {num_queries} different natural language questions where the answer is exactly "{answer_entity}".

Requirements:
1. Questions should be specific and unambiguous - only ONE answer is correct
2. Questions should sound natural, like real user queries
3. Use the relational facts to create questions that require knowledge of relationships
4. Vary question styles: "Who...", "What...", "Which person/organization/place..."
5. Questions should be answerable with just the entity name, not a full sentence
6. Do NOT mention the answer in the question
7. Make questions that would be challenging but fair - they require specific knowledge

Output format (JSON array):
[
  {{"question": "...", "reasoning": "brief note on which relations this uses"}},
  ...
]

Generate exactly {num_queries} questions."""


def generate_queries_for_entity(
    answer_entity: str,
    relations: list[tuple],
    num_queries: int = 3,
    model: str = "gpt-4o-mini"
) -> list[dict]:
    """
    Use LLM to generate natural language queries for a target entity.
    Following STARK's approach of combining relational constraints into natural queries.
    """
    # Format relations as readable text
    relations_text = "\n".join([
        f"- {subj} --[{pred}]--> {obj}"
        for subj, pred, obj in relations
    ])

    prompt = QUERY_GENERATION_PROMPT.format(
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
    # Handle potential markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    try:
        queries = json.loads(content.strip())
    except json.JSONDecodeError:
        # Fallback: try to extract array
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
    queries = []
    attempts = 0
    max_attempts = num_queries * 3  # Allow retries for failed generations

    while len(queries) < num_queries and attempts < max_attempts:
        attempts += 1

        # Step 1: Sample answer entity (STARK methodology)
        answer_entity, relations = select_answer_entity(graph, min_relations=2)

        if not answer_entity:
            continue

        # Step 2: Build relational context
        selected_relations, query_type = build_context_for_query(
            answer_entity, relations, max_relations=5
        )

        if not selected_relations:
            continue

        # Step 3: Generate queries via LLM
        # Generate fewer per entity to get diversity
        queries_per_entity = min(3, num_queries - len(queries))

        try:
            generated = generate_queries_for_entity(
                answer_entity=answer_entity,
                relations=selected_relations,
                num_queries=queries_per_entity,
                model=model
            )

            for q in generated:
                if len(queries) >= num_queries:
                    break

                queries.append(Query(
                    question=q["question"],
                    answer=answer_entity,
                    answer_entity=answer_entity,
                    supporting_relations=selected_relations,
                    query_type=query_type,
                    graph_idx=graph_idx
                ))

        except Exception as e:
            print(f"  Warning: Query generation failed for {answer_entity}: {e}")
            continue

    return queries


def queries_to_simpleqa_format(queries: list[Query]) -> list[dict]:
    """
    Convert queries to SimpleQA-compatible format.

    SimpleQA format:
    - question: The question text
    - answer: The gold answer (single, unambiguous)
    - metadata: Additional context
    """
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
        # Process single graph
        graph_indices = [args.graph_idx]
    else:
        # Process all available graphs
        graph_files = list(GRAPHS_DIR.glob("graph_*.json"))
        graph_indices = sorted([
            int(f.stem.split("_")[1])
            for f in graph_files
            if f.stem != "combined_graph"
        ])

    for idx in graph_indices:
        print(f"Generating queries for graph_{idx}...")
        try:
            queries = generate_queries_for_graph(
                graph_idx=idx,
                num_queries=args.num_queries,
                model=args.model
            )
            all_queries.extend(queries)
            print(f"  Generated {len(queries)} queries")
        except Exception as e:
            print(f"  Error processing graph_{idx}: {e}")

    # Convert to SimpleQA format and save
    output_data = queries_to_simpleqa_format(all_queries)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(output_data)} queries to {output_path}")

    # Print sample
    if output_data:
        print("\nSample queries:")
        for q in output_data[:3]:
            print(f"  Q: {q['question']}")
            print(f"  A: {q['answer']}")
            print()


if __name__ == "__main__":
    main()
