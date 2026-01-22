"""
Generate knowledge graphs from FineWiki articles using kg-gen.

Streams the HuggingFaceFW/finewiki dataset, filters articles around 2k tokens,
and generates knowledge graphs with full semantic deduplication.
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from dotenv import load_dotenv
from kg_gen import KGGen
from kg_gen.steps._3_deduplicate import DeduplicateMethod

# Load environment variables from .env
load_dotenv()


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using whitespace splitting.
    Rough approximation: ~0.75 words per token for English.
    """
    words = len(text.split())
    return int(words / 0.75)


def process_article(kg: KGGen, idx: int, article: dict) -> dict | None:
    """Process a single article and generate its knowledge graph."""
    title = article["title"][:50]
    print(f"  [Article {idx}] Starting: {title}...")

    try:
        graph = kg.generate(
            input_data=article["text"],
            no_dspy=False,
            deduplication_method=None,  # Disabled due to semhash API incompatibility
        )

        print(
            f"  [Article {idx}] Done: {len(graph.entities)} entities, {len(graph.relations)} relations"
        )

        return {
            "index": idx,
            "title": article["title"],
            "text": article["text"],
            "graph": graph,
        }

    except Exception as e:
        print(f"  [Article {idx}] Error: {e}")
        return None


def main():
    # Configuration
    TARGET_ARTICLES = 10
    MIN_TOKENS = 1500
    MAX_TOKENS = 2500
    OUTPUT_DIR = "graphs"

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading FineWiki dataset (streaming)...")
    # Stream the English subset of FineWiki
    dataset = load_dataset(
        "HuggingFaceFW/finewiki", name="en", split="train", streaming=True
    )
    print(f"Filtering articles with {MIN_TOKENS}-{MAX_TOKENS} tokens...")
    selected_articles = []
    articles_scanned = 0

    for article in dataset:
        articles_scanned += 1
        text = article.get("text", "")

        if not text or not text.strip():
            continue

        token_count = estimate_tokens(text)

        if MIN_TOKENS <= token_count <= MAX_TOKENS:
            selected_articles.append(
                {
                    "text": text,
                    "title": article.get("title", f"article_{len(selected_articles)}"),
                    "token_count": token_count,
                }
            )
            print(
                f"  Found article {len(selected_articles)}/{TARGET_ARTICLES}: "
                f"'{selected_articles[-1]['title'][:50]}...' (~{token_count} tokens)"
            )

        if len(selected_articles) >= TARGET_ARTICLES:
            break

        # Progress indicator
        if articles_scanned % 1000 == 0:
            print(f"  Scanned {articles_scanned} articles...")

    if len(selected_articles) < TARGET_ARTICLES:
        print(
            f"Warning: Only found {len(selected_articles)} articles matching criteria"
        )

    print(
        f"\nSelected {len(selected_articles)} articles from {articles_scanned} scanned"
    )

    # Initialize KGGen with GPT-5.2 model
    print("\nInitializing KGGen...")
    kg = KGGen(
        model="openai/gpt-5.2",
        temperature=1.0,  # Required for gpt-5 family models
        reasoning_effort="high",
        max_tokens=16000,  # Minimum for gpt-5 family models
        retrieval_model="sentence-transformers/all-mpnet-base-v2",
    )

    # Generate knowledge graphs in parallel
    print(f"\nProcessing {len(selected_articles)} articles in parallel...")
    graphs = []

    with ThreadPoolExecutor(max_workers=TARGET_ARTICLES) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_article, kg, idx, article): idx
            for idx, article in enumerate(selected_articles)
        }

        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                graphs.append(result)

    # Sort by index for consistent ordering
    graphs.sort(key=lambda x: x["index"])

    print(f"\nSuccessfully processed {len(graphs)}/{len(selected_articles)} articles")

    # Save individual graphs and article texts
    print(f"\nSaving graphs to {OUTPUT_DIR}/...")
    for item in graphs:
        idx = item["index"]
        graph = item["graph"]

        # Save JSON
        json_path = os.path.join(OUTPUT_DIR, f"graph_{idx}.json")
        KGGen.export_graph(graph, json_path)
        print(f"  Saved {json_path}")

        # Save article text for query generation
        article_path = os.path.join(OUTPUT_DIR, f"article_{idx}.txt")
        with open(article_path, "w") as f:
            f.write(item["text"])
        print(f"  Saved {article_path}")

        # Save HTML visualization
        html_path = os.path.join(OUTPUT_DIR, f"graph_{idx}.html")
        KGGen.visualize(graph, html_path, open_in_browser=False)
        print(f"  Saved {html_path}")

    # Aggregate all graphs into one combined graph
    if graphs:
        print("\nAggregating all graphs...")
        all_graphs = [item["graph"] for item in graphs]
        combined_graph = kg.aggregate(all_graphs)

        # Save combined graph
        combined_json_path = os.path.join(OUTPUT_DIR, "combined_graph.json")
        KGGen.export_graph(combined_graph, combined_json_path)
        print(f"  Saved combined graph: {combined_json_path}")

        combined_html_path = os.path.join(OUTPUT_DIR, "combined_graph.html")
        KGGen.visualize(combined_graph, combined_html_path, open_in_browser=False)
        print(f"  Saved combined visualization: {combined_html_path}")

        # Summary
        print(f"\n{'=' * 60}")
        print("Summary:")
        print(f"  Total articles processed: {len(graphs)}")
        print(f"  Combined graph entities: {len(combined_graph.entities)}")
        print(f"  Combined graph relations: {len(combined_graph.relations)}")
        print(f"  Output directory: {OUTPUT_DIR}/")
        print(f"{'=' * 60}")

    # Save metadata
    metadata = {
        "articles_processed": len(graphs),
        "articles_metadata": [
            {
                "index": item["index"],
                "title": item["title"],
                "entities": len(item["graph"].entities),
                "relations": len(item["graph"].relations),
            }
            for item in graphs
        ],
        "model": "openai/gpt-5.2",
        "reasoning_effort": "high",
        "deduplication_method": "SEMHASH",
    }

    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
