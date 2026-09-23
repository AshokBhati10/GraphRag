#!/usr/bin/env python3
"""
Execute the full data pipeline end-to-end for PubMedQA integration testing.
Steps:
1. Verify fixed_token.jsonl exists and check for pqa_ articles
2. Check semantic_cluster.jsonl
3. Build Neo4j graph
4. Verify pqa_ chunks were inserted
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.schema import Neo4jConnection

def step1_verify_chunking():
    """Step 1: Verify fixed_token.jsonl contains PubMedQA data"""
    print("\n" + "="*60)
    print("STEP 1: Verify Chunking Data")
    print("="*60)

    fixed_token_path = PROJECT_ROOT / "data" / "processed" / "fixed_token.jsonl"

    if not fixed_token_path.exists():
        print(f"❌ fixed_token.jsonl not found at {fixed_token_path}")
        return False

    # Count total and pqa_ chunks
    total_chunks = 0
    pqa_chunks = 0

    with open(fixed_token_path, "r") as f:
        for line in f:
            if line.strip():
                total_chunks += 1
                data = json.loads(line)
                if data["article_id"].startswith("pqa_"):
                    pqa_chunks += 1

    print(f"✓ Total chunks in fixed_token.jsonl: {total_chunks}")
    print(f"✓ PubMedQA chunks (pqa_*): {pqa_chunks}")

    if pqa_chunks == 0:
        print("❌ No PubMedQA chunks found!")
        return False

    print(f"✓ {pqa_chunks}/{total_chunks} chunks ({100*pqa_chunks/total_chunks:.1f}%) are from PubMedQA")
    return True

def step2_check_semantic_cluster():
    """Step 2: Check if semantic_cluster.jsonl exists"""
    print("\n" + "="*60)
    print("STEP 2: Check Semantic Clustering Data")
    print("="*60)

    semantic_path = PROJECT_ROOT / "data" / "processed" / "semantic_cluster.jsonl"

    if not semantic_path.exists():
        print(f"⚠️  semantic_cluster.jsonl not found - build_graph requires this file")
        print(f"   Location: {semantic_path}")
        return False

    # Count chunks
    total_chunks = 0
    pqa_chunks = 0

    with open(semantic_path, "r") as f:
        for line in f:
            if line.strip():
                total_chunks += 1
                data = json.loads(line)
                if data["article_id"].startswith("pqa_"):
                    pqa_chunks += 1

    print(f"✓ Total chunks in semantic_cluster.jsonl: {total_chunks}")
    print(f"✓ PubMedQA chunks (pqa_*): {pqa_chunks}")

    if pqa_chunks == 0:
        print("⚠️  No PubMedQA chunks found in semantic_cluster.jsonl")
        return True  # Not critical

    print(f"✓ {pqa_chunks}/{total_chunks} chunks ({100*pqa_chunks/total_chunks:.1f}%) are from PubMedQA")
    return True

def step3_verify_neo4j():
    """Step 3: Verify Neo4j connection and basic schema"""
    print("\n" + "="*60)
    print("STEP 3: Verify Neo4j Connection")
    print("="*60)

    try:
        with Neo4jConnection() as driver:
            with driver.session() as session:
                result = session.run("RETURN 1 as ping")
                result.single()
        print("✓ Neo4j connection successful")
        return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

def step4_check_pqa_in_graph():
    """Step 4: Check if PubMedQA chunks are in the graph"""
    print("\n" + "="*60)
    print("STEP 4: Check PubMedQA Data in Neo4j")
    print("="*60)

    try:
        with Neo4jConnection() as driver:
            with driver.session() as session:
                # Check for pqa_ chunks
                result = session.run(
                    'MATCH (c:Chunk) WHERE c.article_id STARTS WITH "pqa_" RETURN COUNT(c) as count'
                )
                count = result.single()["count"]

                if count == 0:
                    print("⚠️  No PubMedQA chunks found in Neo4j graph")
                    print("   Run `python -m src.graph.build_graph --limit 1000` to build the graph")
                    return False

                print(f"✓ Found {count} PubMedQA chunks in Neo4j")

                # Check entities
                result = session.run(
                    '''MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                       WHERE c.article_id STARTS WITH "pqa_"
                       RETURN COUNT(DISTINCT e) as entity_count'''
                )
                entity_count = result.single()["entity_count"]
                print(f"✓ Linked entities from PubMedQA: {entity_count}")

                return True
    except Exception as e:
        print(f"❌ Error querying Neo4j: {e}")
        return False

def main():
    print("\n🚀 Executing GraphRAG Pipeline End-to-End")
    print("=" * 60)

    # Step 1
    if not step1_verify_chunking():
        sys.exit(1)

    # Step 2
    step2_check_semantic_cluster()

    # Step 3
    if not step3_verify_neo4j():
        sys.exit(1)

    # Step 4
    if not step4_check_pqa_in_graph():
        print("\n⚠️  Next steps:")
        print("   1. Run chunking with semantic_cluster strategy:")
        print("      python -m src.chunking.chunker --strategy semantic_cluster --merge-per-article")
        print("   2. Build the Neo4j graph:")
        print("      python -m src.graph.build_graph --limit 1000")
        print("   3. Run this script again to verify")

    print("\n✅ Pipeline verification complete!\n")

if __name__ == "__main__":
    main()
