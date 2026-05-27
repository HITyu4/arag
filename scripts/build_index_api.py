#!/usr/bin/env python3
"""
Build sentence-level embedding index using API.
Supports OpenAI-compatible embedding APIs (Qwen, OpenAI, etc.).

Usage:
    python scripts/build_index_api.py \
        --chunks data/chunks.json \
        --output data/index \
        --model Qwen/Qwen3-Embedding-0.6B \
        --api-key YOUR_API_KEY \
        --api-base https://api.example.com/v1
"""

import os
import json
import re
import pickle
import argparse
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r'[.!?\n]+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def load_chunks(chunks_file: str) -> List[Dict[str, Any]]:
    """Load chunks from file."""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if data and isinstance(data[0], dict):
        return data

    chunks = []
    for item in data:
        if isinstance(item, str):
            parts = item.split(':', 1)
            if len(parts) == 2:
                chunks.append({'id': parts[0], 'text': parts[1]})
    return chunks


def encode_sentences_api(
    sentences: List[str],
    api_key: str,
    api_base: str,
    model_name: str,
    batch_size: int = 32,
    dimensions: int = None
) -> np.ndarray:
    """Encode sentences using OpenAI-compatible embedding API."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    all_embeddings = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding"):
        batch = sentences[i:i + batch_size]

        payload = {
            "model": model_name,
            "input": batch
        }

        if dimensions:
            payload["dimensions"] = dimensions

        response = requests.post(
            f"{api_base.rstrip('/')}/embeddings",
            headers=headers,
            json=payload,
            timeout=300
        )
        response.raise_for_status()

        result = response.json()
        embeddings = [item['embedding'] for item in result['data']]
        all_embeddings.extend(embeddings)

    return np.array(all_embeddings)


def build_index(
    chunks_file: str,
    output_dir: str,
    model_name: str,
    api_key: str = None,
    api_base: str = None,
    batch_size: int = 32,
    dimensions: int = None
):
    """Build sentence-level embedding index using API."""

    # Load chunks
    print(f"Loading chunks from: {chunks_file}")
    chunks = load_chunks(chunks_file)
    print(f"Loaded {len(chunks)} chunks")

    # Create chunk lookup
    chunk_lookup = {c['id']: c for c in chunks}

    # Extract sentences
    print("Extracting sentences...")
    sentences = []
    sentence_to_chunk = []

    for chunk in tqdm(chunks, desc="Processing chunks"):
        chunk_sentences = split_sentences(chunk['text'])
        for sent in chunk_sentences:
            sentences.append(sent)
            sentence_to_chunk.append(chunk['id'])

    print(f"Total sentences: {len(sentences)}")

    # Get API credentials
    if not api_key:
        api_key = os.getenv("ARAG_API_KEY")
    if not api_key:
        raise ValueError("API key required. Set --api-key or ARAG_API_KEY env var.")

    if not api_base:
        api_base = os.getenv("ARAG_BASE_URL", "https://api.openai.com/v1")

    print(f"Encoding via API: {api_base}")
    print(f"Model: {model_name}")

    # Encode sentences via API
    embeddings = encode_sentences_api(
        sentences=sentences,
        api_key=api_key,
        api_base=api_base,
        model_name=model_name,
        batch_size=batch_size,
        dimensions=dimensions
    )

    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Save index
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    index_file = output_path / "sentence_index.pkl"
    index_data = {
        'sentences': sentences,
        'embeddings': embeddings,
        'sentence_to_chunk': sentence_to_chunk,
        'chunks': chunk_lookup,
        'model_name': model_name
    }

    print(f"Saving index to: {index_file}")
    with open(index_file, 'wb') as f:
        pickle.dump(index_data, f)

    print(f"Index built successfully!")
    print(f"  - Chunks: {len(chunks)}")
    print(f"  - Sentences: {len(sentences)}")
    print(f"  - Embedding dim: {embeddings.shape[1]}")


def main():
    parser = argparse.ArgumentParser(description="Build semantic search index via API")
    parser.add_argument("--chunks", "-c", required=True, help="Path to chunks.json")
    parser.add_argument("--output", "-o", required=True, help="Output directory for index")
    parser.add_argument("--model", "-m", default="text-embedding-v4",required=True, help="Embedding model name")
    parser.add_argument("--api-key", help="API key (or set ARAG_API_KEY env var)")
    parser.add_argument("--api-base", help="API base URL (or set ARAG_BASE_URL env var)")
    parser.add_argument("--batch-size", "-b", type=int, default=10, help="Batch size")
    parser.add_argument("--dimensions", "-d", type=int, default=None,
                       help="Embedding dimensions (if required by API)")

    args = parser.parse_args()

    build_index(
        chunks_file=args.chunks,
        output_dir=args.output,
        model_name=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        batch_size=args.batch_size,
        dimensions=args.dimensions
    )


if __name__ == "__main__":
    main()