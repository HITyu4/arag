"""Semantic search tool using API-based embedding."""

import os
import pickle
import numpy as np
import requests
from typing import Dict, List, Any, Tuple, TYPE_CHECKING

from dotenv import load_dotenv
load_dotenv()

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


class SemanticSearchToolAPI(BaseTool):
    """Semantic search using API-based embedding (e.g., DashScope, OpenAI)."""

    def __init__(
        self,
        chunks_file: str,
        index_dir: str = "index",
        model_name: str = "text-embedding-v4",
        api_key: str = None,
        api_base: str = None
    ):
        if not HAS_TIKTOKEN:
            raise ImportError("tiktoken required. Install: pip install tiktoken")

        self.chunks_file = chunks_file
        self.index_dir = index_dir
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ARAG_API_KEY")
        self.api_base = api_base or os.getenv("ARAG_BASE_URL", "https://api.openai.com/v1")

        self._load_index()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    def _load_index(self):
        index_file = os.path.join(self.index_dir, "sentence_index.pkl")

        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index not found: {index_file}")

        with open(index_file, 'rb') as f:
            index_data = pickle.load(f)

        self.sentences = index_data['sentences']
        self.embeddings = index_data['embeddings']
        self.sentence_to_chunk = index_data['sentence_to_chunk']
        self.chunks = index_data['chunks']

    def _encode_query_api(self, query: str) -> np.ndarray:
        """Encode query using API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "input": [query]
        }

        response = requests.post(
            f"{self.api_base.rstrip('/')}/embeddings",
            headers=headers,
            json=payload,
            timeout=300
        )
        response.raise_for_status()

        result = response.json()
        embedding = result['data'][0]['embedding']
        return np.array(embedding)

    @property
    def name(self) -> str:
        return "semantic_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": """Semantic search using embedding similarity. Matches your query against sentences in each chunk via vector similarity.

WHEN TO USE:
- When keyword search fails to find relevant information
- When exact wording in documents is unknown
- For conceptual/meaning-based matching

RETURNS: Abbreviated snippets with matched sentences. Use read_chunk to get full text for answering.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query describing what information you're looking for"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of most relevant results to return (default: 5, max: 20)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    def execute(self, context: 'AgentContext', query: str, top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
        top_k = min(top_k, 20)

        # Encode query via API
        query_embedding = self._encode_query_api(query)

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k * 3]

        chunk_sentences = {}
        for idx in top_indices:
            sentence = self.sentences[idx]
            chunk_id = self.sentence_to_chunk[idx]
            similarity = float(similarities[idx])

            if chunk_id not in chunk_sentences:
                chunk_sentences[chunk_id] = []
            chunk_sentences[chunk_id].append({
                'sentence': sentence,
                'similarity': similarity,
                'position': idx
            })

        chunk_scores = []
        for chunk_id, sents in chunk_sentences.items():
            max_similarity = max(s['similarity'] for s in sents)
            chunk_scores.append((chunk_id, max_similarity, sents))

        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = chunk_scores[:top_k]

        if not top_chunks:
            return f"No results for: {query}", {"retrieved_tokens": 0, "chunks_found": 0}

        result_parts = []
        for chunk_id, max_sim, sents in top_chunks:
            chunk_text = self.chunks[chunk_id]['text']
            sents_sorted = sorted(sents, key=lambda x: chunk_text.find(x['sentence']))
            matched_text = "... " + " ... ".join([s['sentence'] for s in sents_sorted]) + " ..."
            result_parts.append(f"Chunk ID: {chunk_id} (Similarity: {max_sim:.3f})\nMatched: {matched_text}")

        tool_result = "\n\n".join(result_parts)

        all_matched = []
        for _, _, sents in top_chunks:
            all_matched.extend([s['sentence'] for s in sents])

        retrieved_tokens = len(self.tokenizer.encode("\n".join(all_matched))) if all_matched else 0

        context.add_retrieval_log(
            tool_name="semantic_search",
            tokens=retrieved_tokens,
            metadata={"query": query, "chunks_found": len(top_chunks)}
        )

        return tool_result, {"retrieved_tokens": retrieved_tokens, "chunks_found": len(top_chunks)}