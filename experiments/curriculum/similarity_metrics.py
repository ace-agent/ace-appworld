#!/usr/bin/env python3
"""
Similarity metrics for comparing text instructions.

This module provides various text similarity metrics that can be used
for clustering and comparing task instructions.

Available metrics:
- Jaccard similarity: Word-based set similarity
- TF-IDF Cosine similarity: Weighted term similarity with IDF scoring
- Levenshtein similarity: Character-level edit distance similarity
- Oracle similarity: Ground truth similarity based on task family ID
- Embedding similarity: Semantic similarity using sentence embeddings (requires API key)
"""

import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional


def extract_task_family_id(task_id: str) -> str:
    """Extract the family ID from a task ID.

    Args:
        task_id: Task ID in format like "76f2c72_2" or "76f2c72_1"

    Returns:
        Family ID (e.g., "76f2c72")
    """
    # Split by underscore and take the first part
    parts = task_id.split('_')
    return parts[0] if parts else task_id


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    # Convert to lowercase and split by non-alphanumeric characters
    return re.findall(r'\b\w+\b', text.lower())


def calculate_jaccard_similarity(s1: str, s2: str) -> float:
    """Calculate Jaccard similarity between two strings based on word tokens.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not s1 or not s2:
        return 0.0

    words1 = set(tokenize(s1))
    words2 = set(tokenize(s2))

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def calculate_cosine_similarity(s1: str, s2: str, idf_scores: Dict[str, float]) -> float:
    """Calculate TF-IDF cosine similarity between two strings.

    Args:
        s1: First string
        s2: Second string
        idf_scores: Dictionary mapping terms to their IDF scores

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not s1 or not s2:
        return 0.0

    tokens1 = tokenize(s1)
    tokens2 = tokenize(s2)

    # Calculate TF-IDF vectors
    def get_tfidf_vector(tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        total_terms = len(tokens)
        tfidf = {}
        for term, count in tf.items():
            tf_score = count / total_terms if total_terms > 0 else 0
            idf_score = idf_scores.get(term, 0)
            tfidf[term] = tf_score * idf_score
        return tfidf

    vec1 = get_tfidf_vector(tokens1)
    vec2 = get_tfidf_vector(tokens2)

    # Calculate cosine similarity
    all_terms = set(vec1.keys()) | set(vec2.keys())

    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in all_terms)
    magnitude1 = math.sqrt(sum(v * v for v in vec1.values()))
    magnitude2 = math.sqrt(sum(v * v for v in vec2.values()))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def calculate_levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculate normalized Levenshtein (edit distance) similarity.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not s1 or not s2:
        return 0.0

    # Levenshtein distance using dynamic programming
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return 0.0 if len2 > 0 else 1.0
    if len2 == 0:
        return 0.0

    # Create distance matrix
    distances = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        distances[i][0] = i
    for j in range(len2 + 1):
        distances[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            distances[i][j] = min(
                distances[i - 1][j] + 1,      # deletion
                distances[i][j - 1] + 1,      # insertion
                distances[i - 1][j - 1] + cost  # substitution
            )

    max_len = max(len1, len2)
    return 1.0 - (distances[len1][len2] / max_len)


def calculate_oracle_similarity(task_id1: str, task_id2: str) -> float:
    """Calculate oracle similarity based on task family ID.

    This is a "ground truth" similarity metric that returns 1.0 if two tasks
    belong to the same family (same prefix before underscore), 0.0 otherwise.

    Args:
        task_id1: First task ID (e.g., "76f2c72_2")
        task_id2: Second task ID (e.g., "76f2c72_3")

    Returns:
        1.0 if same family, 0.0 otherwise
    """
    if not task_id1 or not task_id2:
        return 0.0

    family1 = extract_task_family_id(task_id1)
    family2 = extract_task_family_id(task_id2)

    return 1.0 if family1 == family2 else 0.0


def get_embedding(
    text: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "text-embedding-3-large"
) -> List[float]:
    """Get embedding vector for a text using OpenAI API.

    Args:
        text: Text to embed
        api_key: OpenAI API key (if None, will try to use OPENAI_API_KEY env var)
        base_url: API base URL (default: None, uses OpenAI's default)
        model: Embedding model to use (default: text-embedding-3-large)
               Other options: "text-embedding-3-large", "text-embedding-ada-002"

    Returns:
        Embedding vector as list of floats

    Raises:
        ImportError: If openai package is not installed
        ValueError: If API key is not provided
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required for embedding similarity. Install with: pip install openai")

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key required for embedding similarity. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

    # Create client with or without custom base_url
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    # Use OpenAI's embedding model
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        # Provide helpful error message with available models
        raise RuntimeError(
            f"Failed to get embedding with model '{model}'. "
            f"Error: {str(e)}. "
            f"Available OpenAI models: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002"
        ) from e


def calculate_embedding_similarity(
    s1: str,
    s2: str,
    embeddings_cache: Optional[Dict[str, List[float]]] = None,
    api_key: Optional[str] = None
) -> float:
    """Calculate cosine similarity between embeddings of two texts.

    Args:
        s1: First string
        s2: Second string
        embeddings_cache: Optional cache of pre-computed embeddings {text: embedding}
        api_key: OpenAI API key

    Returns:
        Cosine similarity score between 0.0 and 1.0
    """
    if not s1 or not s2:
        return 0.0

    # Get embeddings from cache or compute them
    if embeddings_cache and s1 in embeddings_cache:
        emb1 = embeddings_cache[s1]
    else:
        emb1 = get_embedding(s1, api_key)
        if embeddings_cache is not None:
            embeddings_cache[s1] = emb1

    if embeddings_cache and s2 in embeddings_cache:
        emb2 = embeddings_cache[s2]
    else:
        emb2 = get_embedding(s2, api_key)
        if embeddings_cache is not None:
            embeddings_cache[s2] = emb2

    # Calculate cosine similarity
    dot_product = sum(a * b for a, b in zip(emb1, emb2))
    magnitude1 = math.sqrt(sum(a * a for a in emb1))
    magnitude2 = math.sqrt(sum(b * b for b in emb2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # Cosine similarity ranges from -1 to 1, normalize to 0 to 1
    similarity = dot_product / (magnitude1 * magnitude2)
    return (similarity + 1) / 2


def compute_embeddings_batch(
    texts: List[str],
    api_key: Optional[str] = None,
    model: str = "text-embedding-3-large"
) -> Dict[str, List[float]]:
    """Pre-compute embeddings for a list of texts.

    Args:
        texts: List of texts to embed
        api_key: OpenAI API key
        model: Embedding model to use

    Returns:
        Dictionary mapping each text to its embedding vector
    """
    embeddings_cache = {}
    for i, text in enumerate(texts):
        if text and text not in embeddings_cache:
            print(f"    Progress: {i+1}/{len(texts)}", end='\r')
            embeddings_cache[text] = get_embedding(text, api_key, model=model)
    print(f"    Progress: {len(texts)}/{len(texts)} - Complete!")
    return embeddings_cache


def compute_idf_scores(texts: List[str]) -> Dict[str, float]:
    """Compute IDF (Inverse Document Frequency) scores for all terms in a text corpus.

    Args:
        texts: List of text documents

    Returns:
        Dictionary mapping terms to their IDF scores
    """
    num_docs = len(texts)
    if num_docs == 0:
        return {}

    # Count document frequency for each term
    doc_freq = defaultdict(int)
    for text in texts:
        unique_terms = set(tokenize(text))
        for term in unique_terms:
            doc_freq[term] += 1

    # Calculate IDF scores
    idf_scores = {}
    for term, freq in doc_freq.items():
        idf_scores[term] = math.log(num_docs / freq) if freq > 0 else 0

    return idf_scores


def calculate_similarity(
    s1: str,
    s2: str,
    metric: str = "jaccard",
    idf_scores: Dict[str, float] = None,
    task_id1: Optional[str] = None,
    task_id2: Optional[str] = None,
    embeddings_cache: Optional[Dict[str, List[float]]] = None,
    api_key: Optional[str] = None
) -> float:
    """Calculate similarity between two strings using the specified metric.

    Args:
        s1: First string (instruction text or task ID for oracle)
        s2: Second string (instruction text or task ID for oracle)
        metric: Similarity metric to use ('jaccard', 'cosine', 'levenshtein', 'oracle', 'embedding')
        idf_scores: IDF scores (required for cosine similarity)
        task_id1: First task ID (required for oracle similarity)
        task_id2: Second task ID (required for oracle similarity)
        embeddings_cache: Cache of embeddings (for embedding similarity)
        api_key: API key for embedding service (for embedding similarity)

    Returns:
        Similarity score between 0.0 and 1.0

    Raises:
        ValueError: If an unknown metric is specified or if required parameters are missing
    """
    if metric == "jaccard":
        return calculate_jaccard_similarity(s1, s2)
    elif metric == "cosine":
        if idf_scores is None:
            raise ValueError("IDF scores required for cosine similarity")
        return calculate_cosine_similarity(s1, s2, idf_scores)
    elif metric == "levenshtein":
        return calculate_levenshtein_similarity(s1, s2)
    elif metric == "oracle":
        if task_id1 is None or task_id2 is None:
            raise ValueError("Task IDs required for oracle similarity")
        return calculate_oracle_similarity(task_id1, task_id2)
    elif metric == "embedding":
        return calculate_embedding_similarity(s1, s2, embeddings_cache, api_key)
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")
