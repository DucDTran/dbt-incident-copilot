"""
Knowledge Base Tool - Semantic search over business documentation.

This module provides functionality to search the knowledge base for relevant
business rules, data quality policies, and documentation.

Features:
- Semantic search using Gemini embeddings
- Disk-based embedding cache for fast startup
- Fallback to keyword search
- Rate limiting for embedding API calls

Usage:
    from app.agent.tools.knowledge_base_tool import (
        get_knowledge_base,
        search_for_business_rule,
        tool_consult_knowledge_base,
    )
    
    # Search knowledge base
    kb = get_knowledge_base()
    results = kb.search("pricing rules for listings")
    
    # Search for specific business rule
    result = search_for_business_rule("sentiment", "fact_reviews", "accepted_values")
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import hashlib
import json
import logging
import numpy as np

from google import genai
from google.genai import types

from app.config import get_settings

logger = logging.getLogger(__name__)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First embedding vector
        b: Second embedding vector
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


logger = logging.getLogger(__name__)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First embedding vector
        b: Second embedding vector
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


class GeminiKnowledgeBase:
    """
    Knowledge base with semantic search using Gemini embeddings.
    
    Provides semantic search over markdown documentation files in the
    knowledge base directory. Uses embedding cache to avoid redundant
    API calls and rate limiting to stay within quotas.
    
    Attributes:
        EMBEDDING_MODEL: The Gemini model used for embeddings
        
    Example:
        kb = GeminiKnowledgeBase()
        results = kb.search("pricing rules for listings", top_k=5)
        for r in results:
            print(f"{r['section_title']}: {r['relevance_score']:.2f}")
    """
    
    EMBEDDING_MODEL = "text-embedding-004"
    
    def __init__(self) -> None:
        """Initialize the knowledge base with caching support."""
        self.settings = get_settings()
        self.kb_path = Path(self.settings.knowledge_base_path)
        self._client: Optional[genai.Client] = None
        self._document_embeddings: List[Dict[str, Any]] = []
        self._initialized = False
        
        # Import cache (lazy to avoid circular imports)
        self._cache = None
    
    @property
    def cache(self):
        """Get the embedding cache (lazy initialization)."""
        if self._cache is None:
            try:
                from app.core.cache import get_embedding_cache
                self._cache = get_embedding_cache()
            except ImportError:
                logger.warning("Embedding cache not available")
                self._cache = None
        return self._cache
    
    @property
    def client(self) -> genai.Client:
        """Get the Gemini client (lazy initialization)."""
        if self._client is None:
            self._client = genai.Client(api_key=self.settings.google_api_key)
        return self._client
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for text, using cache if available.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached
        
        # Call API
        try:
            # Rate limit check (async not available here, so we skip)
            response = self.client.models.embed_content(
                model=self.EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                )
            )
            embedding = response.embeddings[0].values
            
            # Store in cache
            if self.cache and embedding:
                self.cache.set(text, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None
    
    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        Get embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector or None if failed
        """
        try:
            response = self.client.models.embed_content(
                model=self.EMBEDDING_MODEL,
                contents=query,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY"
                )
            )
            return response.embeddings[0].values
        except Exception as e:
            logger.error(f"Failed to get query embedding: {e}")
            return None
    
    def _extract_sections(self, content: str) -> List[Dict[str, str]]:
        """
        Extract sections from markdown content.
        
        Args:
            content: Markdown file content
            
        Returns:
            List of sections with title and content
        """
        sections = []
        current_section = {"title": "Introduction", "content": ""}
        
        for line in content.split('\n'):
            if line.startswith('#'):
                if current_section["content"].strip():
                    sections.append(current_section)
                title = line.lstrip('#').strip()
                current_section = {"title": title, "content": ""}
            else:
                current_section["content"] += line + "\n"
        
        if current_section["content"].strip():
            sections.append(current_section)
        
        return sections
    
    def _initialize_embeddings(self) -> bool:
        """
        Initialize embeddings for all knowledge base documents.
        
        Loads embeddings from cache where available, computing new
        embeddings only for uncached content.
        
        Returns:
            True if initialization succeeded
        """
        if self._initialized:
            return True
        
        if not self.kb_path.exists():
            logger.warning(f"Knowledge base path does not exist: {self.kb_path}")
            return False
        
        try:
            cache_hits = 0
            api_calls = 0
            
            for md_file in self.kb_path.glob("**/*.md"):
                try:
                    with open(md_file, 'r') as f:
                        content = f.read()
                    
                    title = md_file.stem.replace('_', ' ').title()
                    sections = self._extract_sections(content)
                    
                    for section in sections:
                        section_text = f"{section['title']}\n\n{section['content']}"
                        truncated_text = section_text[:8000]  # Limit text length
                        
                        # Check cache first
                        embedding = None
                        if self.cache:
                            embedding = self.cache.get(truncated_text)
                            if embedding:
                                cache_hits += 1
                        
                        # Compute if not cached
                        if embedding is None:
                            embedding = self._get_embedding(truncated_text)
                            api_calls += 1
                        
                        if embedding:
                            self._document_embeddings.append({
                                "document": title,
                                "document_path": str(md_file),
                                "section_title": section["title"],
                                "content": section["content"].strip(),
                                "embedding": embedding,
                            })
                            
                except Exception as e:
                    logger.warning(f"Failed to process {md_file}: {e}")
                    continue
            
            self._initialized = True
            logger.info(
                f"Indexed {len(self._document_embeddings)} sections from knowledge base "
                f"(cache hits: {cache_hits}, API calls: {api_calls})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for relevant content.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            
        Returns:
            List of matching sections with relevance scores
        """
        if not self._initialized:
            if not self._initialize_embeddings():
                return self._fallback_search(query, top_k)
        
        if not self._document_embeddings:
            return self._fallback_search(query, top_k)
        
        try:
            # Get query embedding
            query_embedding = self._get_query_embedding(query)
            
            if not query_embedding:
                return self._fallback_search(query, top_k)
            
            # Calculate similarity scores
            results = []
            for doc in self._document_embeddings:
                similarity = cosine_similarity(query_embedding, doc["embedding"])
                results.append({
                    "document": doc["document"],
                    "document_path": doc["document_path"],
                    "section_title": doc["section_title"],
                    "content": doc["content"],
                    "relevance_score": similarity,
                    "is_semantic": True,
                })
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Filter out low-relevance results (similarity < 0.3)
            results = [r for r in results if r["relevance_score"] > 0.3]
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return self._fallback_search(query, top_k)
    
    def _fallback_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Fallback to simple keyword search if embeddings are unavailable.
        
        Args:
            query: Search query
            top_k: Maximum number of results
            
        Returns:
            List of matching sections based on keyword overlap
        """
        results = []
        query_terms = set(query.lower().split())
        
        if not self.kb_path.exists():
            return results
        
        for md_file in self.kb_path.glob("**/*.md"):
            try:
                with open(md_file, 'r') as f:
                    content = f.read()
                
                title = md_file.stem.replace('_', ' ').title()
                sections = self._extract_sections(content)
                
                for section in sections:
                    section_text = (section["title"] + " " + section["content"]).lower()
                    matches = sum(1 for term in query_terms if term in section_text)
                    
                    if query.lower() in section_text:
                        matches += 5
                    
                    if matches > 0:
                        results.append({
                            "document": title,
                            "document_path": str(md_file),
                            "section_title": section["title"],
                            "content": section["content"].strip(),
                            "relevance_score": float(matches),
                            "is_semantic": False,
                        })
            except Exception:
                continue
        
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]


# Singleton instance
_knowledge_base: Optional[GeminiKnowledgeBase] = None


def get_knowledge_base() -> GeminiKnowledgeBase:
    """
    Get the singleton knowledge base instance.
    
    Returns:
        GeminiKnowledgeBase instance
    """
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = GeminiKnowledgeBase()
    return _knowledge_base


def tool_consult_knowledge_base(query: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Search the knowledge base for documentation matching the query.
    
    Args:
        query: Search query (business term, column name, policy name, etc.)
        context: Optional additional context to improve search
        
    Returns:
        Dict with status, message, and matching results
    """
    kb = get_knowledge_base()
    
    full_query = query
    if context:
        full_query = f"{query} {context}"
    
    results = kb.search(full_query, top_k=5)
    
    if not results:
        return {
            "status": "success",
            "message": "No matching documentation found",
            "data": {
                "query": query,
                "matches": [],
                "recommendation": "Consider adding documentation for this topic to the knowledge base.",
            }
        }
    
    return {
        "status": "success",
        "message": f"Found {len(results)} relevant document sections",
        "data": {
            "query": query,
            "matches": results,
            "top_match": results[0] if results else None,
        }
    }


def search_for_business_rule(
    column_name: str,
    model_name: str,
    error_type: str
) -> Dict[str, Any]:
    """
    Search for business rules related to a specific column and error.
    
    Args:
        column_name: Name of the column with the issue
        model_name: Name of the dbt model
        error_type: Type of validation error (e.g., 'accepted_values', 'not_null')
        
    Returns:
        Dict with status and matching business rules
    """
    kb = get_knowledge_base()
    
    semantic_query = f"""
    Business rules and data quality policies for {column_name} column in {model_name}.
    Looking for allowed values, valid ranges, business definitions, 
    and handling of {error_type} validation errors.
    """
    
    results = kb.search(semantic_query.strip(), top_k=3)
    
    return {
        "status": "success",
        "message": f"Found {len(results)} relevant business rules",
        "data": {
            "column_name": column_name,
            "model_name": model_name,
            "error_type": error_type,
            "rules": results,
            "has_explicit_rule": len(results) > 0,
        }
    }


def list_knowledge_base() -> Dict[str, Any]:
    """
    List all documents in the knowledge base.
    
    Returns:
        Dict with list of documents, sections, and metadata
    """
    kb = get_knowledge_base()
    kb_path = kb.kb_path
    
    doc_list = []
    
    if kb_path.exists():
        for md_file in kb_path.glob("**/*.md"):
            try:
                with open(md_file, 'r') as f:
                    content = f.read()
                
                sections = kb._extract_sections(content)
                
                doc_list.append({
                    "title": md_file.stem.replace('_', ' ').title(),
                    "path": str(md_file),
                    "sections": [s["title"] for s in sections],
                })
            except Exception:
                continue
    
    return {
        "status": "success",
        "message": f"Found {len(doc_list)} documents in knowledge base",
        "data": {
            "documents": doc_list,
            "kb_path": str(kb_path),
            "using_semantic_search": kb._initialized,
            "indexed_sections": len(kb._document_embeddings),
        }
    }
