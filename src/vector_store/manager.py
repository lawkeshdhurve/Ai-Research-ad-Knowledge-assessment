import os
import math
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings
from config.settings import settings

class VectorStoreManager:
    """
    Vector Store Manager providing persistent embedding storage, 
    dense vector similarity search, keyword matching, and hybrid RRF retrieval.
    """

    def __init__(self):
        self.persist_directory = str(settings.VECTOR_DB_DIR)
        os.makedirs(self.persist_directory, exist_ok=True)
        self._embedding_model = None
        
        # Initialize Persistent ChromaDB Client
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name="research_documents",
            metadata={"hnsw:space": "cosine"}
        )

    @property
    def embedding_model(self):
        """Lazily loads embedding model on first vector encode request."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        return self._embedding_model

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embeds and indexes document chunks in ChromaDB vector store.
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []
        texts_to_embed = []

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            text = chunk["text"]
            
            ids.append(chunk_id)
            documents.append(text)
            texts_to_embed.append(text)
            metadatas.append({
                "doc_id": chunk["doc_id"],
                "file_name": chunk.get("file_name", "Unknown"),
                "page_number": int(chunk.get("page_number", 1)),
                "chunk_index": int(chunk.get("chunk_index", 0))
            })

        embeddings = self.embedding_model.encode(texts_to_embed, show_progress_bar=False).tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        return len(chunks)

    def delete_document_chunks(self, doc_id: str) -> bool:
        """Removes all chunks associated with a specific document ID from vector index."""
        try:
            self.collection.delete(where={"doc_id": doc_id})
            return True
        except Exception as e:
            print(f"Error deleting chunks for doc_id {doc_id}: {e}")
            return False

    def semantic_search(self, query: str, top_k: int = 4, doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Dense vector similarity search using cosine distance."""
        if not query.strip():
            return []

        query_embedding = self.embedding_model.encode([query], show_progress_bar=False).tolist()[0]
        
        where_clause = None
        if doc_ids and len(doc_ids) > 0:
            if len(doc_ids) == 1:
                where_clause = {"doc_id": doc_ids[0]}
            else:
                where_clause = {"$or": [{"doc_id": d_id} for d_id in doc_ids]}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause
        )

        formatted_results = []
        if results and results.get("documents") and len(results["documents"][0]) > 0:
            for idx in range(len(results["documents"][0])):
                doc_text = results["documents"][0][idx]
                meta = results["metadatas"][0][idx]
                chunk_id = results["ids"][0][idx]
                distance = results["distances"][0][idx] if "distances" in results else 0.0
                score = round(1.0 - max(0.0, distance), 4)

                formatted_results.append({
                    "chunk_id": chunk_id,
                    "doc_id": meta.get("doc_id"),
                    "file_name": meta.get("file_name"),
                    "page_number": meta.get("page_number"),
                    "text": doc_text,
                    "score": score
                })

        return formatted_results

    def keyword_search(self, query: str, top_k: int = 4, doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Exact keyword matching search across indexed documents."""
        all_docs = self.collection.get()
        if not all_docs or not all_docs.get("documents"):
            return []

        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        if not query_terms:
            query_terms = [query.lower()]

        scored_chunks = []
        for idx in range(len(all_docs["documents"])):
            text = all_docs["documents"][idx]
            meta = all_docs["metadatas"][idx]
            chunk_id = all_docs["ids"][idx]

            if doc_ids and meta.get("doc_id") not in doc_ids:
                continue

            text_lower = text.lower()
            term_matches = sum(1 for term in query_terms if term in text_lower)
            if term_matches > 0:
                match_score = term_matches / len(query_terms)
                scored_chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": meta.get("doc_id"),
                    "file_name": meta.get("file_name"),
                    "page_number": meta.get("page_number"),
                    "text": text,
                    "score": round(match_score, 4)
                })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    def hybrid_search(self, query: str, top_k: int = 4, doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Hybrid Search combining Dense Semantic Search and Sparse Keyword Search 
        using Reciprocal Rank Fusion (RRF).
        """
        dense_results = self.semantic_search(query, top_k=top_k * 2, doc_ids=doc_ids)
        keyword_results = self.keyword_search(query, top_k=top_k * 2, doc_ids=doc_ids)

        rrf_scores = {}
        chunks_by_id = {}
        rrf_k = 60

        # RRF for dense results
        for rank, res in enumerate(dense_results):
            cid = res["chunk_id"]
            chunks_by_id[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

        # RRF for keyword results
        for rank, res in enumerate(keyword_results):
            cid = res["chunk_id"]
            chunks_by_id[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))

        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        
        final_results = []
        for cid in sorted_cids[:top_k]:
            item = chunks_by_id[cid]
            item["hybrid_score"] = round(rrf_scores[cid], 5)
            final_results.append(item)

        return final_results

    def get_all_chunks_for_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        """Returns all text chunks for a document sorted by page and chunk index."""
        all_docs = self.collection.get(where={"doc_id": doc_id})
        if not all_docs or not all_docs.get("documents"):
            return []

        chunks = []
        for idx in range(len(all_docs["documents"])):
            chunks.append({
                "chunk_id": all_docs["ids"][idx],
                "doc_id": all_docs["metadatas"][idx].get("doc_id"),
                "file_name": all_docs["metadatas"][idx].get("file_name"),
                "page_number": all_docs["metadatas"][idx].get("page_number", 1),
                "chunk_index": all_docs["metadatas"][idx].get("chunk_index", 0),
                "text": all_docs["documents"][idx]
            })

        chunks.sort(key=lambda x: (x["page_number"], x["chunk_index"]))
        return chunks

    def get_total_chunk_count(self) -> int:
        """Returns total number of chunks currently indexed in vector DB."""
        return self.collection.count()
