from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            self._use_chroma = True
            client = chromadb.EphemeralClient()
            self._collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata or {},
            "embedding": embedding
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        
        scored = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored.append((record, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for record, score in scored[:top_k]:
            result = record.copy()
            result["score"] = score
            results.append(result)
        
        return results

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma and self._collection:
            ids = []
            documents = []
            embeddings = []
            
            for doc in docs:
                record = self._make_record(doc)
                ids.append(doc.id)
                documents.append(doc.content)
                embeddings.append(record["embedding"])
            
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=[doc.metadata or {} for doc in docs]
            )
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)
                self._next_index += 1

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection:
            results_chroma = self._collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            results = []
            if results_chroma["ids"] and len(results_chroma["ids"]) > 0:
                for i, id_val in enumerate(results_chroma["ids"][0]):
                    distance = results_chroma["distances"][0][i] if "distances" in results_chroma else 0
                    doc = results_chroma["documents"][0][i] if "documents" in results_chroma else ""
                    metadata = results_chroma["metadatas"][0][i] if "metadatas" in results_chroma else {}
                    
                    results.append({
                        "id": id_val,
                        "content": doc,
                        "metadata": metadata,
                        "score": 1 - distance if distance is not None else 0
                    })
            return results
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection:
            return self._collection.count()
        else:
            return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            return self.search(query, top_k)
        
        if self._use_chroma and self._collection:
            results_chroma = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=metadata_filter
            )
            
            results = []
            if results_chroma["ids"] and len(results_chroma["ids"]) > 0:
                for i, id_val in enumerate(results_chroma["ids"][0]):
                    distance = results_chroma["distances"][0][i] if "distances" in results_chroma else 0
                    doc = results_chroma["documents"][0][i] if "documents" in results_chroma else ""
                    metadata = results_chroma["metadatas"][0][i] if "metadatas" in results_chroma else {}
                    
                    results.append({
                        "id": id_val,
                        "content": doc,
                        "metadata": metadata,
                        "score": 1 - distance if distance is not None else 0
                    })
            return results
        else:
            filtered_records = []
            for record in self._store:
                match = True
                for key, value in metadata_filter.items():
                    if record.get("metadata", {}).get(key) != value:
                        match = False
                        break
                if match:
                    filtered_records.append(record)
            
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection:
            try:
                self._collection.delete(ids=[doc_id])
                return True
            except Exception:
                return False
        else:
            initial_count = len(self._store)
            self._store = [record for record in self._store if record.get("id") != doc_id]
            return len(self._store) < initial_count
