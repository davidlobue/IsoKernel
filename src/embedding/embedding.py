import logging
from typing import List, Dict, Any
from src.core.logger import setup_logger

logger = setup_logger("embedding_service")

class EmbeddingService:
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 clustering_method: str = "agglomerative",
                 similarity_threshold: float = 0.8):
        import os
        
        # Check if localized model exists in lib/models/
        local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib", "models", embedding_model))
        if os.path.exists(local_path):
            self.embedding_model = local_path
            logger.info(f"Using localized model directory: {self.embedding_model}")
        else:
            self.embedding_model = embedding_model
            logger.warning(f"No local model found at {local_path}. Fetching '{self.embedding_model}' from HF Hub.")
            
        self.clustering_method = clustering_method
        self.similarity_threshold = similarity_threshold
        self.embedder = None
        
        logger.info(f"Initializing embedding model '{self.embedding_model}'")
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.embedding_model)
        except ImportError:
            logger.error("sentence-transformers is not installed but use_embeddings is True.")

    def encode(self, texts: List[str]):
        if not self.embedder:
            raise ValueError("Embedder not initialized.")
        return self.embedder.encode(texts)
        
    def semantic_compression(self, triples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not self.embedder:
            logger.warning("Embedder unavailable. Skipping compression.")
            return triples
            
        logger.info("Running Semantic Compression (Option B)...")
        # Extract all unique nodes (subjects and objects)
        unique_nodes = set()
        for t in triples:
            unique_nodes.add(t['subject'])
            unique_nodes.add(t['object'])
            
        nodes_list = list(unique_nodes)
        
        if not nodes_list:
            return triples
            
        logger.info(f"Computing embeddings for {len(nodes_list)} unique nodes.")
        embeddings = self.encode(nodes_list)
        
        node_mapping = {}
        
        if self.clustering_method == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering
            import numpy as np
            
            # Using cosine distance
            distance_threshold = 1.0 - self.similarity_threshold
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric='cosine',
                linkage='average'
            )
            labels = clusterer.fit_predict(embeddings)
            
            # Create mapping from node to a representative hypernym (e.g., shortest string in cluster)
            clusters = {}
            for node, label in zip(nodes_list, labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(node)
                
            for label, members in clusters.items():
                if len(members) > 1:
                    logger.debug(f"Found cluster resolving: {members}")
                # Simple heuristical hypernym: shortest string
                representative = sorted(members, key=len)[0]
                for member in members:
                    node_mapping[member] = representative
        else:
            logger.warning(f"Clustering method {self.clustering_method} not implemented fully, skipping compression.")
            return triples
            
        # Apply the mapping to triples
        compressed_triples = []
        for t in triples:
            compressed_triples.append({
                "subject": node_mapping.get(t['subject'], t['subject']),
                "predicate": t['predicate'],
                "object": node_mapping.get(t['object'], t['object'])
            })
            
        return compressed_triples
