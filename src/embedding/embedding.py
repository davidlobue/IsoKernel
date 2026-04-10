import logging
from typing import List, Dict, Any, Tuple
from src.core.logger import setup_logger

logger = setup_logger("embedding_service")

class EmbeddingService:
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 clustering_method: str = "agglomerative",
                 similarity_threshold: float = 0.8,
                 compression_mode: str = "unified",
                 compress_fields: List[str] = None,
                 hypernym_resolution: str = "semantic_centroid"):
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
        self.compression_mode = compression_mode
        self.compress_fields = compress_fields if compress_fields is not None else ["subject", "object"]
        self.hypernym_resolution = hypernym_resolution
        
        self.embedder = None
        self.sync_client = None
        self.llm_model = None
        
        # Initialize LLM backend if required
        if self.hypernym_resolution in ["llm_resolution", "semantic_centroid"]:
            import instructor
            from openai import OpenAI
            from dotenv import load_dotenv, find_dotenv
            load_dotenv(find_dotenv())
            
            provider = os.getenv("LLM_PROVIDER", "openai").lower()
            self.llm_model = os.getenv("LLM_MODEL_NAME", "gpt-4o")
            base_url = os.getenv("LLM_BASE_URL", None)
            
            logger.info(f"Initializing synchronous LLM client for Taxonomic/Hypernym mapping via '{provider}'")
            if provider == "local":
                client = OpenAI(base_url=base_url, api_key="ollama")
                self.sync_client = instructor.from_openai(client, mode=instructor.Mode.JSON)
            elif provider == "google":
                try:
                    import vertexai
                    from vertexai.generative_models import GenerativeModel
                    model_instance = GenerativeModel(self.llm_model)
                    self.sync_client = instructor.from_vertexai(model_instance, mode=instructor.Mode.VERTEXAI_TOOLS)
                except ImportError:
                    logger.error("google-cloud-aiplatform is required for vertexai in synchronous mode")
            else:
                self.sync_client = instructor.from_openai(OpenAI())
                
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

    def _cluster_and_map(self, nodes_list: List[str], triples: List[Dict[str, str]], target_fields: List[str]) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
        if not nodes_list:
            return {}, {}
            
        logger.info(f"Computing embeddings for {len(nodes_list)} unique nodes in fields: {target_fields}")
        embeddings = self.encode(nodes_list)
        
        node_mapping = {}
        cluster_logs = {}
        
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
            
            # Track frequencies for hypernym resolution tiebreaking
            counts = {}
            for t in triples:
                for f in target_fields:
                    if f in t:
                        val = t[f]
                        counts[val] = counts.get(val, 0) + 1
            
            clusters = {}
            for node, label in zip(nodes_list, labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(node)
                
            if self.hypernym_resolution == "semantic_centroid":
                from sklearn.metrics.pairwise import cosine_similarity
                
                # Math arrays prep
                node_idx_map = {node: idx for idx, node in enumerate(nodes_list)}
                centroid_clusters = {}
                
                # Calculate absolute internal tensors
                for label, members in clusters.items():
                    member_idx = [node_idx_map[m] for m in members]
                    cluster_emb = embeddings[member_idx]
                    centroid_vec = np.mean(cluster_emb, axis=0).reshape(1, -1)
                    
                    similarities = cosine_similarity(centroid_vec, cluster_emb)[0]
                    closest_idx = np.argmax(similarities)
                    centroid_member = members[closest_idx]
                    
                    centroid_clusters[label] = {
                        "centroid": centroid_member,
                        "members": members
                    }
                    
                if self.sync_client:
                    import json
                    from src.extraction.prompts import Prompts
                    from src.core.models import TaxonomicLiftingResult
                    
                    logger.info(f"Submitting {len(centroid_clusters)} isolated centroids for Taxonomic Lifting deductive logic...")
                    try:
                        response = self.sync_client.chat.completions.create(
                            model=self.llm_model,
                            messages=[
                                {"role": "system", "content": Prompts.TAXONOMIC_LIFTING_SYSTEM},
                                {"role": "user", "content": Prompts.get_taxonomic_user(json.dumps(centroid_clusters, indent=2))}
                            ],
                            response_model=TaxonomicLiftingResult
                        )
                        
                        resolution_map = {}
                        for res in response.resolutions:
                            if res.members_verified:
                                resolution_map[res.cluster_id] = res.formal_hypernym
                                logger.info(f"Taxonomic Lift Verified [{res.cluster_id}]: {res.centroid} -> {res.formal_hypernym}")
                            else:
                                resolution_map[res.cluster_id] = res.centroid
                                logger.warning(f"Taxonomic Lift Rejected [Is-A failed!]: {res.centroid} -> Reverting instantly to raw geometric centroid.")
                                
                        for label, members in clusters.items():
                            cluster_id_str = f"c_{label}"
                            mapped_val = resolution_map.get(str(label), centroid_clusters[label]["centroid"])
                            for member in members:
                                node_mapping[member] = mapped_val
                                cluster_logs[member] = {
                                    "nlp_cluster_id": cluster_id_str,
                                    "final_hypernym": mapped_val
                                }
                    except Exception as e:
                        logger.error(f"Taxonomic Engine failed natively: {e}. Executing unconditional mathematical Centroid mapping globally.")
                        for label, members in clusters.items():
                            cluster_id_str = f"c_{label}"
                            mapped_val = centroid_clusters[label]["centroid"]
                            for member in members:
                                node_mapping[member] = mapped_val
                                cluster_logs[member] = {
                                    "nlp_cluster_id": cluster_id_str,
                                    "final_hypernym": mapped_val
                                }
                else:
                    logger.info("Executing isolated local Semantic Centroid mapping across clustering geometries...")
                    for label, members in clusters.items():
                        cluster_id_str = f"c_{label}"
                        mapped_val = centroid_clusters[label]["centroid"]
                        for member in members:
                            node_mapping[member] = mapped_val
                            cluster_logs[member] = {
                                "nlp_cluster_id": cluster_id_str,
                                "final_hypernym": mapped_val
                            }
                            
            elif self.hypernym_resolution == "llm_resolution" and self.sync_client:
                import json
                from src.extraction.prompts import Prompts
                from src.core.models import LLMHypernymResolutionResult
                
                logger.info(f"Sending {len(clusters)} clusters to LLM for semantic hypernym resolution...")
                
                cluster_map = {str(k): v for k, v in clusters.items()}
                
                try:
                    response = self.sync_client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": Prompts.HYPERNYM_SYSTEM},
                            {"role": "user", "content": Prompts.get_hypernym_user(json.dumps(cluster_map, indent=2))}
                        ],
                        response_model=LLMHypernymResolutionResult
                    )
                    
                    resolution_map = {res.cluster_id: res.hypernym for res in response.resolutions}
                    
                    for label, members in clusters.items():
                        cluster_id_str = f"c_{label}"
                        mapped_val = resolution_map.get(str(label), sorted(members, key=lambda x: -counts.get(x, 0))[0]) # tiebreaker
                        for member in members:
                            node_mapping[member] = mapped_val
                            cluster_logs[member] = {
                                "nlp_cluster_id": cluster_id_str,
                                "final_hypernym": mapped_val
                            }
                            
                except Exception as e:
                    logger.error(f"LLM Resolution failed: {e}. Falling back to most_frequent.")
                    for label, members in clusters.items():
                        cluster_id_str = f"c_{label}"
                        representative = sorted(members, key=lambda x: (-counts.get(x, 0), len(x)))[0]
                        for member in members:
                            node_mapping[member] = representative
                            cluster_logs[member] = {
                                "nlp_cluster_id": cluster_id_str,
                                "final_hypernym": representative
                            }
            else:
                for label, members in clusters.items():
                    cluster_id_str = f"c_{label}"
                    if self.hypernym_resolution == "most_frequent":
                        representative = sorted(members, key=lambda x: (-counts.get(x, 0), len(x)))[0]
                    else: 
                        representative = sorted(members, key=len)[0]
                        
                    for member in members:
                        node_mapping[member] = representative
                        cluster_logs[member] = {
                            "nlp_cluster_id": cluster_id_str,
                            "final_hypernym": representative
                        }
        else:
            logger.warning(f"Clustering method {self.clustering_method} not implemented fully.")
            
        return node_mapping, cluster_logs

    def semantic_compression(self, triples: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        if not self.embedder:
            logger.warning("Embedder unavailable. Skipping compression.")
            return triples, []
            
        if not self.compress_fields:
            logger.info("Compression fields array is empty. Skipping compression.")
            return triples, []
            
        logger.info(f"Running Semantic Compression (Mode: {self.compression_mode}, Fields: {self.compress_fields})...")
        
        global_node_mapping = {}
        field_node_mappings = {f: {} for f in self.compress_fields}
        all_logs = []
        
        if self.compression_mode == "unified":
            unique_nodes = set()
            for t in triples:
                for f in self.compress_fields:
                    if f in t:
                        unique_nodes.add(t[f])
            
            global_node_mapping, global_logs = self._cluster_and_map(list(unique_nodes), triples, self.compress_fields)
            
            # format logs
            for orig, data in global_logs.items():
                all_logs.append({
                    "field_type": "unified",
                    "original_text": orig,
                    "nlp_cluster_id": data["nlp_cluster_id"],
                    "final_hypernym": data["final_hypernym"]
                })
        else:
            for field in self.compress_fields:
                unique_nodes = set()
                for t in triples:
                    if field in t:
                        unique_nodes.add(t[field])
                
                f_mapping, f_logs = self._cluster_and_map(list(unique_nodes), triples, [field])
                field_node_mappings[field] = f_mapping
                
                # format logs
                for orig, data in f_logs.items():
                    all_logs.append({
                        "field_type": field,
                        "original_text": orig,
                        "nlp_cluster_id": data["nlp_cluster_id"],
                        "final_hypernym": data["final_hypernym"]
                    })
        
        # Apply the mapping to triples
        compressed_triples = []
        for t in triples:
            mapped_t = {}
            for key, val in t.items():
                mapped_t[key] = val  # copy default
                
            if self.compression_mode == "unified":
                for f in self.compress_fields:
                    if f in t:
                        mapped_t[f] = global_node_mapping.get(t[f], t[f])
            else:
                for f in self.compress_fields:
                    if f in t:
                        mapped_t[f] = field_node_mappings.get(f, {}).get(t[f], t[f])
                        
            # Carry over original strings tracking
            compressed_triples.append({
                "subject": mapped_t.get('subject', t.get('subject')),
                "predicate": mapped_t.get('predicate', t.get('predicate')),
                "object": mapped_t.get('object', t.get('object')),
                "original_subject": t.get('subject', ''),
                "original_predicate": t.get('predicate', ''),
                "original_object": t.get('object', ''),
                "quote": t.get('quote', ''),
                "certainty_score": t.get('certainty_score', '')
            })
            
        return compressed_triples, all_logs
