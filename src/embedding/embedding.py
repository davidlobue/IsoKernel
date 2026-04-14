import logging
from typing import List, Dict, Any, Tuple
from src.core.logger import setup_logger
from src.core.utils import run_sync

logger = setup_logger("embedding_service")

class EmbeddingService:
    def __init__(self, 
                 embedding_model: str = "BAAI/bge-m3",
                 clustering_method: str = "agglomerative",
                 similarity_threshold: float = 0.8,
                 compression_mode: str = "unified",
                 compress_fields: List[str] = None,
                 hypernym_resolution: str = "semantic_centroid",
                 use_spectral_decomposition: bool = True,
                 spectral_variance_retention: float = 0.90,
                 max_concurrent_llm_calls: int = 3):
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
        self.max_concurrent_llm_calls = max_concurrent_llm_calls
        self.compress_fields = compress_fields if compress_fields is not None else ["subject", "object"]
        self.hypernym_resolution = hypernym_resolution
        self.use_spectral_decomposition = use_spectral_decomposition
        self.spectral_variance_retention = spectral_variance_retention
        
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

    def calculate_embeddings(self, nodes_list: List[str]):
        if not nodes_list:
            return None
        logger.info(f"Computing embeddings for {len(nodes_list)} unique nodes...")
        return self.encode(nodes_list)

    def apply_spectral_decomposition(self, node_counts: Dict[str, int], embeddings):
        if self.use_spectral_decomposition:
            from sklearn.decomposition import PCA
            import numpy as np
            
            nodes_list = list(node_counts.keys())
            
            # Feature dimensions bounds
            max_possible_components = min(len(nodes_list), embeddings.shape[1])
            
            if max_possible_components > 1:
                logger.info(f"Applying Frequency-Weighted Spectral Geometry (PCA) anchored around high-density terms...")
                
                # Apply Topological Gravity via Weighted PCA
                weights = np.array([node_counts[node] for node in nodes_list], dtype=np.float64)
                
                # 1. Compute weighted center of gravity
                weighted_mean = np.average(embeddings, axis=0, weights=weights)
                
                # 2. Center the hyper-vectors
                embeddings_centered = embeddings - weighted_mean
                
                # 3. Apply radial density mass (sqrt of frequency weights)
                embeddings_weighted = embeddings_centered * np.sqrt(weights[:, np.newaxis])
                
                requested_variance = self.spectral_variance_retention
                
                try:
                    # Dynamically pick the n_components that explain the variance threshold
                    pca = PCA(n_components=requested_variance, svd_solver='full')
                    pca.fit(embeddings_weighted)
                    n_comp_used = pca.n_components_
                except Exception as e:
                    logger.warning(f"Dynamic Variance PCA failed ({e}). Falling back to integer-bounded math.")
                    n_comp_fallback = max(1, max_possible_components - 1)
                    pca = PCA(n_components=n_comp_fallback)
                    pca.fit(embeddings_weighted)
                    n_comp_used = n_comp_fallback
                
                # Project all pure centered coordinates identically against the warped dimensions
                embeddings = pca.transform(embeddings_centered)
                
                logger.info(f"Successfully collapsed {embeddings_centered.shape[1]}-dimensions down to top {n_comp_used} dynamically-warped principal eigenvectors (Variance Target: {requested_variance}).")
                
            else:
                logger.info("Node density too low to deploy Spectral decomposition bounds algebraically. Skipping mathematics.")
        
        return embeddings

    def cluster_embeddings(self, nodes_list: List[str], embeddings) -> Dict[str, List[str]]:
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
            
            clusters = {}
            for node, label in zip(nodes_list, labels):
                l_str = str(label)
                if l_str not in clusters:
                    clusters[l_str] = []
                clusters[l_str].append(node)
            return clusters
        else:
            logger.warning(f"Clustering method {self.clustering_method} not implemented fully.")
            return {"0": nodes_list}

    def resolve_hypernyms(self, clusters: Dict[str, List[str]], nodes_list: List[str], embeddings, triples: List[Dict[str, str]], target_fields: List[str], master_domain: str = None) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
        node_mapping = {}
        cluster_logs = {}
        
        # Track frequencies for hypernym resolution tiebreaking
        counts = {}
        for t in triples:
            for f in target_fields:
                if f in t:
                    val = t[f]
                    counts[val] = counts.get(val, 0) + 1

        if self.hypernym_resolution == "semantic_centroid":
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
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
                from src.embedding.prompts import EmbeddingPrompts
                from src.core.models import TaxonomicLiftingResult
                
                payload_json_data = {}
                centroid_to_label = {}
                for label, data in centroid_clusters.items():
                    c_member = data["centroid"]
                    payload_json_data[c_member] = data["members"]
                    centroid_to_label[c_member.lower()] = label
                
                logger.info(f"Submitting {len(centroid_clusters)} isolated centroids for Taxonomic Lifting deductive logic...")
                try:
                    response = self.sync_client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": EmbeddingPrompts.TAXONOMIC_LIFTING_SYSTEM},
                            {"role": "user", "content": EmbeddingPrompts.get_taxonomic_user(json.dumps(payload_json_data, indent=2), master_domain)}
                        ],
                        response_model=TaxonomicLiftingResult
                    )
                    
                    resolution_map = {}
                    for res in response.resolutions:
                        # Match native centroid string back against corresponding integer label safely 
                        returned_key = str(res.cluster_id).lower()
                        original_label = centroid_to_label.get(returned_key)
                        
                        if original_label is not None:
                            if res.members_verified:
                                resolution_map[str(original_label)] = res.formal_hypernym
                                logger.info(f"Taxonomic Lift Verified [{res.cluster_id}]: {res.centroid} -> {res.formal_hypernym}")
                            else:
                                resolution_map[str(original_label)] = res.centroid
                                logger.warning(f"Taxonomic Lift Rejected [Is-A failed!]: {res.centroid} -> Reverting instantly to raw geometric centroid.")
                        else:
                            logger.error(f"Taxonomic Lift returned untrackable key: '{res.cluster_id}'")
                            
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
            from src.embedding.prompts import EmbeddingPrompts
            from src.core.models import LLMHypernymResolutionResult
            
            logger.info(f"Sending {len(clusters)} clusters to LLM for semantic hypernym resolution...")
            
            try:
                response = self.sync_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": EmbeddingPrompts.HYPERNYM_SYSTEM},
                        {"role": "user", "content": EmbeddingPrompts.get_hypernym_user(json.dumps(clusters, indent=2), master_domain)}
                    ],
                    response_model=LLMHypernymResolutionResult
                )
                
                resolution_map = {str(res.cluster_id): res.canonical_string for res in response.resolutions}
                
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
        
        try:
            import json, urllib.request, os
            url = f"{os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1').replace('/v1', '/api')}/generate"
            model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o")
            data = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=2.0)
            logger.info("Cleared Memory & Ollama VRAM after Taxonomic Consolidation.")
        except Exception:
            pass
            
        return node_mapping, cluster_logs


    def preprocess_normalize_nodes(self, nodes_list: List[str], master_domain: str = None) -> Dict[str, str]:
        if not nodes_list or not self.sync_client:
            return {node: node for node in nodes_list}

        import asyncio
        import os
        import json
        import gc
        import requests
        from openai import AsyncOpenAI
        import instructor
        from src.embedding.prompts import EmbeddingPrompts
        from src.core.models import LLMHypernymResolutionResult

        logger.info(f"Submitting {len(nodes_list)} raw text nodes for Semantic Normalization Preprocessing...")
        
        # Batch nodes to prevent token overload
        batch_size = 50
        batches = [nodes_list[i:i + batch_size] for i in range(0, len(nodes_list), batch_size)]
        
        async def _run_normalization():
            max_concurrent = self.max_concurrent_llm_calls
            sem = asyncio.Semaphore(max_concurrent)
            
            client = AsyncOpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key="ollama")
            async_client = instructor.from_openai(client, mode=instructor.Mode.JSON)
            
            async def _process_batch(batch):
                async with sem:
                    # Treat each node as its own singleton cluster for the prompt
                    cluster_payload = {f"n_{i}": [node] for i, node in enumerate(batch)}
                    payload_json = json.dumps(cluster_payload, indent=2)
                    
                    try:
                        res = await async_client.chat.completions.create(
                            model=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
                            messages=[
                                {"role": "system", "content": EmbeddingPrompts.HYPERNYM_SYSTEM},
                                {"role": "user", "content": EmbeddingPrompts.get_hypernym_user(payload_json, master_domain)}
                            ],
                            response_model=LLMHypernymResolutionResult
                        )
                        
                        # Map back n_i to original node text, and assign canonical output
                        mapping = {}
                        for resolution in res.resolutions:
                            cid = resolution.cluster_id
                            if cid in cluster_payload:
                                original_text = cluster_payload[cid][0]
                                mapping[original_text] = resolution.canonical_string
                        return mapping
                    except Exception as e:
                        logger.warning(f"Normalization failed for batch: {e}. Passing nodes unaltered.")
                        return {node: node for node in batch}

            tasks = [_process_batch(batch) for batch in batches]
            _res = await asyncio.gather(*tasks)

            try:
                import gc, requests
                gc.collect()
                url = f"{os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1').replace('/v1', '/api')}/generate"
                requests.post(url, json={"model": os.getenv("LLM_MODEL_NAME", "gpt-4o"), "keep_alive": 0}, timeout=5.0)
                logger.info("Cleared Ollama VRAM after Preprocessing Normalization.")
            except Exception:
                pass
                
            return _res, tasks, async_client

        results, tasks, async_client = run_sync(_run_normalization)

        final_mapping = {}
        for r in results:
            final_mapping.update(r)
            
        # Ensure mapping is complete mathematically (any dropped nodes fallback seamlessly)
        for node in nodes_list:
            if node not in final_mapping:
                final_mapping[node] = node
                
        return final_mapping

    def verify_clusters(self, clusters: Dict[str, List[str]], master_domain: str = None) -> Dict[str, List[str]]:
        """
        Intercepts mathematical Agglomerative geometry proposals and verifies them contextually natively via Instructor async processing.
        Splits rejected clusters aggressively back into protective singletons.
        """
        if not self.sync_client:
            logger.info("No LLM client configured for cluster verification. Skipping contextual validation hook.")
            return clusters
            
        verified_clusters = {}
        clusters_to_verify = {}
        
        # Protective Filter: Only verify groupings (Size > 1). Pass singletons immediately.
        for label, members in clusters.items():
            if len(members) <= 1:
                verified_clusters[label] = members
            else:
                clusters_to_verify[label] = members
                
        if not clusters_to_verify:
             return verified_clusters
             
        import os, instructor, json, asyncio
        import gc
        import requests
        from openai import AsyncOpenAI
        from src.embedding.prompts import EmbeddingPrompts
        from src.core.models import ClusterContextualValidation
        
        logger.info(f"Submitting {len(clusters_to_verify)} mathematical spatial clusters for Contextual Verification...")
        
        async def _run_validations():
            max_concurrent = self.max_concurrent_llm_calls
            sem = asyncio.Semaphore(max_concurrent)
            
            client = AsyncOpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key="ollama")
            async_client = instructor.from_openai(client, mode=instructor.Mode.JSON)
            
            async def _check_single(c_id, members):
                async with sem:
                    payload = json.dumps({"cluster_id": c_id, "proposed_members": members})
                    try:
                        res = await async_client.chat.completions.create(
                            model=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
                            messages=[
                                {"role": "system", "content": EmbeddingPrompts.CONTEXTUAL_VALIDATION_SYSTEM},
                                {"role": "user", "content": EmbeddingPrompts.get_validation_user(payload, master_domain)}
                            ],
                            response_model=ClusterContextualValidation
                        )
                        return c_id, members, res.accuracy_destroyed, res.reasoning, res.condition_detected
                    except Exception as e:
                        logger.warning(f"Contextual validation locally failed for cluster {c_id}: {e}. Defaulting to split to preserve accuracy.")
                        return c_id, members, True, str(e), "Exception Fallback"
                        
            tasks = [_check_single(c_id, members) for c_id, members in clusters_to_verify.items()]
            _res = await asyncio.gather(*tasks)

            try:
                import gc, requests
                gc.collect()
                url = f"{os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1').replace('/v1', '/api')}/generate"
                requests.post(url, json={"model": os.getenv("LLM_MODEL_NAME", "gpt-4o"), "keep_alive": 0}, timeout=5.0)
                logger.info("Cleared Ollama VRAM after Contextual Validation.")
            except Exception:
                pass

            return _res, tasks, async_client

        results, tasks, async_client = run_sync(_run_validations)

        split_counter = 99000  # Offset to strictly prevent overlapping singleton ID collisions
        rejected_attempts = []
        for c_id, members, accuracy_destroyed, reasoning, condition_detected in results:
            if accuracy_destroyed:
                logger.warning(f"Validation Engine intervened! Rejected contextual mapping for cluster '{c_id}' due to '{condition_detected}'. Splitting {len(members)} entries back into singletons. Reason: {reasoning}")
                rejected_attempts.append({
                    "cluster_id": c_id,
                    "proposed_members": members,
                    "reasoning": reasoning
                })
                for ind_member in members:
                    verified_clusters[str(split_counter)] = [ind_member]
                    split_counter += 1
            else:
                verified_clusters[c_id] = members

        if rejected_attempts:
            try:
                out_dir = os.path.join("outputs", "schemas")
                os.makedirs(out_dir, exist_ok=True)
                rejected_file = os.path.join(out_dir, "rejected_clusters.json")
                
                with open(rejected_file, "w") as f:
                    json.dump(rejected_attempts, f, indent=4)
                    
                logger.info(f"Saved {len(rejected_attempts)} rejected contextual mapping attempts to {rejected_file} for human review.")
            except Exception as e:
                logger.error(f"Failed to save rejected clusters log: {e}")

        return verified_clusters

    def _cluster_and_map(self, node_counts: Dict[str, int], triples: List[Dict[str, str]], target_fields: List[str], master_domain: str = None) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
        if not node_counts:
            return {}, {}
            
        nodes_list = list(node_counts.keys())
            
        embeddings = self.calculate_embeddings(nodes_list)
        embeddings = self.apply_spectral_decomposition(node_counts, embeddings)
        clusters = self.cluster_embeddings(nodes_list, embeddings)
        
        # Step 2.5 Contextual Constraint Validation execution
        clusters = self.verify_clusters(clusters, master_domain)
        
        node_mapping, cluster_logs = self.resolve_hypernyms(clusters, nodes_list, embeddings, triples, target_fields, master_domain)
        return node_mapping, cluster_logs

    def semantic_compression(self, triples: List[Dict[str, str]], master_domain: str = None) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
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
            node_counts = {}
            for t in triples:
                for f in self.compress_fields:
                    if f in t:
                        val = t[f]
                        node_counts[val] = node_counts.get(val, 0) + 1
            
            global_node_mapping, global_logs = self._cluster_and_map(node_counts, triples, self.compress_fields)
            
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
                node_counts = {}
                for t in triples:
                    if field in t:
                        val = t[field]
                        node_counts[val] = node_counts.get(val, 0) + 1
                
                f_mapping, f_logs = self._cluster_and_map(node_counts, triples, [field])
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
