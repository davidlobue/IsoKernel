import os
import json
import asyncio
import yaml
from src.core.logger import setup_logger
from src.core.models import DocumentSource
from src.extraction.extractor import TripleExtractor
from src.topology.processor import GraphProcessor

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

logger = setup_logger("orchestrator")

class PipelineOrchestrator:
    def __init__(self, config_path: str, domain: str = None, verbose: bool = False):
        self.config_path = config_path
        self.config = self._load_config()
        self.domain = domain
        self.verbose = verbose

    def _clear_ollama_vram(self, phase_name: str):
        import gc
        import json
        import urllib.request
        gc.collect()
        try:
            url = f"{os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1').replace('/v1', '/api')}/generate"
            model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o")
            data = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=2.0)
            logger.info(f"Cleared Memory & Ollama VRAM after {phase_name}.")
        except Exception:
            pass

    def _load_config(self) -> dict:
        if not os.path.exists(self.config_path):
            logger.error(f"Config file not found: {self.config_path}")
            return {}
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _discover_files(self, input_paths: list) -> list:
        valid_extensions = {".txt", ".md", ".csv"}
        discovered = []
        for path in input_paths:
            if os.path.isfile(path):
                discovered.append(path)
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if any(file.endswith(ext) for ext in valid_extensions):
                            discovered.append(os.path.join(root, file))
            else:
                logger.warning(f"Input path '{path}' not found or invalid.")
        return discovered

    def prepare_documents(self, inputs: list) -> list:
        if isinstance(inputs, str):
            inputs = [inputs]
        
        raw_paths = [item for item in inputs if isinstance(item, str)]
        preloaded_docs = [item for item in inputs if isinstance(item, DocumentSource)]
        
        files_to_process = self._discover_files(raw_paths)
        
        documents_to_process = list(preloaded_docs)
        for file_path in files_to_process:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            documents_to_process.append(DocumentSource(id=os.path.basename(file_path), text_content=text))
            
        return documents_to_process

    def extract_themes(self, documents_to_process: list) -> dict:
        from src.core.chunking import SemanticChunker
        ext_cfg = self.config.get("extraction", {})
        extractor = TripleExtractor(
            model=ext_cfg.get("model", None),
            domain=self.domain if self.domain else ext_cfg.get("domain", "general knowledge")
        )
        
        async def _extract_all_themes(docs):
            max_concurrent = self.config.get("pipeline", {}).get("max_concurrent_llm_calls", 3)
            sem = asyncio.Semaphore(max_concurrent)
            
            # 1. Expand standard documents into heavily chunked overlaps
            virtual_chunks = []
            for doc in docs:
                chunks = SemanticChunker.chunk_text(doc.text_content, max_words=ext_cfg.get("theme_chunk_max_words", 4000), overlap=21)
                for idx, text in enumerate(chunks):
                    virtual_chunks.append((doc.id, DocumentSource(id=f"{doc.id}_chunk_{idx}", text_content=text)))
            
            async def _extract_single(orig_id, vdoc, delay):
                async with sem:
                    if delay > 0:
                        await asyncio.sleep(delay)
                    logger.info(f"Discovering themes for chunk: {vdoc.id}")
                    try:
                        res = await extractor.extract_themes(vdoc)
                        return orig_id, [t.model_dump() for t in res.themes]
                    except Exception as e:
                        logger.error(f"Failed to discover themes for chunk {vdoc.id}: {e}")
                        return orig_id, []

            tasks = [_extract_single(orig_id, vdoc, i * 0.5) for i, (orig_id, vdoc) in enumerate(virtual_chunks)]
            results = await asyncio.gather(*tasks)
            
            await extractor.close()
            
            # Collapse back down rigorously locally to exact files stably
            consolidated = {}
            for orig_id, themes in results:
                if orig_id not in consolidated:
                    consolidated[orig_id] = []
                
                # Check for explicit dictionary title duplicates securely natively
                seen_titles = {t['title'] for t in consolidated[orig_id]}
                for t in themes:
                    if t['title'] not in seen_titles:
                        consolidated[orig_id].append(t)
                        seen_titles.add(t['title'])
                        
            return consolidated

        loop = asyncio.get_event_loop()
        final_doc_themes = loop.run_until_complete(_extract_all_themes(documents_to_process))
        self._clear_ollama_vram("Theme Extraction")
        
        output_dir = self.config.get("output", {}).get("themes_dir", "outputs/themes")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "01_document_themes.json"), "w") as f:
            json.dump(final_doc_themes, f, indent=2)
            
        return final_doc_themes

    def consolidate_themes(self, theme_maps: dict) -> dict:
        ext_cfg = self.config.get("extraction", {})
        extractor = TripleExtractor(
            model=ext_cfg.get("model", None),
            domain=self.domain if self.domain else ext_cfg.get("domain", "general knowledge")
        )
        
        # Flatten dict mapping into a massive string output list
        all_flattened_topics = []
        for doc_id, themes in theme_maps.items():
            for t in themes:
                all_flattened_topics.append(t)
                
        logger.info(f"Consolidating {len(all_flattened_topics)} document-level themes into a Master Corpus Ontology...")
        
        async def _run_consolidation():
            res = await extractor.consolidate_themes(all_flattened_topics)
            await extractor.close()
            return {
                "master_domain": res.master_domain,
                "themes": [t.model_dump() for t in res.themes]
            }

        loop = asyncio.get_event_loop()
        master_themes = loop.run_until_complete(_run_consolidation())
        self._clear_ollama_vram("Theme Consolidation")
        
        output_dir = self.config.get("output", {}).get("themes_dir", "outputs/themes")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "02_master_themes.json"), "w") as f:
            json.dump(master_themes, f, indent=2)
            
        return master_themes

    def extract_triples(self, documents_to_process: list, discovered_themes_map: dict = None) -> list:
        from src.core.chunking import SemanticChunker
        ext_cfg = self.config.get("extraction", {})
        extractor = TripleExtractor(
            model=ext_cfg.get("model", None),
            domain=self.domain if self.domain else ext_cfg.get("domain", "general knowledge")
        )
        
        async def _extract_all(docs):
            max_concurrent = self.config.get("pipeline", {}).get("max_concurrent_llm_calls", 3)
            sem = asyncio.Semaphore(max_concurrent)
            
            # 1. Expand logically tracking the specific boundaries exactly dynamically explicitly
            virtual_chunks = []
            for doc in docs:
                chunks = SemanticChunker.chunk_text(doc.text_content, max_words=ext_cfg.get("triple_chunk_max_words", 1000), overlap=50)
                for idx, text in enumerate(chunks):
                    virtual_chunks.append((doc.id, DocumentSource(id=f"{doc.id}_chunk_{idx}", text_content=text)))
            
            async def _extract_single(orig_id, vdoc, delay):
                async with sem:
                    if delay > 0:
                        await asyncio.sleep(delay)
                    logger.info(f"Processing structural extraction for chunk: {vdoc.id}")
                    try:
                        if isinstance(discovered_themes_map, list):
                            themes_for_doc = discovered_themes_map
                        else:
                            themes_for_doc = discovered_themes_map.get(orig_id) if discovered_themes_map else None
                            
                        res = await extractor.extract_raw_triples(vdoc, themes=themes_for_doc)
                        return [t.model_dump() for t in res.triples]
                    except Exception as e:
                        logger.error(f"Failed to process chunk {vdoc.id}: {e}")
                        return []

            # Stagger the start of each request gracefully bounding overlapping scaling logically
            tasks = [_extract_single(orig_id, vdoc, i * 0.5) for i, (orig_id, vdoc) in enumerate(virtual_chunks)]
            results = await asyncio.gather(*tasks)
            
            await extractor.close()
            
            all_triples = []
            for res in results:
                all_triples.extend(res)
            return all_triples

        loop = asyncio.get_event_loop()
        final_triples = loop.run_until_complete(_extract_all(documents_to_process))
        self._clear_ollama_vram("Triple Extraction")
        
        # Persist raw extractions log
        out_dir = self.config.get("output", {}).get("schemas_dir", "outputs/schemas")
        os.makedirs(out_dir, exist_ok=True)
        raw_output_path = os.path.join(out_dir, "phase1_raw_triples.json")
        with open(raw_output_path, "w") as f:
            json.dump(final_triples, f, indent=4)
            
        return final_triples

    def refine_graph(self, triples: list, master_domain: str = None):
        # Sanitize all string variables globally natively stripping mathematical noise
        sanitized_triples = []
        for t in triples:
            t_clean = {}
            for k, v in t.items():
                if isinstance(v, str):
                    t_clean[k] = v.replace("_", " ").strip()
                else:
                    t_clean[k] = v
            sanitized_triples.append(t_clean)
        triples = sanitized_triples
        
        processor = self._get_processor()
        graphs_dir = self.config.get("output", {}).get("graphs_dir", "outputs/graphs")
        schemas_dir = self.config.get("output", {}).get("schemas_dir", "outputs/schemas")
        os.makedirs(graphs_dir, exist_ok=True)
        os.makedirs(schemas_dir, exist_ok=True)
        
        logger.info("Saving checkpoint: Graph visualization of raw text extracts...")
        initial_graph = processor._build_graph(triples)
        processor.save_visualization(initial_graph, os.path.join(graphs_dir, "01_raw_text_extracts.html"))
        
        compressed_triples = triples
        embed_service = self._get_embedding_service()
        
        if embed_service and embed_service.compress_fields:
            logger.info("Executing Semantic Compression Flow...")
            node_counts = self.extract_unique_nodes(triples)
            
            logger.info("Running pre-process LLM Normalization natively on extracted raw string nodes...")
            norm_mapping = embed_service.preprocess_normalize_nodes(list(node_counts.keys()), master_domain)
            triples = self.apply_compression(triples, norm_mapping)
            node_counts = self.extract_unique_nodes(triples)
            
            embeddings = self.create_embeddings(node_counts)
            embeddings = self.apply_spectral_decomposition(node_counts, embeddings)
            
            # Step 2.4 Agglomerative Structural Proposals
            clusters = self.compute_clusters(node_counts, embeddings)
            
            # --- NEW: Visual Checkpoint for Semantic Proximity ---
            logger.info("Saving checkpoint: Interactive t-SNE Semantic Proximity Map...")
            nodes_list = list(node_counts.keys())
            processor.save_scatter_visualization(embeddings, nodes_list, clusters, os.path.join(graphs_dir, "02_agglomerative_semantic_clusters.html"))
            # ----------------------------------------------------
            
            logger.info("Saving checkpoint: Step 2.4 Agglomerative Structural Proposals (Clustering)...")
            step_2_4_mapping = {}
            for cid, members in clusters.items():
                if len(members) > 0:
                    sorted_members = sorted(members, key=lambda x: (len(x), x))
                    label = f"Cluster_{cid} ({sorted_members[0]})"
                    for m in members:
                        step_2_4_mapping[m] = label
                        
            step_2_4_triples = []
            fields = embed_service.compress_fields
            for t in triples:
                mapped_t = t.copy()
                for key, val in t.items():
                    mapped_t[key] = val
                for f in fields:
                    if f in mapped_t:
                        mapped_t[f] = step_2_4_mapping.get(mapped_t[f], mapped_t[f])
                step_2_4_triples.append(mapped_t)

            step_2_4_graph = processor._build_graph(step_2_4_triples)
            processor.save_visualization(step_2_4_graph, os.path.join(schemas_dir, "step_2_4_agglomerative_proposals.html"))
            
            verified_clusters = self.verify_clusters(clusters, master_domain)
            nodes_list = list(node_counts.keys())
            node_mapping, cluster_logs = self.resolve_hypernyms(verified_clusters, nodes_list, embeddings, triples, master_domain)
            
            compressed_triples = self.apply_compression(triples, node_mapping)
            
        G, output_html, schemas_out = self.build_and_detect_communities(compressed_triples)
        return G, output_html, schemas_out

    def _get_processor(self) -> GraphProcessor:
        ref_cfg = self.config.get("refinement", {})
        return GraphProcessor(
            community_detection=ref_cfg.get("community_detection", "louvain")
        )

    def _get_embedding_service(self):
        ref_cfg = self.config.get("refinement", {})
        if not ref_cfg.get("use_embeddings", True):
            return None
            
        from src.embedding.embedding import EmbeddingService
        return EmbeddingService(
            embedding_model=ref_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
            clustering_method=ref_cfg.get("clustering_method", "agglomerative"),
            similarity_threshold=ref_cfg.get("similarity_threshold", 0.8),
            compression_mode=ref_cfg.get("compression_mode", "unified"),
            compress_fields=ref_cfg.get("compress_fields", ["subject", "object"]),
            hypernym_resolution=ref_cfg.get("hypernym_resolution", "shortest_string"),
            use_spectral_decomposition=ref_cfg.get("use_spectral_decomposition", True),
            spectral_variance_retention=ref_cfg.get("spectral_variance_retention", 0.90),
            max_concurrent_llm_calls=self.config.get("pipeline", {}).get("max_concurrent_llm_calls", 3)
        )

    def extract_unique_nodes(self, triples: list) -> dict:
        embed_service = self._get_embedding_service()
        if not embed_service or not embed_service.compress_fields:
            return {}
        node_counts = {}
        for t in triples:
            for f in embed_service.compress_fields:
                if f in t:
                    val = t[f]
                    node_counts[val] = node_counts.get(val, 0) + 1
        return node_counts

    def create_embeddings(self, node_counts: dict):
        embed_service = self._get_embedding_service()
        if not embed_service: return None
        return embed_service.calculate_embeddings(list(node_counts.keys()))

    def apply_spectral_decomposition(self, node_counts: dict, embeddings):
        embed_service = self._get_embedding_service()
        if not embed_service: return embeddings
        return embed_service.apply_spectral_decomposition(node_counts, embeddings)

    def compute_clusters(self, node_counts: dict, embeddings) -> dict:
        embed_service = self._get_embedding_service()
        if not embed_service: return {}
        return embed_service.cluster_embeddings(list(node_counts.keys()), embeddings)

    def verify_clusters(self, clusters: dict, master_domain: str = None) -> dict:
        embed_service = self._get_embedding_service()
        if not embed_service: return clusters
        return embed_service.verify_clusters(clusters, master_domain)

    def resolve_hypernyms(self, clusters: dict, nodes_list: list, embeddings, triples: list, master_domain: str = None) -> tuple:
        import pandas as pd
        embed_service = self._get_embedding_service()
        if not embed_service: return {}, {}
        node_mapping, cluster_logs = embed_service.resolve_hypernyms(clusters, nodes_list, embeddings, triples, embed_service.compress_fields, master_domain)
        
        # Persist Checkpoint Log
        out_dir = self.config.get("output", {}).get("schemas_dir", "outputs/schemas")
        os.makedirs(out_dir, exist_ok=True)
        if cluster_logs:
            df = pd.DataFrame(cluster_logs)
            df.to_csv(os.path.join(out_dir, "nlp_entity_clusters.csv"), index=False)
            
        return node_mapping, cluster_logs

    def apply_compression(self, triples: list, node_mapping: dict) -> list:
        embed_service = self._get_embedding_service()
        compressed_triples = []
        fields = embed_service.compress_fields if embed_service else []
        for t in triples:
            mapped_t = {}
            for key, val in t.items():
                mapped_t[key] = val
            for f in fields:
                if f in t:
                    mapped_t[f] = node_mapping.get(t[f], t[f])
                    
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
            
        # Persist Long-Format Transform Matrix
        import pandas as pd
        out_dir = self.config.get("output", {}).get("schemas_dir", "outputs/schemas")
        os.makedirs(out_dir, exist_ok=True)
        if compressed_triples:
            df = pd.DataFrame(compressed_triples)
            df.to_csv(os.path.join(out_dir, "Hypernym_triplet_transformations.csv"), index=False)
            
        return compressed_triples

    def build_and_detect_communities(self, compressed_triples: list):
        import networkx as nx
        processor = self._get_processor()
        graphs_dir = self.config.get("output", {}).get("graphs_dir", "outputs/graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        
        logger.info("Building Core Directed Graph...")
        G = processor._build_graph(compressed_triples)
        logger.info(f"Graph initialized with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        
        logger.info(f"Running Community Detection ({processor.community_detection})...")
        undirected_G = G.to_undirected()
        partition = {}
        
        try:
            if processor.community_detection == "louvain":
                import community as community_louvain
                partition = community_louvain.best_partition(undirected_G)
            elif processor.community_detection == "leiden":
                from cdlib import algorithms
                coms = algorithms.leiden(undirected_G)
                partition = coms.to_node_community_map()
                partition = {k: v[0] for k, v in partition.items()}
            elif processor.community_detection == "spectral":
                import numpy as np
                from sklearn.cluster import SpectralClustering
                laplacian = nx.normalized_laplacian_matrix(undirected_G).toarray()
                eigenvalues = np.linalg.eigvalsh(laplacian)
                diffs = np.diff(eigenvalues)
                n_clusters = np.argmax(diffs[:len(diffs)//2]) + 1
                if n_clusters < 2: n_clusters = 2
                
                adj_matrix = nx.to_numpy_array(undirected_G)
                sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
                labels = sc.fit_predict(adj_matrix)
                
                for idx, node in enumerate(undirected_G.nodes()):
                    partition[node] = labels[idx]
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            
        for node, comm_id in partition.items():
            G.nodes[node]['group'] = comm_id
            
        output_html = os.path.join(graphs_dir, "03_final_graph_with_communities.html")
        processor.save_visualization(G, output_html)
        
        schemas_dir = self.config.get("output", {}).get("schemas_dir", "outputs/schemas")
        schemas_out = os.path.join(schemas_dir, "phase2_refined_clusters.csv")
        processor.export_schemas(G, schemas_out)
        
        return G, output_html, schemas_out

    def synthesize_schemas(self, G):
        from src.synthesis.synthesizer import SchemaSynthesizer
        syn = SchemaSynthesizer(self.config)
        return syn.synthesize_schemas(G)

    def run(self, inputs: list):
        if self.verbose:
            print("==================================================")
            print("-> [START] ISOKERNEL KNOWLEDGE GRAPH PIPELINE")
            print("==================================================\n")

        logger.info("Starting IsoKernel Knowledge Graph Pipeline")
        
        # Check what phases to run
        run_phase_1 = self.config.get("pipeline", {}).get("run_phase_1", True)
        run_phase_2 = self.config.get("pipeline", {}).get("run_phase_2", True)
        
        documents_to_process = self.prepare_documents(inputs)
        
        if not documents_to_process:
            logger.error(f"No valid input files or DocumentSource objects found in {inputs}.")
            return
            
        logger.info(f"Discovered {len(documents_to_process)} document(s) for processing.")
            
        triples = []
        
        # ---------------- Phase 1 ---------------- #
        if run_phase_1:
            if self.verbose:
                print("==================================================")
                print("-> [STEP] PHASE 1: THEME & TRIPLE EXTRACTION")
                print("==================================================")
            
            # Two-Pass Theme Discovery and Extraction
            logger.info("Starting Pass A: Theme Discovery...")
            discovered_themes_map = self.extract_themes(documents_to_process)
            
            # Pass A.5 Theme Consolidation
            logger.info("Starting Pass A.5: Theme Consolidation...")
            master_themes_list = self.consolidate_themes(discovered_themes_map)
            
            # Save thematic outputs dynamically for review
            out_dir = self.config.get("output", {}).get("schemas_dir", "outputs/schemas")
            os.makedirs(out_dir, exist_ok=True)
            themes_output_path = os.path.join(out_dir, "phase1_extracted_themes.json")
            with open(themes_output_path, "w") as f:
                json.dump({"document_themes": discovered_themes_map, "master_ontology": master_themes_list}, f, indent=4)
                
            logger.info("Starting Pass B: Triplet Extraction with Theme mapping...")
            triples = self.extract_triples(documents_to_process, master_themes_list.get("themes", []))
            
            # Save Phase 1 output
            out_dir = self.config.get("output", {}).get("schemas_dir", "outputs/schemas")
            os.makedirs(out_dir, exist_ok=True)
            raw_output_path = os.path.join(out_dir, "phase1_raw_triples.json")
            with open(raw_output_path, "w") as f:
                json.dump(triples, f, indent=4)
            logger.info(f"Phase 1 completed. {len(triples)} total raw triples saved to {raw_output_path}")
            if self.verbose:
                print(f"-> [SUMMARY] Phase 1 completed: {len(triples)} triples extracted and saved to {raw_output_path}\n")

        elif run_phase_2:
            out_dir = self.config.get("output", {}).get("schemas_dir", "outputs/schemas")
            raw_output_path = os.path.join(out_dir, "phase1_raw_triples.json")
            if os.path.exists(raw_output_path):
                if self.verbose:
                    print("==================================================")
                    print("-> [STEP] PHASE 1: SKIPPED (Loading Cached Triples)")
                    print("==================================================")
                with open(raw_output_path, "r") as f:
                    triples = json.load(f)
                logger.info(f"Loaded {len(triples)} triples from previous Phase 1 run.")
                if self.verbose:
                    print(f"-> [SUMMARY] Cached Phase 1 loaded: {len(triples)} triples from {raw_output_path}\n")
            else:
                logger.error("Phase 1 is disabled and no cached triples were found. Aborting.")
                return

        # ---------------- Phase 2 ---------------- #
        if run_phase_2:
            if self.verbose:
                print("==================================================")
                print("-> [STEP] PHASE 2: GRAPH PROCESSING & REFINEMENT")
                print("==================================================")
                
            # Attempt to inherently load cached master domain logic specifically to bind semantic validation structurally
            loaded_master_domain = None
            themes_path = os.path.join(self.config.get("output", {}).get("schemas_dir", "outputs/schemas"), "phase1_extracted_themes.json")
            if os.path.exists(themes_path):
                try:
                    with open(themes_path, "r") as f:
                        master_ontology = json.load(f).get("master_ontology", {})
                        loaded_master_domain = master_ontology.get("master_domain")
                except Exception:
                    pass
                    
            graph, output_html, schemas_out = self.refine_graph(triples, master_domain=loaded_master_domain)
            
            logger.info(f"Phase 2 completed. Clusters exported to {schemas_out} and graph saved to {output_html}")
            if self.verbose:
                print("-> [SUMMARY] Phase 2 completed:")
                print(f"             Graph HTML: {output_html}")
                print(f"             Schemas CSV: {schemas_out}\n")

        if self.verbose:
            print("==================================================")
            print("-> [PIPELINE COMPLETE] IsoKernel run finished")
            print("==================================================\n")
            
        if run_phase_2:
            try:
                if self.config.get("pipeline", {}).get("run_phase_4", True):
                    sys_out = self.synthesize_schemas(graph)
                    logger.info(f"Synthesizer Engine generated {len(sys_out)} strict python scripts securely.")
            except Exception as e:
                logger.error(f"Failed to automatically trigger Phase 4 Python synthesis locally: {e}")
