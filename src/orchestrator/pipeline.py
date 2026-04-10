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
    def __init__(self, config_path: str, domain: str = None):
        self.config_path = config_path
        self.config = self._load_config()
        self.domain = domain

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

    def run(self, inputs: list):
        if isinstance(inputs, str):
            inputs = [inputs]
        logger.info("Starting IsoKernel Knowledge Graph Pipeline")
        
        # Check what phases to run
        run_phase_1 = self.config.get("pipeline", {}).get("run_phase_1", True)
        run_phase_2 = self.config.get("pipeline", {}).get("run_phase_2", True)
        
        # Separate paths (strings) from preloaded DocumentSource models
        raw_paths = [item for item in inputs if isinstance(item, str)]
        preloaded_docs = [item for item in inputs if isinstance(item, DocumentSource)]
        
        # Discover files
        files_to_process = self._discover_files(raw_paths)
        
        if not files_to_process and not preloaded_docs:
            logger.error(f"No valid input files or DocumentSource objects found in {inputs}.")
            return
            
        logger.info(f"Discovered {len(files_to_process)} file(s) and {len(preloaded_docs)} preloaded document(s) for processing.")
            
        # Variables to pass state
        triples = []
        
        # ---------------- Phase 1 ---------------- #
        if run_phase_1:
            ext_cfg = self.config.get("extraction", {})
            extractor = TripleExtractor(
                model=ext_cfg.get("model", None),
                domain=self.domain if self.domain else ext_cfg.get("domain", "general knowledge")
            )
            
            # Combine preloaded docs with dynamically loaded ones
            documents_to_process = list(preloaded_docs)
            
            for file_path in files_to_process:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                documents_to_process.append(DocumentSource(id=os.path.basename(file_path), text_content=text))

            async def _extract_all(docs):
                async def _extract_single(doc, delay):
                    if delay > 0:
                        await asyncio.sleep(delay)
                    logger.info(f"Processing document ID: {doc.id}")
                    try:
                        res = await extractor.extract_raw_triples(doc)
                        return [t.model_dump() for t in res.triples]
                    except Exception as e:
                        logger.error(f"Failed to process document {doc.id}: {e}")
                        return []

                # Stagger the start of each request by 1.5 seconds
                tasks = [_extract_single(doc, i * 1.5) for i, doc in enumerate(docs)]
                results = await asyncio.gather(*tasks)
                
                all_triples = []
                for res in results:
                    all_triples.extend(res)
                return all_triples

            loop = asyncio.get_event_loop()
            triples.extend(loop.run_until_complete(_extract_all(documents_to_process)))
            
            # Save Phase 1 output
            out_dir = self.config.get("output", {}).get("schemas_dir", "data/outputs/schemas")
            os.makedirs(out_dir, exist_ok=True)
            raw_output_path = os.path.join(out_dir, "phase1_raw_triples.json")
            with open(raw_output_path, "w") as f:
                json.dump(triples, f, indent=4)
            logger.info(f"Phase 1 completed. {len(triples)} total raw triples saved to {raw_output_path}")

        # If phase 1 is disabled but phase 2 is enabled, we need triples from a previous run
        elif run_phase_2:
            out_dir = self.config.get("output", {}).get("schemas_dir", "data/outputs/schemas")
            raw_output_path = os.path.join(out_dir, "phase1_raw_triples.json")
            if os.path.exists(raw_output_path):
                with open(raw_output_path, "r") as f:
                    triples = json.load(f)
                logger.info(f"Loaded {len(triples)} triples from previous Phase 1 run.")
            else:
                logger.error("Phase 1 is disabled and no cached triples were found. Aborting.")
                return

        # ---------------- Phase 2 ---------------- #
        if run_phase_2:
            ref_cfg = self.config.get("refinement", {})
            processor = GraphProcessor(
                use_embeddings=ref_cfg.get("use_embeddings", True),
                embedding_model=ref_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
                clustering_method=ref_cfg.get("clustering_method", "agglomerative"),
                similarity_threshold=ref_cfg.get("similarity_threshold", 0.8),
                community_detection=ref_cfg.get("community_detection", "louvain"),
                compression_mode=ref_cfg.get("compression_mode", "unified"),
                compress_fields=ref_cfg.get("compress_fields", ["subject", "object"]),
                hypernym_resolution=ref_cfg.get("hypernym_resolution", "shortest_string")
            )
            
            graphs_dir = self.config.get("output", {}).get("graphs_dir", "data/outputs/graphs")
            os.makedirs(graphs_dir, exist_ok=True)
            graph = processor.process(triples, graphs_dir=graphs_dir)
            
            output_html = os.path.join(graphs_dir, "03_final_graph_with_communities.html")
            processor.save_visualization(graph, output_html)
            
            schemas_dir = self.config.get("output", {}).get("schemas_dir", "data/outputs/schemas")
            schemas_out = os.path.join(schemas_dir, "phase2_refined_clusters.csv")
            processor.export_schemas(graph, schemas_out)
            
            logger.info(f"Phase 2 completed. Clusters exported to {schemas_out} and graph saved to {output_html}")
