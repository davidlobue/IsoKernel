import os
import networkx as nx
from typing import List, Dict, Any
from src.core.logger import setup_logger
from pyvis.network import Network
import pandas as pd

try:
    import community as community_louvain # python-louvain
except ImportError:
    community_louvain = None

logger = setup_logger("phase2_processor")

class GraphProcessor:
    def __init__(self, 
                 use_embeddings: bool = True,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 clustering_method: str = "agglomerative",
                 similarity_threshold: float = 0.8,
                 community_detection: str = "louvain",
                 compression_mode: str = "unified",
                 compress_fields: List[str] = None,
                 hypernym_resolution: str = "shortest_string"):
        self.use_embeddings = use_embeddings
        self.community_detection = community_detection.lower()
        
        self.embed_service = None
        if self.use_embeddings:
            from src.embedding.embedding import EmbeddingService
            self.embed_service = EmbeddingService(
                embedding_model=embedding_model,
                clustering_method=clustering_method,
                similarity_threshold=similarity_threshold,
                compression_mode=compression_mode,
                compress_fields=compress_fields,
                hypernym_resolution=hypernym_resolution
            )

    def _build_graph(self, triples: List[Dict[str, str]]) -> nx.DiGraph:
        G = nx.DiGraph()
        for t in triples:
            subject = t['subject']
            pred = t['predicate']
            obj = t['object']
            if not G.has_node(subject):
                G.add_node(subject, label=subject)
            if not G.has_node(obj):
                G.add_node(obj, label=obj)
            edge_data = {"label": pred, "title": pred}
            if "original_subject" in t: edge_data["original_subject"] = t["original_subject"]
            if "original_object" in t: edge_data["original_object"] = t["original_object"]
            if "quote" in t: edge_data["quote"] = t["quote"]
            if "certainty_score" in t: edge_data["certainty_score"] = t["certainty_score"]
            G.add_edge(subject, obj, **edge_data)
        return G

    def process(self, triples: List[Dict[str, str]], graphs_dir: str = None) -> nx.DiGraph:
        """
        Main execution pipeline for Phase 2.
        """
        if not triples:
            logger.warning("No triples provided for processing.")
            return nx.DiGraph()

        # Save initial population graph
        if graphs_dir:
            initial_graph = self._build_graph(triples)
            self.save_visualization(initial_graph, os.path.join(graphs_dir, "01_initial_population.html"))

        # Step 1: Semantic Compression (Option B) if enabled
        if self.use_embeddings and self.embed_service:
            processed_triples, nlp_logs = self.embed_service.semantic_compression(triples)
            
            if graphs_dir:
                schemas_dir = graphs_dir.replace("graphs", "schemas")
                os.makedirs(schemas_dir, exist_ok=True)
                
                self.export_nlp_clusters(nlp_logs, os.path.join(schemas_dir, "nlp_entity_clusters.csv"))
                self.export_triplet_transformations(processed_triples, os.path.join(schemas_dir, "nlp_triplet_transformations.csv"))
                
                compressed_graph = self._build_graph(processed_triples)
                self.save_visualization(compressed_graph, os.path.join(graphs_dir, "02_semantic_compression.html"))
        else:
            processed_triples = triples
            
        # Step 2: Build Directed Graph
        logger.info("Building Core Directed Graph...")
        G = self._build_graph(processed_triples)
        logger.info(f"Graph initialized with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        
        # Step 3: Topological Clustering (Option A)
        logger.info(f"Running Community Detection ({self.community_detection})...")
        
        # Louvain typically requires an undirected graph
        undirected_G = G.to_undirected()
        
        partition = {}
        if self.community_detection == "louvain" and community_louvain:
            partition = community_louvain.best_partition(undirected_G)
        elif self.community_detection == "leiden":
            try:
                from cdlib import algorithms
                coms = algorithms.leiden(undirected_G)
                partition = coms.to_node_community_map()
                # cdlib map returns {node: [community_id, ...]}
                partition = {k: v[0] for k, v in partition.items()}
            except ImportError:
                logger.error("cdlib not installed. Falling back to simple partition.")
        else:
            logger.warning("No valid community detection chosen. Proceeding without partitions.")

        # Assign partition to nodes
        for node, comm_id in partition.items():
            G.nodes[node]['group'] = comm_id
            
        return G

    def save_visualization(self, G: nx.DiGraph, output_file: str):
        """
        Saves the graph as an interactive HTML file with a highly sleek, modern design.
        """
        logger.info(f"Saving sleek graph visualization to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Calculate degrees to dynamically scale node sizes
        if G.number_of_nodes() > 0:
            degrees = dict(G.degree())
            max_deg = max(degrees.values()) if max(degrees.values()) > 0 else 1
            for node in G.nodes():
                # Make nodes scale aesthetically between size 10 and 40
                G.nodes[node]['size'] = 10 + (30 * (degrees[node] / max_deg))
                
                # Add hovering tooltips showing the entity data
                group = G.nodes[node].get('group', 'Unclustered')
                G.nodes[node]['title'] = f"Entity: {node}<br>Community: {group}<br>Degree: {degrees[node]}"
        
        # Initialize pyvis network with rich dark-mode aesthetics
        net = Network(height="800px", width="100%", bgcolor="#0F111A", font_color="#E2E8F0", directed=True)
        
        net.from_nx(G)
        
        # Apply premium VisJS configuration
        options = """var options = {
          "nodes": {
            "borderWidth": 1.5,
            "borderWidthSelected": 4,
            "shape": "dot",
            "shadow": {
              "enabled": true,
              "color": "rgba(0,0,0,0.6)",
              "size": 12,
              "x": 3,
              "y": 3
            },
            "font": {
              "size": 14,
              "face": "Inter, Roboto, sans-serif",
              "strokeWidth": 2,
              "strokeColor": "rgba(0,0,0,0.8)"
            }
          },
          "edges": {
            "color": {
              "inherit": "both",
              "opacity": 0.45
            },
            "smooth": {
              "type": "continuous",
              "roundness": 0.6
            },
            "arrows": {
              "to": { "enabled": true, "scaleFactor": 0.5 }
            }
          },
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -80,
              "centralGravity": 0.015,
              "springLength": 120,
              "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based",
            "timestep": 0.4
          }
        }"""
        net.set_options(options)
        
        # If the graph is huge, disable physics after initial simulation to preserve browser rendering
        if G.number_of_nodes() > 700:
            net.toggle_physics(False)
            
        net.save_graph(output_file)
        
    def export_schemas(self, G: nx.DiGraph, output_file: str):
        """
        Exports the distinct logic clusters into a summarized tabular/JSON format 
        that can be fed into Phase 3 (Pydantic Mapping).
        """
        logger.info(f"Exporting logic schemas to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        schema_data = []
        for u, v, data in G.edges(data=True):
            schema_data.append({
                "subject": u,
                "subject_cluster": G.nodes[u].get("group", -1),
                "original_subject": data.get("original_subject", ""),
                "predicate": data.get("label", ""),
                "object": v,
                "object_cluster": G.nodes[v].get("group", -1),
                "original_object": data.get("original_object", ""),
                "quote": data.get("quote", ""),
                "certainty_score": data.get("certainty_score", "")
            })
            
        df = pd.DataFrame(schema_data)
        
        if output_file.endswith(".csv"):
            df.to_csv(output_file, index=False)
        elif output_file.endswith(".json"):
            df.to_json(output_file, orient='records', indent=4)
        else:
            df.to_csv(output_file + ".csv", index=False)

    def export_nlp_clusters(self, nlp_logs: List[Dict[str, str]], output_file: str):
        """
        Exports the raw embedding clusters dictionary natively to CSV.
        """
        if not nlp_logs: return
        logger.info(f"Exporting NLP cluster states to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df = pd.DataFrame(nlp_logs)
        df.to_csv(output_file, index=False)
        
    def export_triplet_transformations(self, compressed_triples: List[Dict[str, str]], output_file: str):
        """
        Exports a dedicated side-by-side comparison of triple strings before and after semantic compression.
        """
        if not compressed_triples: return
        logger.info(f"Exporting triplet transformations to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # We explicitly ensure ordering for readability
        formatted = []
        for t in compressed_triples:
            formatted.append({
                "original_subject": t.get("original_subject", ""),
                "final_subject": t.get("subject", ""),
                "original_predicate": t.get("original_predicate", ""),
                "final_predicate": t.get("predicate", ""),
                "original_object": t.get("original_object", ""),
                "final_object": t.get("object", ""),
                "quote": t.get("quote", ""),
                "certainty_score": t.get("certainty_score", "")
            })
            
        df = pd.DataFrame(formatted)
        df.to_csv(output_file, index=False)
