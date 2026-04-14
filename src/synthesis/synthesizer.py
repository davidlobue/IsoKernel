import os
import json
import logging
import asyncio
import instructor
import networkx as nx
from openai import AsyncOpenAI
from src.synthesis.prompts import SynthesisPrompts
from src.core.models import GeneratedSchema
from src.core.utils import run_sync

logger = logging.getLogger(__name__)

class SchemaSynthesizer:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = self.config.get("output", {}).get("schemas_dir", "outputs/schemas_code")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def extract_communities(self, G: nx.DiGraph) -> dict:
        """
        Parses the graph topology and regroups logic strictly by isolated networkx community metrics.
        Returns a dict mapping community_id -> explicit nx.DiGraph (Subgraph) preserving all mathematical constraints.
        """
        communities = {}
        # Pre-initialize subgraphs
        for node, data in G.nodes(data=True):
            grp = data.get('group', 'Unclustered')
            if grp not in communities:
                communities[grp] = nx.DiGraph()
            communities[grp].add_node(node, **data)
            
        # Migrate strictly bounded edges securely
        for u, v, data in G.edges(data=True):
            group_u = G.nodes[u].get('group')
            if group_u is not None and group_u in communities:
                communities[group_u].add_edge(u, v, **data)
                
        return communities
        
    def package_payload(self, subG: nx.DiGraph) -> dict:
        """
        Parses strictly typed roles recursively.
        """
        if subG.number_of_nodes() == 0:
            return {}
            
        root_entities = []
        nested_entities = []
        terminal_attributes = []
        
        # Categorize native Graph objects securely
        for n, data in subG.nodes(data=True):
            role = data.get('class_role', 'Unclustered')
            if role == "RootEntity": root_entities.append(n)
            elif role == "NestedEntity": nested_entities.append(n)
            elif role == "TerminalAttribute": terminal_attributes.append(n)
            
        # Optional fallback if no strict Root mapped
        if not root_entities:
            degrees = dict(subG.out_degree())
            if degrees:
                fallback_master = max(degrees, key=degrees.get)
                root_entities.append(fallback_master)
                
        # Calculate exactly how the roots map natively outwards
        hierarchy = {
            "root_classes": {},
            "nested_classes": {}
        }
        
        # We loop root nodes
        for root in root_entities:
            hierarchy["root_classes"][root] = {"nested_bridges": {}, "attributes": {}}
            for u, v, data in subG.out_edges(root, data=True):
                pred = data.get('label', '')
                if v in nested_entities:
                    if pred not in hierarchy["root_classes"][root]["nested_bridges"]: hierarchy["root_classes"][root]["nested_bridges"][pred] = []
                    hierarchy["root_classes"][root]["nested_bridges"][pred].append(v)
                elif v in terminal_attributes:
                    if pred not in hierarchy["root_classes"][root]["attributes"]: hierarchy["root_classes"][root]["attributes"][pred] = []
                    hierarchy["root_classes"][root]["attributes"][pred].append(v)
                    
        # Provide the same bridging logic specifically for sub_classes resolving internally
        for nested in nested_entities:
            hierarchy["nested_classes"][nested] = {"nested_bridges": {}, "attributes": {}}
            for u, v, data in subG.out_edges(nested, data=True):
                pred = data.get('label', '')
                if v in terminal_attributes:
                    if pred not in hierarchy["nested_classes"][nested]["attributes"]: hierarchy["nested_classes"][nested]["attributes"][pred] = []
                    hierarchy["nested_classes"][nested]["attributes"][pred].append(v)
                elif v in nested_entities:
                    if pred not in hierarchy["nested_classes"][nested]["nested_bridges"]: hierarchy["nested_classes"][nested]["nested_bridges"][pred] = []
                    hierarchy["nested_classes"][nested]["nested_bridges"][pred].append(v)
                    
        return hierarchy

    async def _generate_class(self, comm_id, payload, async_client):
        try:
            logger.info(f"Synthesizing Topologic Community {comm_id} -> Pydantic Models natively...")
            data_json = json.dumps(payload, indent=2)
            
            res = await async_client.chat.completions.create(
                model=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
                messages=[
                    {"role": "system", "content": SynthesisPrompts.SCHEMA_GENERATION_SYSTEM},
                    {"role": "user", "content": SynthesisPrompts.get_schema_user(data_json)}
                ],
                response_model=GeneratedSchema
            )
            return comm_id, res.class_name, res.python_code
        except Exception as e:
            logger.error(f"Generation mathematically failed for community {comm_id}: {e}")
            return comm_id, None, None

    def synthesize_schemas(self, G: nx.DiGraph):
        logger.info("Initializing Phase 4: Pydantic Code Synthesizer explicitly over network bounds...")
        communities = self.extract_communities(G)
        
        valid_payloads = {}
        for comm_id, edges in communities.items():
            payload = self.package_payload(edges)
            if payload and (payload.get("root_classes") or payload.get("nested_classes")):
                valid_payloads[comm_id] = payload
                
        if not valid_payloads:
            logger.warning("No cohesive topographical communities detected for schema translation natively.")
            return []
            
        # Checkpoint: Save JSON mappings natively before transmitting to LLM
        payloads_out = os.path.join(self.output_dir, "phase3_community_payloads.json")
        with open(payloads_out, "w", encoding="utf-8") as f:
            json.dump(valid_payloads, f, indent=4)
        logger.info(f"Persisted Phase 3 Semantic Payloads strictly into {payloads_out}")
            
        async def _run_synthesis():
            max_concurrent = self.config.get("pipeline", {}).get("max_concurrent_llm_calls", 3)
            sem = asyncio.Semaphore(max_concurrent)
            
            client = AsyncOpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key="ollama")
            async_client = instructor.from_openai(client, mode=instructor.Mode.JSON)
            
            async def _generate_single(c_id, p_load):
                async with sem:
                    return await self._generate_class(c_id, p_load, async_client)
                    
            tasks = [_generate_single(c_id, p_load) for c_id, p_load in valid_payloads.items()]
            _res = await asyncio.gather(*tasks)

            try:
                await client.close()
            except Exception:
                pass

            try:
                import urllib.request
                url = f"{os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1').replace('/v1', '/api')}/generate"
                model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o")
                data = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
                urllib.request.urlopen(req, timeout=2.0)
                logger.info(f"Cleared Memory & Ollama VRAM after Schema Synthesis.")
            except Exception:
                pass

            return _res

        results = run_sync(_run_synthesis)

        written_files = []
        master_file_path = os.path.join(self.output_dir, "schemas.py")
        
        # Declare standard overarching headers
        master_code = "from enum import Enum\nfrom pydantic import BaseModel, Field\nfrom typing import Optional, List\n\n"
        
        has_content = False
        import re
        
        for comm_id, class_name, python_code in results:
            if class_name and python_code:
                has_content = True
                code = python_code.replace("```python", "").replace("```", "").strip()
                
                # Strip out redundant header imports the LLM generated identically
                code = re.sub(r'from enum import Enum', '', code)
                code = re.sub(r'from pydantic import .*', '', code)
                code = re.sub(r'from typing import .*', '', code)
                code = code.strip()
                
                master_code += f"# ====== Schema Generated for Community: {comm_id} ======\n"
                master_code += f"{code}\n\n"
                logger.info(f"Synthesized pure Pydantic architecture: {class_name}")
                
        if has_content:
            with open(master_file_path, "w", encoding="utf-8") as f:
                f.write(master_code)
            written_files.append(master_file_path)
            logger.info(f"Consolidated final Pydantic mappings mapped centrally -> {master_file_path}")
            
        return written_files
