from src.orchestrator.pipeline import PipelineOrchestrator
import json
import logging

logging.basicConfig(level=logging.INFO)

config_path = "config.yaml"
orchestrator = PipelineOrchestrator(config_path=config_path, domain="knowledge")
orchestrator.config["pipeline"]["run_phase_1"] = False
orchestrator.config["pipeline"]["run_phase_2"] = True
orchestrator.verbose = True

with open("outputs/schemas/phase1_raw_triples.json", "r") as f:
    triples = json.load(f)

print(f"Loaded {len(triples)} triples manually")
orchestrator.refine_graph(triples)
print("done")
