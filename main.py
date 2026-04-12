import argparse
from src.orchestrator.pipeline import PipelineOrchestrator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IsoKernel Knowledge Graph Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--input", type=str, nargs='+', required=True, help="Path to input text file or directory")
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step progress and summaries to console")
    
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator(config_path=args.config, verbose=args.verbose)
    orchestrator.run(args.input)
