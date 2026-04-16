import os
import re
import psutil
import subprocess
import logging

logger = logging.getLogger("context_manager")

class ContextManager:
    @staticmethod
    def get_system_vram_gb() -> float:
        """
        Calculates total VRAM across GPUs or falls back to Unified System Memory.
        """
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                encoding='utf-8', timeout=2
            )
            vram_mb = sum([int(x.strip()) for x in output.strip().split('\n') if x.strip().isdigit()])
            if vram_mb > 0:
                return vram_mb / 1024.0
        except Exception:
            pass
            
        # Fallback natively to system RAM assuming Unified Apple Silicon or Grace Hopper without explicit SMI mapping.
        return psutil.virtual_memory().total / (1024.0 ** 3)

    @staticmethod
    def parse_params_billions(model_name: str) -> float:
        """
        Regex isolates parameter counts implicitly out of LLM titles.
        """
        if not model_name:
            return 8.0
            
        match = re.search(r'(\d+(?:\.\d+)?)[bB]', model_name)
        if match:
            return float(match.group(1))
        
        # Heuristic fallback if standard parameter tag is missing
        return 8.0

    @staticmethod
    def get_safe_context_tokens(model_name: str, safety_gb: float = 4.0) -> int:
        """
        Calculates strict maximal cache limits structurally matching system capabilities.
        """
        vram_gb = ContextManager.get_system_vram_gb()
        params_b = ContextManager.parse_params_billions(model_name)
        
        # Estimate weight footprint for local 4-bit configurations cleanly
        weight_gb = params_b * 0.70
        
        headroom_gb = vram_gb - weight_gb - safety_gb
        if headroom_gb <= 0:
            logger.warning(f"Projected Memory Bounds ({vram_gb:.1f}GB) extremely tight against {params_b}B active model limits + OS footprints. Operating on minimal contexts natively.")
            return 1024
            
        headroom_mb = headroom_gb * 1024
        
        # Memory caching mapping roughly correlates token-states heavily into size variants mapping dimensions intrinsically
        kv_cost_mb = params_b * 0.0075 
        if kv_cost_mb <= 0:
            kv_cost_mb = 0.05
            
        max_tokens = int(headroom_mb / kv_cost_mb)
        
        # Architecture mechanical bandwidth caps securely capping infinite sizes
        if max_tokens > 64000:
            max_tokens = 64000
            
        return max_tokens

    @staticmethod
    def calculate_max_chunk_words(model_name: str, safety_gb: float = 4.0) -> int:
        """
        Dynamically limits context extraction windows securely to system hardware metrics.
        """
        max_tokens = ContextManager.get_safe_context_tokens(model_name, safety_gb)
        max_words = int(max_tokens * 0.75)
        
        if max_words < 500:
            return 500
            
        return max_words
