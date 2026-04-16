import os
from typing import List
import instructor
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv

from src.core.logger import setup_logger
from src.core.models import DocumentSource, RawTriple, TripleExtractionResult, ThemeDiscoveryResult, MasterThemeSynthesisResult
from src.extraction.prompts import Prompts

# Load environment variables (finds .env recursively starting from cwd)
load_dotenv(find_dotenv())

logger = setup_logger("phase1_extractor")



class TripleExtractor:
    def __init__(self, model: str = None, domain: str = "general knowledge"):
        self.domain = domain
        
        # Load from overrides or environment
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.model_name = model or os.getenv("LLM_MODEL_NAME", "gpt-4o")
        self.base_url = os.getenv("LLM_BASE_URL", None)

        self._raw_client = None
        
        if self.provider == "local":
            logger.info("Configuring local LLM provider via AsyncOpenAI API schema...")
            self._raw_client = AsyncOpenAI(base_url=self.base_url, api_key="ollama")
            self.async_client = instructor.from_openai(self._raw_client, mode=instructor.Mode.JSON)
            
            from src.orchestrator.context_manager import ContextManager
            safe_tokens = ContextManager.get_safe_context_tokens(self.model_name)
            self._base_extra_kwargs = {
                "extra_body": {
                    "keep_alive": -1,
                    "options": {"num_ctx": safe_tokens}
                }
            }
            
        else: # Default openAI
            logger.info("Configuring Async OpenAI provider...")
            self._raw_client = AsyncOpenAI()
            self.async_client = instructor.from_openai(self._raw_client)
            self._base_extra_kwargs = {}
            
        logger.info(f"Initialized TripleExtractor for domain '{self.domain}' using model '{self.model_name}' via '{self.provider}'")

    async def close(self):
        """Explicitly handles closing the underlying HTTP connection pools securely."""
        if self._raw_client is not None:
            try:
                await self._raw_client.close()
                logger.info("Explicitly closed LLM Async client connections.")
            except Exception as e:
                logger.warning(f"Failed to cleanly close LLM connections: {e}")

    async def extract_themes(self, document: DocumentSource) -> ThemeDiscoveryResult:
        """
        Uses Instructor to execute 'Pass A': Discovering macro-themes.
        """
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": Prompts.THEME_DISCOVERY_SYSTEM},
                {"role": "user", "content": Prompts.get_theme_discovery_user(document.text_content)}
            ],
            response_model=ThemeDiscoveryResult,
            **self._base_extra_kwargs
        )
        return response

    async def consolidate_themes(self, all_themes: list) -> MasterThemeSynthesisResult:
        """
        Uses Instructor to execute 'Pass A.5': Consolidating macro-themes into a master list.
        """
        formatted_themes = json.dumps(all_themes, indent=2)
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": Prompts.MASTER_THEME_SYSTEM},
                {"role": "user", "content": Prompts.get_master_theme_user(formatted_themes)}
            ],
            response_model=MasterThemeSynthesisResult,
            **self._base_extra_kwargs
        )
        return response

    async def extract_raw_triples(self, document: DocumentSource, themes: list = None) -> TripleExtractionResult:
        """
        Uses Instructor to execute 'Pass B': Extracting raw semantic Triples and routing them to themes.
        """
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": Prompts.DISCOVERY_SYSTEM},
                {"role": "user", "content": Prompts.get_discovery_user(document.text_content, themes)}
            ],
            response_model=TripleExtractionResult,
            **self._base_extra_kwargs
        )
        return response

    def format_as_markdown(self, triples: List[dict]) -> str:
        """
        Formats a list of triple dictionaries into a Markdown table.
        """
        md = "| subject | predicate | object |\n"
        md += "| :--- | :--- | :--- |\n"
        for t in triples:
            md += f"| {t['subject']} | {t['predicate']} | {t['object']} |\n"
        return md
