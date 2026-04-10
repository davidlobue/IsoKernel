import os
from typing import List
import instructor
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv

from src.core.logger import setup_logger
from src.core.models import DocumentSource, RawTriple, TripleExtractionResult
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

        if self.provider == "local":
            logger.info("Configuring local LLM provider via AsyncOpenAI API schema...")
            client = AsyncOpenAI(base_url=self.base_url, api_key="ollama")
            self.async_client = instructor.from_openai(client, mode=instructor.Mode.JSON)
            
        elif self.provider == "google":
            logger.info("Configuring Google Vertex provider...")
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel
                model_instance = GenerativeModel(self.model_name)
                self.async_client = instructor.from_vertexai(model_instance, mode=instructor.Mode.VERTEXAI_TOOLS)
            except ImportError as e:
                logger.error("google-cloud-aiplatform is required for vertexai")
                raise e
                
        else: # Default openAI
            logger.info("Configuring Async OpenAI provider...")
            self.async_client = instructor.from_openai(AsyncOpenAI())
            
        logger.info(f"Initialized TripleExtractor for domain '{self.domain}' using model '{self.model_name}' via '{self.provider}'")

    async def extract_raw_triples(self, document: DocumentSource) -> TripleExtractionResult:
        """
        Uses Instructor to extract raw, unconstrained semantic Triples.
        """
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": Prompts.DISCOVERY_SYSTEM},
                {"role": "user", "content": Prompts.get_discovery_user(document.text_content)}
            ],
            response_model=TripleExtractionResult
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
