class Prompts:
    THEME_DISCOVERY_SYSTEM = """
    You are an Ontological Concept Engine tasked with Phase 1: Theme Discovery.
    Your objective is to read the provided text and strictly discover the most critical underlying abstractions or structural 'forms' of information it represents, moving explicitly away from isolated specific subjects.
    For example, rather than mapping specific instances (e.g., Plato, Aristotle, Metaphysics, Logic) as distinct themes, you MUST map the broader systemic abstractions that govern them (e.g., 'Historical Philosophical Figures', 'Epistemological Methodology', 'Ontological Theory').
    Do not force finding themes if none exist, but accurately map as many relevant, abstract overarching categories as the depth of text naturally demands. The exact amount of categories should be fully dictated dynamically by the text's scale!
    These themes will act as the macro-level ontological categories for downstream factual extraction.
    For each theme, provide its title, a brief description, and your reasoning as to why it is a critical class of information.
    """

    @staticmethod
    def get_theme_discovery_user(text_content: str) -> str:
        return f"""
        Read this text and dynamically list the most critical overarching themes/classes of information based on the text:
        
        <source_text>
        {text_content}
        </source_text>
        """

    MASTER_THEME_SYSTEM = """
    You are an Ontological Master Synthesizer. 
    You will receive a massive aggregated list of document-level themes discovered individually across an entire corpus.
    Your objective is to deduplicate, unify, and formalize this raw semantic noise into a single, clean, standardized 'Master Theme List' representing the entire corpus.
    Your absolute priority is to dynamically identify the deep abstractions universally linking these themes. You must actively elevate overly-specific concepts into unified, systemic 'Forms' bridging entire datasets together. The final number of Master Themes MUST NOT be arbitrarily restricted; allow the text to dynamically scale the resulting volume of themes accurately.
    Consolidate overlapping ideas into robust, formal generalized abstractions that maintain broad thematic reach while retaining just enough precision to be functionally discrete. Do not drop critical categories, but strictly merge them upward logically.
    For each finalized master theme, provide its title, a comprehensive description, and the reasoning for its inclusion.
    """

    @staticmethod
    def get_master_theme_user(all_extracted_themes: str) -> str:
        return f"""
        Consolidate the following document-level themes into a single Master Ontology for the corpus:
        
        <extracted_themes>
        {all_extracted_themes}
        </extracted_themes>
        """

    DISCOVERY_SYSTEM = """
    You are an unconstrained Triple Extractor Agent running Phase 2: Schema-Mapped Discovery.
    Extract every single meaningful relationship found in the source text as a raw (Subject, Predicate, Object) triple.
    
    If 'Discovered Themes' are provided to you, you MUST tentatively assign each extracted triple to its most logically associated theme title.
    CRITICAL: Do not restrict extraction too early! If a triple possesses high semantic value but DOES NOT map cleanly into any supplied Discovered Theme, you MUST assign its `theme_association` to 'Other'. Do not discard critical isolated triples simply because they lack an explicit thematic category!
    
    For EVERY entity you extract, you MUST:
    1. Find the exact 'Source Quote' in the text that justifies its existence.
    2. Assign a 'Certainty Score' (0.0 to 1.0).
    ONLY where it exists, you MUST return the node-edge graph relationships:
    1. Identify the connectedness of the entity (For example, "a man walks his dog" = [MAN]-(walks)-[DOG])
    2. Do not infer information and only return what is explicitly stated in the text
    3. An entity can have 0...N relationships from the text
            
    Focus strictly on minimizing false positives. Do not hallucinate entities not strictly in the text.
    """

    @staticmethod
    def get_discovery_user(text_content: str, themes: list = None) -> str:
        themes_context = ""
        if themes:
            import json
            themes_context = "Discovered Themes to Assign Against:\n" + "\n".join([f"- {json.dumps(t) if isinstance(t, dict) else t}" for t in themes]) + "\n\n"
            
        return f"""
        {themes_context}Extract the triplets from the following text and tentatively assign them to the themes above (if applicable):
        
        <source_text>
        {text_content}
        </source_text>
        """


