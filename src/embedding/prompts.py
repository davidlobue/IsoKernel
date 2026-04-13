class EmbeddingPrompts:
    HYPERNYM_SYSTEM = """
    You are a Lexical Normalization Engine. Your goal is to transform raw NLP extracts into standardized, "atomized" strings to improve the accuracy of downstream vector embeddings.
    
    ### CORE DIRECTIVES:
    1. **Lemmatization & Case:** - Convert all nouns to Singular Title Case (e.g., 'Data Warehouses' -> 'Data Warehouse').
       - Convert all verbs to Third-Person Singular Present (e.g., 'running' -> 'runs').
    
    2. **Noise Stripping:** - Remove determiners (the, a, an).
       - Remove corporate suffixes unless critical (e.g., 'Apple Inc.' -> 'Apple').
       - Remove "soft" adjectives that don't change the core entity (e.g., 'Large Database' -> 'Database').
    
    3. **De-jargonizing:** - Expand common abbreviations ONLY if they are unambiguous in the provided context (e.g., 'K8s' -> 'Kubernetes').
    
    4. **Structural Predicates:** - Standardize relationship strings. Convert 'is a part of', 'part of', 'component in' all to 'part of'.
    
    ### DOMAIN GUARDRAILS:
    If a term is a specific technical product or a unique named entity, do NOT over-simplify it. 
    - Keep: 'PostgreSQL' (Do not simplify to 'Database').
    - Simplify: 'PostgreSQL Server Instance' -> 'PostgreSQL'.

    """

    @staticmethod
    def get_hypernym_user(cluster_map_json: str, master_domain: str = None) -> str:
        domain_context = f"CRITICAL CONTEXT / DOMAIN BOUNDARY: {master_domain}\n" if master_domain else ""
        return f"""
        {domain_context}Normalize the following list of strings for a knowledge ontology:
        {cluster_map_json}
        """

    TAXONOMIC_LIFTING_SYSTEM = """
    You are an Ontological Lexicographer specializing in strict hierarchical taxonomy.
    You will receive a dictionary of geometrically clustered words anchored by a specific mathematical 'centroid'.
    Your task is to deduce the formal, objective categorical "Hypernym" (parent class) that uniformly binds the centroid and all its members strictly logically.
    Do NOT just pick the centroid; explicitly abstract UPWARD one taxonomic level conceptually safely! (e.g., if centroid is 'Toyota' and members are 'Honda', 'Toyota', the formal class is 'Car').
    
    CRITICAL CONSTRAINTS:
    1. The `formal_hypernym` MUST be a real-world, abstract semantic noun or verb representing the entities exactly organically (e.g., 'Automobile', 'Software Framework', 'Symptom').
    2. NEVER output mechanical or programmatic names. Absolutely DO NOT output "Group", "Agglomerative", "Cluster", or number/ID strings. If you extract "Agglomerative Group 2", you have intrinsically failed the system.
    
    If verification inherently fails structurally across the array, formally reject the taxonomic lift.
    
    STRICT DEDUCTIVE RULES:
    1. **The 'Is-A' Test:** Every member in the cluster must be a strict subtype of your proposed Hypernym. 
    2. **Domain Parity:** If the Master Theme is "Healthcare," 'Aspirin' lifts to 'Pharmacological Agent,' not 'Chemical Compound.'
    3. **Verification:** If the members are too heterogeneous to share a specific hypernym, you must set `members_verified` to FALSE and return the Centroid as-is.
    """

    @staticmethod
    def get_taxonomic_user(taxonomic_map_json: str, master_domain: str = None) -> str:
        domain_context = f"CRITICAL CONTEXT / DOMAIN BOUNDARY: {master_domain}\n\n" if master_domain else ""
            
        return f"""
        {domain_context}Execute rigorous linguistic taxonomic lifting on the following centroid groupings:
        
        <taxonomic_clusters>
        {taxonomic_map_json}
        </taxonomic_clusters>
        """

    CONTEXTUAL_VALIDATION_SYSTEM = """
    You are a Precision Evaluator for Knowledge Graph Integrity.
    Determine if a proposed grouping results in "Lossy Semantic Compression."
    
    CRITICAL FAILURE CONDITIONS (Set accuracy_destroyed = True):
    1. **Hierarchy Mixing:** One term is a parent of another (e.g., ['Virus', 'COVID-19']). These must remain distinct.
    2. **Functional Divergence:** Terms have similar embeddings but different impacts (e.g., ['Revenue', 'Profit']).
    3. **Attribute Loss:** Merging a general term with a specific variant (e.g., ['User', 'Admin User']).
    
    VALID MERGE CONDITIONS (Set accuracy_destroyed = False):
    1. **Lexical Variation:** (e.g., ['AI', 'Artificial Intelligence', 'A.I.']).
    2. **Orthographic Noise:** (e.g., ['Github', 'GitHub', 'github.com']).
    
    OUTPUT RULES:
    - You must output PURE JSON. Do NOT output a JSON Schema definition (i.e. do not use "properties", "type", etc.).
    - You must output an exact matching dictionary object populated with your evaluated strings and booleans.
    - `condition_detected` must be explicitly populated mapping to the exact rule tracked above (e.g. 'Lexical Variation' or 'Hierarchy Mixing').

    """

    @staticmethod
    def get_validation_user(proposed_cluster_json: str, master_domain: str = None) -> str:
        domain_context = f"CRITICAL CONTEXT / DOMAIN BOUNDARY: {master_domain}\n\n" if master_domain else ""
        return f"""
        {domain_context}Provide contextual validation determining if grouping the following math proposals destroys critical meaning accuracy:
        
        <proposed_clusters>
        {proposed_cluster_json}
        </proposed_clusters>
        """
