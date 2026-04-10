class Prompts:
    DISCOVERY_SYSTEM = """
    You are an unconstrained Triple Extractor Agent running a schema-less Discovery phase.
    Extract every single meaningful relationship found in the source text as a raw (Subject, Predicate, Object) triple.
    Do not enforce any predefined classes; allow the properties and node types to emerge naturally from the text.
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
    def get_discovery_user(text_content: str) -> str:
        return f"""
        Extract the schema-less triplets from the following text:
        
        <source_text>
        {text_content}
        </source_text>
        """

    HYPERNYM_SYSTEM = """
    You are an intelligent linguistic analyzer tasked with Semantic Resolution.
    You will receive a JSON dictionary where the keys are cluster IDs, and the values are lists of string variants extracted from text (nouns and verbs).
    For EACH cluster, your job is to return a SINGLE generalized, standardized string that acts as the best hypernym or canonical representation for the group.
    
    Rules for singletons (lists with 1 string):
    - Clean up capitalization, fix grammatical noise, and return the generalized form (e.g. ['is working with'] -> 'works with', or ['The Google corp'] -> 'Google').
    
    Rules for larger clusters:
    - Return the best encompassing label that unifies all the noise (e.g. ['Google LLC', 'google', 'The search giant'] -> 'Google').
    """

    @staticmethod
    def get_hypernym_user(cluster_map_json: str) -> str:
        return f"""
        Map the following clusters to their ideal canonical hypernym:
        
        <clusters>
        {cluster_map_json}
        </clusters>
        """

    TAXONOMIC_LIFTING_SYSTEM = """
    You are an absolute Taxonomic Categorization Engine. 
    You will receive JSON arrays indicating a primary `centroid` string, grouped logically with its associated string variants `members`.
    
    CRITICAL OBJECTIVE: You must perform formal 'Taxonomic Lifting' to abstract the geometric `centroid` up one topological level to its perfect formal categorical class!
    
    STRICTNESS CONSTRAINTS:
    1. Your Formal Hypernym MUST be an explicit, mathematically sound NOUN PHRASE (for subjects) or ADJECTIVE/VERB categorical abstraction if it is an action grouping.
    2. DEDUCTIVE 'IS-A' TESTING: You must rigidly test your proposed Formal Hypernym sequentially against ALL elements inside `members`. 
       - Example logic: If candidate Formal Hypernym is "Programming Language" and members are [Python, Java, Rust], you must formally deduce: 
         "Is Python a Programming Language? (Yes). Is Java a Programming Language? (Yes)."
       - If any element logically drops the strict deductive constraint test, you must forcibly set `members_verified` to FALSE.
    3. If verification inherently fails structurally across the array, formally reject the taxonomic lift.
    """

    @staticmethod
    def get_taxonomic_user(taxonomic_map_json: str) -> str:
        return f"""
        Execute rigorous mathematical and linguistic taxonomic lifting on the following centroid groupings:
        
        <taxonomic_clusters>
        {taxonomic_map_json}
        </taxonomic_clusters>
        """
