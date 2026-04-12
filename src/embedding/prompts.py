class EmbeddingPrompts:
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

    CONTEXTUAL_VALIDATION_SYSTEM = """
    You are an incredibly strict Semantic Context Verifier. 
    You will receive an array of string entities that a mathematical clustering engine proposes grouping into a single unified semantic concept.
    Your ONLY job is to determine: "Does merging these entities conceptually group them in a way that destroys critical distinguishing accuracy needed for our resulting schema?"
    
    If merging them technically works mathematically but functionally ruins contextual precision (e.g., merging ['Type 1 Diabetes', 'Diabetes']), you MUST set accuracy_destroyed = True.
    If merging them preserves strict semantic equality seamlessly identically (e.g., merging ['patient', 'client', 'the patient']), you MUST set accuracy_destroyed = False.
    """

    @staticmethod
    def get_validation_user(proposed_cluster_json: str) -> str:
        return f"""
        Provide contextual validation determining if grouping the following math proposals destroys critical accuracy:
        
        <proposed_clusters>
        {proposed_cluster_json}
        </proposed_clusters>
        """
