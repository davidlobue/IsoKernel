
Project "IsoKernel" Executive Summary: Modular Knowledge Graph Pipeline
Overall Goal: To build an extensible, multi-phase system that transforms unstructured text into highly structured, validated data schemas. The pipeline moves from raw linguistic extraction to topological refinement, eventually terminating in fixed data classes.
Tools:  highly modular python with only required external package dependencies (keep it slim)- (e.g., networkx, Instructor, pydantic, openai for local LLM calls).  Use a master "config" file where possible to simplify user interaction/overrides.
Style:  Single pipeline "orchestrator" for streamlined processes of modular imports for specialized function calls.  Include main.py as well as a jupyter notebook for testing components.  Include detailed logs in a single location but differentiable by task/phase.

Phase 1 and 2 are detailed below.  Additional Phases will be added in the future.

Phase 1: Schema-Less Triple Extraction using LLM API calls
Focus: High-fidelity linguistic parsing.
Action: Uses chain-of-thought prompting to extract atomic, snake_case semantic triples (Subject-Predicate-Object) into Markdown tables.
Key Output: A "raw" topological map of the document that resolves coreferences and preserves evidence provenance.

Phase 2: Refinement & Logic Clustering
Focus: Structural and semantic optimization.
Action: A modular Python engine (networkx) that applies community detection (Louvain/Leiden) and vector embeddings to resolve anonymized entity collisions and collapse redundant nodes into hypernyms.
Key Output: A reduced, "logic-dense" schema ready for mapping into Phase 3 (Pydantic Class Extraction).
Architectural Principle: The system must remain modular. Every stage should allow for independent execution or toggled features (like embeddings) to ensure efficiency as the project expands into subsequent phases.


Phase 1: Schema-Less Triple Extraction

Objective: To extract triples (Subject-Predicate-Object) for a document knowledge ontology, focusing on structural integrity, semantic clarity, and schema adherence.
Purpose: Scans input documents to extract raw topological triples (unguided relationships and entities) without relying on any predefined structure.

The "optimal" prompt uses Few-Shot Prompting and Chain-of-Thought reasoning to ensure the AI doesn't just pull random sentences, but identifies the underlying logic of the text.
Information Extraction Triple Prompt (Domain will be a field optionally provided by the user):
### ROLE
You are an expert Ontologist and Knowledge Graph Engineer specializing in data extraction for [domain].

### TASK
Analyze the provided text and extract formal semantic triples (Subject-Predicate-Object) to build a Knowledge Graph Ontology.

### EXTRACTION RULES
1. **Granularity:** Break complex sentences into atomic triples.
2. **Coreference Resolution:** Replace all pronouns (e.g., "it", "this system", "they") with the specific Named Entity.
3. **Naming Convention:** Use snake_case for all Subjects, Predicates, and Objects (e.g., technical_manual, authored_by, safety_protocol).

### ONTOLOGY GUIDELINES
1. SUBJECT: Must be a specific entity, concept, or document element.
2. PREDICATE: Must be a relationship verb in snake_case. Use consistent relationship types.
3. OBJECT: Must be another entity, a literal value, or a conceptual attribute.

### OUTPUT FORMAT
Return only a Markdown table with the following columns:
| subject | predicate | object | evidence_quote |
| :--- | :--- | :--- | :--- |

### INPUT TEXT
[PASTE DOCUMENT HERE]



Phase 2: Knowledge Ontology Refinement & Logic Clustering 

Objective: Develop a modular Python-based processing engine that transforms raw, anonymized triples (Subject-Predicate-Object) into refined, structurally distinct logic clusters. This phase bridges the gap between raw text extraction and structured schema generation using the following strategies:

1. Topological Analysis: Utilize networkx to apply Louvain and Leiden community detection, identifying distinct thematic neighborhoods based on graph connectivity.
2.  Semantic Compression: Implement optional embedding-based clustering to identify hypernyms and synonyms. This reduces graph order by collapsing redundant or similar-meaning nodes into representative concepts.
3. Anonymization Resolution: Resolve entity collisions (e.g., multiple "PERSON" tags in anonymized reports) by analyzing contextual neighborhoods and relationship similarity.
4. Modular Pipeline: Ensure a flexible architecture where embedding and graph operations can be toggled or reordered to minimize compute redundancy.
5. Schema Alignment: The final output must condense complex graph data into a simplified, representative structure optimized for (Phase 3) extraction into fixed Pydantic classes.

Knowledge Refinement Prompt:
### ROLE
You are a Senior Graph Data Scientist and Python Developer specializing in Knowledge Engineering.

### OBJECTIVE
Build a modular Python pipeline that transforms raw Subject-Predicate-Object triples into refined, clustered logic schemas. The goal is to reduce noise and identify hypernyms to prepare data for fixed Pydantic class extraction.

### INPUT DATA
- Format: List of dictionaries or a CSV/Markdown table containing `subject`, `predicate`, and `object` (snake_case) from Phase 1.

### TASK REQUIREMENTS
Develop a modular Python script with the following two-pronged strategy:

#### 1. Topological Clustering (Option A)
- Use `networkx` to build a directed graph from the triples.
- Implement community detection using the Louvain and Leiden algorithms.
- **Logic:** Group entities based on their structural connectivity (graph topology heuristics).

#### 2. Semantic Embedding & Hypernym Resolution (Option B)
- Integrate a modular embedding step (e.g., `sentence-transformers`).
- Convert Subject, Predicate, and Object into vector representations.
- Implement a clustering method (e.g., HDBSCAN or AgglomerativeClustering) to identify similar meaning words.
- **Objective:** Collapse redundant nodes (e.g., `user_account` and `account_user`) into a single hypernym to reduce graph order and size.

### ARCHITECTURAL CONSTRAINTS
- **Modular Design:** Create a `GraphProcessor` class where embedding can be toggled on/off. Ensure the embedding step can be executed at the start or deferred to later stages to minimize redundant compute.
- **Graph Operations:** Include methods for:
    - Node degree analysis.
    - Pruning of low-confidence or orphaned triples.
    - Exporting clusters as "Logic Schemas" for Pydantic mapping.
- **Pydantic Alignment:** The final output must be structured such that a subsequent LLM pass can easily map these clusters into fixed `Instructor` Pydantic classes.

### OUTPUT
Provide the complete Python implementation, including:
1. Requirements.txt (networkx, python-louvain, cdlib, sentence-transformers, scikit-learn).
2. The `GraphProcessor` class.
3. A main execution block demonstrating both Option A and Option B.
4. Saved output visualizations of the graph (.html format) from the initial population through each transformation stage.




