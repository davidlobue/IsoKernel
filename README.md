# IsoKernel: Semantic Knowledge Engineering Architecture

IsoKernel is a high-performance, completely offline-capable pipeline that transforms raw unstructured text into fully structured, strictly defined Knowledge Graphs. 

It accomplishes this by dynamically bridging unconstrained LLM Natural Language Processing (Phase 1) with localized high-dimensional physics embeddings (Phase 2), strictly verified semantic logic filtering (Phase 3), and complex graph topology interactions (Phase 4).

## ⚠️ Crucial Terminology: Disentangling "Clusters"
Throughout this pipeline, you will see data undergo two entirely independent grouping processes. It is absolutely critical to understand the mathematical difference between them:

1. **Geometric Clusters (NLP Vectors):** This occurs during the Semantic Compression phase. It evaluates words as literal coordinates in math space strictly based on their dictionary definition (e.g., placing "Python" near "Java"). It has absolutely zero concept of edges, relationships, or what the words are connected to in your graph.
2. **Topological Communities (Graph Clusters):** This occurs at the very end of the pipeline during `community_detection` (Louvain). This evaluates pure network density and physics (nodes linked together by active relationship edges). It explicitly maps "Communities" based purely on how entities physically interact with each other in the graph, completely ignoring what their dictionary definition means!

---

## The Core Pipeline Execution

### Phase 1: Unconstrained LLM Triplet Extraction (Discovery Phase)
The Orchestrator reads raw text and injects it into a generalized LLM (Local Ollama, Vertex, or OpenAI). The Instructional prompt forces the LLM to extract every single explicitly stated relationship identically as a Triple (Subject -> Predicate -> Object). 
It generates wildly messy permutations structurally limited only by the text (e.g. `Google`, `Google Inc.`, and `The Google Corporation`).

### Phase 2: NLP Semantic Compression (Geometric Clustering)
The raw triplets are intercepted and passed into the `SentenceTransformer` HuggingFace engine. 

#### The Mathematics:
- **Vectorization**: Every single extracted string is mapped onto a 384-dimensional mathematical matrix coordinate.
- **Spectral Dimensionality Destruction (`use_spectral_decomposition`)**: If enabled, the pipeline natively calculates the **Principal Eigenvectors** (PCA) of the embeddings, violently shattering the Linguistic Fluff tensors down to a strict, core-essence geometry (e.g. 12 dense dimensions). This dramatically isolates fundamental mathematical "truth" from AI noise parameters.
- **Deduplication Geometry (`compression_mode`)**: 
  - If set to `"isolated"`, Nouns (Subjects/Objects) and Verbs (Predicates) are physically firewalled. Verbs only mathematically group with Verbs.
  - If `"unified"`, they are dumped into one massive dimensional space globally.
- **`AgglomerativeClustering`**: `scikit-learn` natively traverses the tensors and calculates the **Cosine Distance** between every single coordinate. If the vectors structurally sit closer together than your `similarity_threshold` (e.g. `0.8`), they are merged mathematically into a **Geometric Cluster**.

### Phase 3: Taxonomic Hypernym Resolution
Once the mathematical boundaries of a Geometric Cluster are established (e.g., grouping `['Python', 'Java', 'Rust']`), the system must decide on a single uniform label (the Hypernym) to structurally represent the entire group inside your final graph.

**Resolution Algorithms (`hypernym_resolution`):**
1. **`semantic_centroid` (Default)**: 
   - **Math**: Traverses the actual localized tensors belonging strictly to the cluster members. It leverages `numpy.mean(..., axis=0)` to physically calculate an invisible mean vector (the exact dead-center "Centroid" of the scatter plot).
   - **Proximity**: Maps the Cosine Similarity of every literal string inside the cluster against the invisible Centroid to declare the closest actual string as the anchor term.
   - **Taxonomic Lifting**: Boots up a secondary LLM pipeline explicitly requesting a categorical classification of that Centroid term. It strictly forces a deductive `Is-A` boolean pass sequentially across all array members. If verified, the cluster adopts the Formal concept natively. If verification logically fails, it falls safely back to the unmodified geometry Centroid.
2. **`most_frequent`**: Defaults to the literal text string that statically occurred the highest number of times in the source extraction.
3. **`shortest_string`**: Exclusively selects the literal string computationally with the shortest length.

*At the conclusion of Phase 3, explicit "Before & After" CSV logic logs (`nlp_triplet_transformations.csv`) are automatically printed detailing pristine trace mappings covering every NLP adjustment made.*

### Phase 4: Graph Topology Analysis
The completely filtered and standardized logic arrays natively deploy into `networkx`. Because all strings are cleanly unified, overlapping subjects logically fuse together into massive structural Super Nodes!

- **Community Physics**: The algorithm loops mathematics sequentially across the network to aggressively evaluate the literal density of physical relationship edges between nodes compared against the wider graph.
  - `"louvain"` or `"leiden"`: Maximizes loop modularity (hunting for circular relationship edge-chains). 
  - **`"spectral"`**: Explicitly extracts the **Normalized Laplacian Matrix** natively from your network. It tracks the raw Eigenvalues generated and utilizes a programmatic **Eigengap Heuristic Array** to perfectly deduce the mathematical optimal number of total distinct communities hidden in the nodes! It then relies on `scikit-learn` to forcefully shatter the graph along those boundaries.
- Dense logic interactions are isolated dynamically into explicitly numbered **Topological Communities**. 

The final shape natively prints visually to an interactive HTML canvas, color-coded directly to its Topology physics!
