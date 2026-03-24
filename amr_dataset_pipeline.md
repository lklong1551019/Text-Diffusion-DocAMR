# DocAMR Dataset Processing Pipeline (Discrete Masked Diffusion)

This document provides a comprehensive, step-by-step description of the data processing pipeline implemented in `amr_dataset.py`.

This pipeline acts as a bridge between the raw Document-level Abstract Meaning Representation (DocAMR) `.out` files and a PyTorch Geometric (PyG) + Discrete Diffusion model infrastructure (similar to MaskGIT / LLada). It tackles translating literal graph notation into continuous tensor features and encoding text targets as discrete sequences.

---

## Phase 1: Initialization & Parsing (`__init__` and `_parse_docamr_file`)

When `DocAMRDataset` is instantiated, it loads the raw data into memory and prepares the global vocabularies via the following steps:

1. **Model Loading:** Initializes a Hugging Face Tokenizer and Text Encoder (defaulting to `roberta-base`). The text encoder is used exclusively to extract contextual word embeddings for the graph's AMR node concepts. The tokenizer extracts the discrete sequence integers.
2. **File Reading & Block Splitting:** It opens the provided `.out` file and splits the content into individual document chunks (separated by double blank lines `\n\n`).
3. **Data Extraction per Block:** For every document block:
   - **Text Extraction:** It locates the `# ::tok` line containing the raw sentences, strips formatting tokens like `<next_sent>`, and retrieves the pure raw text.
   - **Graph Construction:** It extracts all lines not starting with a `#` (the clean Penman notation) and uses the `penman` library to decode it into a parsed Graph object.
4. **Vocabulary Building:** As it parses the graph, it streams through all edges and dynamically populates `edge_type_to_id`, a dictionary mapping each string role (e.g., `:ARG0`, `:same-as`, `:location`) to a unique integer ID.

At the end of initialization, `self.documents` holds a list of dictionaries with `"text"` and `"graph"` for every document in the file.

---

## Phase 2: Single Document Processing (`__getitem__`)

When the DataLoader requests a specific document via `__getitem__`, the pipeline transforms the text and corresponding graph into a model-ready PyG `Data` object through three sub-steps:

### Step 2a: Target Text Sequences (The Objective `target_ids`)
The dataset generates the discrete integer sequence (`target_ids` and `target_mask`) of the entire document's text. This serves as the objective target that the cross-entropy loss and discrete diffusion model will learn to reconstruct from `[MASK]` tokens.
- The raw text is tokenized up to the `max_seq_len` (with standard huggingface padding and truncation).
- The resulting tensors are raw vocabulary index mappings (`input_ids` and `attention_mask`).
- Resulting shape for each: `[max_seq_len]`.

### Step 2b: AMR Node Processing (Graph Vertices)
Penman formats "being a concept" as an edge relation (the `:instance` role). To build nodes:
- The script iterates across `graph.instances()` specifically looking for concept definitions (e.g., `s1.p` is an instance of `person`).
- It extracts unique node IDs and assigns them an integer index `[0, 1, 2...]` to create `node_id_to_idx`.
- The text equivalent of these concepts (e.g. "person", "go-02") are gathered into a string array.
- These concept labels are passed through the mapping and the full `text_encoder` transformer. The continuous representation (acting as node features `x`) is taken from the first token's (`[:, 0, :]`) hidden state representing the semantic meaning of that concept.

### Step 2c: AMR Edge Processing (Graph Connectivity)
With the nodes indexed, the script iterates through standard relations/edges in the graph (`graph.edges()`), looking for connections between nodes:
- It filters out edges pointing to string/number literals (e.g., exact names or quantities) because PyG requires edges to strictly connect node integer IDs to node integer IDs.
- For valid connections, it converts the string source and target bounds into their integer indices based on the `node_id_to_idx` map.
- The edge relation string (e.g., `:ARG1`) is converted to its integer form using the `edge_type_to_id` vocabulary setup in Phase 1.
- Both metrics finalize to PyTorch tensors `edge_index` (shape `[2, num_edges]`) and `edge_type` (shape `[num_edges]`).

*(Note: The result is bundled into a PyG `Data(x, edge_index, edge_type, target_ids, target_mask)` object).*

---

## Phase 3: Custom Collation & Batching (`docamr_collate_fn`)

PyTorch DataLoader requires rules on how to bunch multiple documents together. Given that graph data and sequence continuous data are structurally very different, the dataset implements a custom collation function:

1. **Extract Dense Sequences:** It steps through each `Data` object in the batch, pulling out the discrete text variables (`target_ids` and `target_mask`) and stacking them along `dim=0` (standard batching). This yields uniform dense tensor grids of shape `[batch_size, seq_len]`.
2. **Prevent Illegal Batching:** It explicitly deletes the `target_ids` and `target_mask` attributes off the PyG `Data` object so PyTorch Geometric's backend won't mistakenly try to flatten and merge the token sequences into a massive 1D text block along with the graph nodes.
3. **Block Diagonal Graph Batching:** The residual `Data` objects containing only `x`, `edge_index`, and `edge_type` are pushed through PyG's `Batch.from_data_list()`. This stacks nodes feature matrices and dynamically modifies the `edge_index` offsets to ensure separate graph components don't inter-connect, converting the batch of small graphs into one massive disjointed graph.

**Final Output returned to the training loop:**
A dictionary heavily optimized for the discrete diffusion pipeline containing the discrete sequence text grids (`"target_ids"`, `"target_mask"`) and the structural conditioning block (`"graph_batch"`).
