"""
This module defines the PyTorch Dataset for loading and processing Document-level 
Abstract Meaning Representation (DocAMR) data. It is responsible for parsing Penman 
notation graphs, extracting continuous text embeddings using a transformer model, 
and converting the AMR graphs into PyTorch Geometric (PyG) Data objects.
"""

import torch
import penman
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer, AutoModel

class DocAMRDataset(Dataset):
    """
    A PyTorch Dataset for IBM/docAMR output files.
    Reads .out files containing raw text and Penman notation graphs.
    
    This dataset acts as the bridge between raw DocAMR data and the diffusion model, 
    preparing both the continuous space text target and the conditioning AMR graphs.
    """
    def __init__(self, file_path, base_model_name="roberta-base", max_seq_len=128):
        """
        Initializes the DocAMRDataset step-by-step:
        1. Loads the specified Hugging Face tokenizer and text encoder model.
        2. Initializes an empty vocabulary for edge types.
        3. Calls `_parse_docamr_file` to parse the provided .out file and build the dataset in memory.
        
        Args:
            file_path (str): Path to the .out DocAMR file.
            base_model_name (str): Name of the pretrained text encoder (default: 'roberta-base').
            max_seq_len (int): Maximum sequence length for the continuous text target.
        """
        self.file_path = file_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.text_encoder = AutoModel.from_pretrained(base_model_name)
        self.max_seq_len = max_seq_len
        
        # Build Vocabularies for Edge Types incrementally
        self.edge_type_to_id = {}
        
        # Parse the .out file
        self.documents = self._parse_docamr_file(file_path)

    def _parse_docamr_file(self, file_path):
        """
        Parses a DocAMR .out file to extract raw text and construct Penman graph objects.
        
        Step-by-step parsing process:
        1. Reads the full text of the .out file.
        2. Splits the file into individual graph blocks separated by double newlines.
        3. For each block, identifies the `# ::tok` line to extract the raw text snippet.
        4. Extracts all non-comment lines to form the raw Penman string and decodes it.
        5. Dynamically registers newly encountered edge types (roles) into the edge vocabulary.
        
        Args:
            file_path (str): Path to the .out file to parse.
            
        Returns:
            list[dict]: A list of dictionaries containing "text" and "graph" for each document.
        """
        documents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # DocAMR blocks are separated by blank lines
        blocks = content.strip().split('\n\n')
        for block in blocks:
            lines = block.strip().split('\n')
            tok_line = next((l for l in lines if l.startswith('# ::tok ')), None)
            
            if tok_line:
                # tok_line sample: 
                # ::tok Hailey is going to London tomorrow . <next_sent> She is planning to go to Italy after London . 
                # <next_sent> She is going to see the Big Ben . <next_sent> Her friend Phil is meeting her in London .
                raw_text = tok_line.replace('# ::tok ', '').replace('<next_sent>', '').strip()            
                
                # Extract the pure PENMAN graph (all lines not starting with #)
                graph_string = "\n".join(l for l in lines if not l.startswith('#'))
                
                try:
                    # Look at explain_penman_graph.md for more detail of this graph
                    graph = penman.decode(graph_string)                

                    documents.append({
                        "text": raw_text,
                        "graph": graph
                    })
                    
                    # Register new edge types into vocabulary
                    for source, role, target in graph.edges():
                        # Role is like: :snt1, :same-as, :ARG1, :location
                        if role not in self.edge_type_to_id:                            
                            self.edge_type_to_id[role] = len(self.edge_type_to_id)
                except Exception as e:
                    print(f"Failed to parse graph block: {e}")
                    
        return documents

    @property
    def num_relations(self):
        """Used to instantiate the RGCN num_relations argument"""
        return len(self.edge_type_to_id)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single document into model-ready tensors at the specified index.
        
        Step-by-step processing:
        1. Text to Continuous Embeddings: Tokenizes the raw text and extracts contextual 
           word embeddings to serve as the continuous target (x_0) for the diffusion model.
        2. Node Processing: Iterates through the graph instances to extract node concepts, 
           handling cases of empty graphs, and retrieves node string features using the text encoder.
        3. Edge Processing: Maps literal AMR relations into PyTorch Geometric edge indices 
           (source, target) and edge types using the dynamically built vocabulary.
        4. Graph Construction: Bundles the node embeddings, edge indices, edge types, and
           the clean continuous text embeddings into a PyG Data object.
           
        Args:
            idx (int): The index of the document in the dataset.
            
        Returns:
            torch_geometric.data.Data: PyG Data object containing graph structures and text embeddings.
        """
        doc = self.documents[idx]
        # sample text: Hailey is going to London tomorrow . She is planning to go to Italy after London . She is going to see the Big Ben . Her friend Phil is meeting her in London .
        # Graph: penman parsed graph, look at explain_penman_graph.md for more detail of this graph
        text = doc["text"]
        graph = doc["graph"]

        # 1. Text to Continuous Embeddings (The Noise Target x_0)
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_seq_len
        )
        # {'input_ids': tensor([[    0, 24017,  9422,    16,   164,     7,   928,  3859,   479,  1437,
        #    264,    16,  1884,     7,   213,     7,  2627,    71,   928,   479,
        #   1437,   264,    16,   164,     7,   192,     5,  1776,  1664,   479,
        #   1437,  1405,  1441,  4720,    16,   529,    69,    11,   928,   479,
        #      2,     1,     1,     1,     1,     1,     1,     1,     1,     1,
        #      1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
        #      1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
        #      1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
        #      1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
        #      1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
        #      1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
        #      1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
        #      1,     1,     1,     1,     1,     1,     1,     1]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0]])}
        
        # For discrete masked diffusion, we extract raw token ids
        target_ids = inputs.input_ids.squeeze(0)
        target_mask = inputs.attention_mask.squeeze(0)

        # 2. Process Graph Nodes (AMR Concepts)
        node_ids = set()
        node_concepts = {}
        
        # Graph instances are things like (d / document) or (s1 / go-02)
        # graph.instances() will only returns the triples where the role is :instance
        for source, role, target in graph.instances():
            # Triple: (source, role, target)
            # 3 main types of triple:

            # A. Node definitions (the :instance roles): Instead of storing nodes in a separate list, penman treats "being a concept" as an edge called :instance
            # Example: (s1.p, :instance, person)
            # Meaning: There is a node whose ID is s1.p and its concept is person
            # In our code: this is used to build the "node_id_to_idx" mappings, eg, node "s1.p" gets the text "person"

            # B. Relations between nodes (edges): these triples connect 2 actual node IDs together with a specific relation (role).
            # Example 1: (s1.g, :ARG0, s1.p)
            # Meaning: The go-02 node (s1.g) is connected to the person node (s1.p) via the :ARG0 role / relation. (In english: The person is the one doing the going)
            # Example 2: (s3.s2, :same-as, s2.s)
            # Meaning: This is a DocAMR-specific coreference edge. It links "she" in sentence 3 (s3.s2) to "she" in sentence 2 (s2.s), telling the model they are the same entity.
            # In our code: this is used to build PyG "edge_index" and "edge_type" tensors.

            # C. Attributes / Literals: These are edges that point from a node to a raw string or number, rather than another node.
            # Example: (s1.n2, :op1, London)
            # Meaning: The "name" node (s1.n2) has an exact string value of "London"
            # In our code: our loop filters these out. Because "London" is not a node ID (it never had an :instance declaration), it is skipped.
        
        
            node_ids.add(source)
            # graph.instances() will only returns the triples where the role is :instance
            # Asigning a concept to a variable is only defined once, eg, (s1.p / person) defines the node s1.p as an instance of "person"
            # So, the concept person with s1.p is not repeated. And hence, this won't override any existing key-value
            node_concepts[source] = target            
        
        # node_id_to_idx sample:
        # {'s2.a': 0, 's1.n': 1, 's3.s2': 2, 's4.s': 3, 's2.p': 4, 's4.f': 5, 's4.h': 6, 's3.s': 7, 's1.n2': 8, 's4.n2': 9, 's2.c': 10, 
        # 's1.p': 11, 's2.g': 12, 's4.m': 13, 's4.p': 14, 's1.t': 15, 's2.c2': 16, 's1.c': 17, 's4.c': 18, 's1.g': 19, 's2.s': 20,
        #  's4.n': 21, 'd': 22, 's2.n': 23, 's3.b': 24, 's3.n': 25, 's2.n2': 26}
        node_id_to_idx = {nid: i for i, nid in enumerate(list(node_ids))}
        # node_texts sample:
        # ['after', 'name', 'she', 'she', 'plan-01', 'friend', 'have-rel-role-91', 'see-01', 'name', 'name', 'city', 'person', 'go-02',
        #  'meet-03', 'person', 'tomorrow', 'country', 'city', 'city', 'go-02', 'she', 'name', 'document', 'name', 'building', 'name', 'name']
        node_texts = [str(node_concepts.get(nid, "entity")) for nid in node_id_to_idx.keys()]    
        
        # Handle empty graph fallback
        if len(node_texts) == 0:
            node_texts = ["empty"]
            node_id_to_idx = {"empty": 0}
            
        # These concept labels are passed through the tokenizer
        node_inputs = self.tokenizer(
            node_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=50 # Node values are just words, so lengths are tiny
        )
        # Then passed through text encoder to get node embeddings
        with torch.no_grad():
            node_embeds = self.text_encoder(**node_inputs).last_hidden_state[:, 0, :]


            
        # 3. Process Graph Edges (Relational Mapping)
        edge_sources = []
        edge_targets = []
        edge_types = []
        
        for source, role, target in graph.edges():            
            # In AMR, "targets" can be string literals (e.g., "Hailey"). "Source" cannot be string literals
            # PyG edges strictly connect nodes to nodes. We skip literal edges for GNN connectivity mapping.
            if source in node_id_to_idx and target in node_id_to_idx:             
                # source: s1.g, role: :ARG0, target: s1.p
                # node_id_to_idx[source] = 15
                # node_id_to_idx[target] = 23
                # edge_type_to_id[role] = 1
                edge_sources.append(node_id_to_idx[source])
                edge_targets.append(node_id_to_idx[target])
                edge_types.append(self.edge_type_to_id[role])            
                
        # Format explicitly for PyG specification [2, num_edges]
        if len(edge_sources) > 0:
            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)
        # Edge_index: tensor([[21, 15, 23, 15,  3, 15, 21,  0,  0,  5,  5, 11,  5, 19, 12, 12, 21, 16,
        #  24, 16, 22, 21, 25,  2, 13, 13, 10, 13, 25, 25,  1,  1],
        # [15, 23,  4,  3, 14,  9,  0, 26,  5, 26, 11, 17, 19, 12, 18,  3, 16, 24,
        #  26, 22,  7, 25,  2,  8,  2, 10, 26,  6, 10,  1, 20,  3]])

        # Edge_type: tensor([ 0,  1,  2,  3,  2,  4,  5,  1,  6,  1,  3,  2,  4,  7,  2,  8,  9,  1,
        #  8,  6,  2, 10,  1,  2,  1,  6,  8, 11,  6, 12,  2,  8])


            
        # Create standard standalone PyG Data object
        data = Data(
            x=node_embeds,          
            edge_index=edge_index,  
            edge_type=edge_type,    
            target_ids=target_ids,
            target_mask=target_mask
        )
        return data

def docamr_collate_fn(data_list):
    """
    Custom collate function for batching DocAMR PyG Data objects for Discrete Text Diffusion.
    
    Why this is needed:
    - Discrete sequence tokens (`target_ids`, `target_mask`) require standard padding and dense batching
      giving a final shape of [batch_size, seq_len].
    - AMR Graph data (nodes and edges) require PyTorch Geometric's Block Diagonal Batching 
      which merges separate graph samples into a single massive disconnected graph batch.
    
    Step-by-step batching process:
    1. Extracts discrete `target_ids` and `target_mask` from every data object and stacks them compactly.
    2. Deletes these sequential arrays from the PyG data objects to prevent PyG's internal systems 
       from misinterpreting and flattening the text sequences along the node dimension.
    3. Calls PyG's `Batch.from_data_list()` to properly block-diagonalize the individual node graphs.
    
    Args:
        data_list (list[Data]): A list of PyG Data objects from `__getitem__`.
        
    Returns:
        dict: A dictionary containing standard 2D discrete text matrices ("target_ids", "target_mask") 
              and the block diagonal PyG batch ("graph_batch").
    """

    target_ids_list = [data.target_ids for data in data_list]
    target_mask_list = [data.target_mask for data in data_list]
    
    target_ids_batch = torch.stack(target_ids_list, dim=0)    
    target_mask_batch = torch.stack(target_mask_list, dim=0)
    
    # Remove them from PyG objects to prevent PyG from blindly concatenating the sequence dimensions
    for data in data_list:
        del data.target_ids
        del data.target_mask
        if hasattr(data, 'clean_embeds'):
            del data.clean_embeds
        
    graph_batch = Batch.from_data_list(data_list)
    
    return {
        "target_ids": target_ids_batch,
        "target_mask": target_mask_batch,
        "graph_batch": graph_batch
    }
