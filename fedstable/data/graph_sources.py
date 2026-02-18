"""Build DeFi graph from raw sources (DeFi Llama or synthetic). Placeholder for Q2."""

from pathlib import Path
from typing import Any, Dict

# PyTorch Geometric optional
# from torch_geometric.data import Data


def build_graph_from_defillama(data_dir: Path, config: Dict[str, Any]) -> Any:
    """Build global graph from DeFi Llama data. Returns PyG Data or dict. Stub."""
    raise NotImplementedError("Q2: implement DeFi Llama graph build")


def build_synthetic_graph(num_nodes: int, num_edges: int, seed: int = 42) -> Any:
    """Build small synthetic graph for tests. Stub."""
    raise NotImplementedError("Q2: implement synthetic graph")
