"""
Graph Reasoning Module for PATH(n) Task (P2.2)

Implements Datalog-based reachability reasoning for graph pathfinding,
as required by Tsamoura et al. to demonstrate framework compositionality.

The PATH(n) task:
  - Input: n node images representing a sequence of graph nodes
  - Question: "Is there a path from node A to node B?"
  - The neural module recognizes node identities from images
  - The symbolic module reasons about graph reachability via Datalog rules

Symbolic rules (Datalog):
  reachable(X, Y) :- edge(X, Y).
  reachable(X, Z) :- edge(X, Y), reachable(Y, Z).
  path(S, T)      :- reachable(S, T).
"""

import numpy as np
import clingo
from typing import Dict, List, Tuple, Optional, Any
from collections import deque


# ---------------------------------------------------------------------------
# ASP Knowledge Base for Graph Reachability
# ---------------------------------------------------------------------------

_GRAPH_KB = """
% Reachability rules (Datalog)
reachable(X, Y) :- edge(X, Y).
reachable(X, Z) :- edge(X, Y), reachable(Y, Z).
"""

_PATH_QUERY = """
{graph_kb}
{edges}
path_exists :- reachable({source}, {target}).
#show path_exists/0.
"""

_ALL_PATHS_QUERY = """
{graph_kb}
{edges}
reachable_pair(X, Y) :- reachable(X, Y).
#show reachable_pair/2.
"""

_ABDUCTION_QUERY = """
{graph_kb}
{edges}
% Find which edges would need to exist to make a path
needed_edge(X, Y) :- node(X), node(Y), not edge(X, Y), not reachable({source}, {target}).
#show needed_edge/2.
"""


class GraphKnowledgeBase:
    """
    Datalog-based graph reasoning engine using clingo.

    Encodes graph structure as edge/2 facts and reasons about
    reachability via transitive closure rules.
    """

    def __init__(self, num_nodes: int = 10):
        self.num_nodes = num_nodes
        self.edges = set()

        # Smoke-test clingo
        ctl = clingo.Control()
        ctl.add("base", [], "node(0).")
        ctl.ground([("base", [])])

    def set_graph(self, edges: List[Tuple[int, int]]):
        """Set the graph edges."""
        self.edges = set(edges)

    def add_edge(self, u: int, v: int):
        self.edges.add((u, v))

    def _edges_asp(self) -> str:
        """Convert edges to ASP facts."""
        lines = [f"node(0..{self.num_nodes - 1})."]
        for u, v in self.edges:
            lines.append(f"edge({u}, {v}).")
        return "\n".join(lines)

    def check_path(self, source: int, target: int) -> bool:
        """Check if there is a path from source to target using Datalog rules."""
        program = _PATH_QUERY.format(
            graph_kb=_GRAPH_KB,
            edges=self._edges_asp(),
            source=source,
            target=target,
        )

        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], program)
        ctl.ground([("base", [])])

        found = False
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                for atom in model.symbols(shown=True):
                    if atom.name == "path_exists":
                        found = True
        return found

    def get_all_reachable(self) -> List[Tuple[int, int]]:
        """Get all reachable pairs in the graph."""
        program = _ALL_PATHS_QUERY.format(
            graph_kb=_GRAPH_KB,
            edges=self._edges_asp(),
        )

        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], program)
        ctl.ground([("base", [])])

        pairs = []
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                for atom in model.symbols(shown=True):
                    if atom.name == "reachable_pair":
                        args = atom.arguments
                        pairs.append((int(str(args[0])), int(str(args[1]))))
        return pairs

    def find_shortest_path(self, source: int, target: int) -> Optional[List[int]]:
        """BFS shortest path (Python fallback for path extraction)."""
        adj = {}
        for u, v in self.edges:
            adj.setdefault(u, []).append(v)

        visited = {source}
        queue = deque([(source, [source])])

        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def abduce_missing_edges(self, source: int, target: int) -> List[Tuple[int, int]]:
        """
        Abduction: find which single edge additions would create a path
        from source to target.
        """
        if self.check_path(source, target):
            return []  # Path already exists

        # Try adding each possible edge and check if it creates a path
        missing_edges = []
        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                if u != v and (u, v) not in self.edges:
                    self.edges.add((u, v))
                    if self.check_path(source, target):
                        missing_edges.append((u, v))
                    self.edges.remove((u, v))

        return missing_edges


# ---------------------------------------------------------------------------
# PATH(n) Symbolic Module
# ---------------------------------------------------------------------------

class PathSymbolicModule:
    """
    Symbolic module for the PATH(n) task.

    Implements the same SymbolicModule interface as the arithmetic module,
    demonstrating framework compositionality.

    Deduction: given predicted node sequence, check if path exists
    Abduction: given desired reachability, find node corrections
    """

    def __init__(self, num_nodes: int = 10):
        self.kb = GraphKnowledgeBase(num_nodes)
        self.num_nodes = num_nodes

    def set_graph(self, edges: List[Tuple[int, int]]):
        """Configure the graph for this task instance."""
        self.kb.set_graph(edges)

    def symbolic_deduction(self, input_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward reasoning: check if the predicted node sequence forms a valid path.

        Args:
            input_state: {
                'symbols': [node_id_0, node_id_1, ...],  # predicted node sequence
                'source': int,  # query source (optional, defaults to first node)
                'target': int,  # query target (optional, defaults to last node)
            }

        Returns:
            Same format as arithmetic deduction.
        """
        out = {
            'valid': False,
            'result': None,
            'derivation': [],
            'contradictions': [],
            'intermediate_states': [],
        }

        symbols = input_state.get('symbols', [])
        if not symbols:
            out['contradictions'].append('empty_node_sequence')
            return out

        # Parse node IDs
        try:
            nodes = [int(s) for s in symbols]
        except (ValueError, TypeError):
            out['contradictions'].append('non_integer_node')
            out['derivation'].append(f"Node IDs must be integers: {symbols}")
            return out

        # Check range
        for n in nodes:
            if n < 0 or n >= self.num_nodes:
                out['contradictions'].append('node_out_of_range')
                out['derivation'].append(f"Node {n} out of range [0, {self.num_nodes})")
                return out

        source = input_state.get('source', nodes[0])
        target = input_state.get('target', nodes[-1])

        # Check reachability via Datalog
        path_exists = self.kb.check_path(source, target)

        out['valid'] = True
        out['result'] = 1.0 if path_exists else 0.0
        out['derivation'].append(
            f"Datalog ⊢ reachable({source}, {target}) = {path_exists}"
        )

        # Also check if each consecutive pair in the sequence is connected
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            connected = (u, v) in self.kb.edges
            out['intermediate_states'].append({
                'step': i,
                'edge': (u, v),
                'exists': connected,
            })
            if connected:
                out['derivation'].append(f"  edge({u}, {v}) ✓")
            else:
                out['derivation'].append(f"  edge({u}, {v}) ✗")

        # Get actual shortest path if it exists
        shortest = self.kb.find_shortest_path(source, target)
        if shortest:
            out['derivation'].append(f"  Shortest path: {shortest}")

        return out

    def symbolic_abduction(
        self,
        desired_output: float,
        current_state: Dict[str, Any],
        neural_probs: Dict[str, np.ndarray],
    ) -> List[Any]:
        """
        Backward reasoning: find node corrections that achieve desired reachability.

        Args:
            desired_output: 1.0 for "path should exist", 0.0 for "no path"
            current_state: current predicted node sequence
            neural_probs: per-position node probabilities

        Returns:
            List of correction dicts or symbol lists.
        """
        symbols = current_state.get('symbols', [])
        if not symbols:
            return []

        try:
            nodes = [int(s) for s in symbols]
        except (ValueError, TypeError):
            return []

        source = current_state.get('source', nodes[0])
        target = current_state.get('target', nodes[-1])

        if desired_output > 0.5:
            # Want a path to exist: find alternative node sequences
            return self._abduce_path_exists(source, target, len(nodes), neural_probs)
        else:
            # Want no path: find node corrections that break connectivity
            return self._abduce_no_path(source, target, len(nodes), neural_probs)

    def _abduce_path_exists(self, source, target, seq_len, neural_probs):
        """Find node sequences that form a valid path."""
        path = self.kb.find_shortest_path(source, target)
        if path is None:
            return []

        results = []

        # Pad or trim path to match expected sequence length
        if len(path) <= seq_len:
            padded = path + [path[-1]] * (seq_len - len(path))
            correction = [str(n) for n in padded[:seq_len]]
            results.append(correction)

        # Also try alternative paths via different intermediate nodes
        for mid in range(self.num_nodes):
            if mid == source or mid == target:
                continue
            path1 = self.kb.find_shortest_path(source, mid)
            path2 = self.kb.find_shortest_path(mid, target)
            if path1 and path2:
                full_path = path1 + path2[1:]
                if len(full_path) <= seq_len:
                    padded = full_path + [full_path[-1]] * (seq_len - len(full_path))
                    correction = [str(n) for n in padded[:seq_len]]
                    if correction not in results:
                        results.append(correction)

            if len(results) >= 20:
                break

        return results

    def _abduce_no_path(self, source, target, seq_len, neural_probs):
        """Find node sequences where no path exists."""
        results = []

        for _ in range(50):
            nodes = [np.random.randint(0, self.num_nodes) for _ in range(seq_len)]
            nodes[0] = source
            s, t = nodes[0], nodes[-1]
            if not self.kb.check_path(s, t):
                results.append([str(n) for n in nodes])
                if len(results) >= 10:
                    break

        return results

    def add_constraint(self, constraint_name: str, constraint_fn):
        pass  # Graph constraints are encoded in the KB

    def get_rules(self) -> List[str]:
        return [
            'reachable(X,Y) :- edge(X,Y)',
            'reachable(X,Z) :- edge(X,Y), reachable(Y,Z)',
            'path(S,T) :- reachable(S,T)',
        ]


# ---------------------------------------------------------------------------
# PATH(n) Dataset Generator
# ---------------------------------------------------------------------------

class PathDataset:
    """
    Dataset for the PATH(n) task.

    Generates random graphs and path queries with node images.
    Each sample is a sequence of node images + a binary label
    (path exists / no path).
    """

    def __init__(self, num_samples: int = 1000, num_nodes: int = 10,
                 path_length: int = 3, edge_density: float = 0.3,
                 split: str = 'train'):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.path_length = path_length
        self.edge_density = edge_density

        # Generate random graph
        self.edges = self._generate_graph()
        self.kb = GraphKnowledgeBase(num_nodes)
        self.kb.set_graph(self.edges)

        # Generate samples
        self.data = []
        self._generate_dataset()

    def _generate_graph(self) -> List[Tuple[int, int]]:
        """Generate a random directed graph."""
        edges = []
        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                if u != v and np.random.random() < self.edge_density:
                    edges.append((u, v))
        return edges

    def _generate_node_image(self, node_id: int) -> np.ndarray:
        """
        Generate a synthetic 28×28 image for a node ID.

        Uses distinct visual patterns for each node (0-9).
        """
        img = np.zeros((28, 28), dtype=np.float32)

        # Create a unique pattern based on node_id
        np.random.seed(node_id * 1000 + 42)

        # Draw a number-like pattern
        cx, cy = 14, 14
        for _ in range(50 + node_id * 10):
            x = int(np.clip(cx + np.random.randn() * (3 + node_id * 0.5), 2, 25))
            y = int(np.clip(cy + np.random.randn() * (3 + node_id * 0.5), 2, 25))
            img[y, x] = min(1.0, img[y, x] + 0.3)

        # Add a distinct ring/shape for the node
        angle_offset = node_id * 36  # degrees
        for angle in range(0, 360, 10):
            rad = np.radians(angle + angle_offset)
            r = 8 + 2 * np.sin(np.radians(angle * node_id))
            x = int(np.clip(cx + r * np.cos(rad), 0, 27))
            y = int(np.clip(cy + r * np.sin(rad), 0, 27))
            img[y, x] = 1.0

        np.random.seed(None)  # Reset seed
        return img

    def _generate_dataset(self):
        """Generate path query samples."""
        reachable = set(self.kb.get_all_reachable())

        for _ in range(self.num_samples):
            # Pick random source and target
            source = np.random.randint(0, self.num_nodes)
            target = np.random.randint(0, self.num_nodes)
            while target == source:
                target = np.random.randint(0, self.num_nodes)

            has_path = (source, target) in reachable
            label = 1.0 if has_path else 0.0

            # Generate intermediate nodes
            if has_path:
                path = self.kb.find_shortest_path(source, target)
                if path and len(path) >= self.path_length:
                    nodes = path[:self.path_length]
                else:
                    nodes = [source] + [np.random.randint(0, self.num_nodes)
                                        for _ in range(self.path_length - 2)] + [target]
            else:
                nodes = [source] + [np.random.randint(0, self.num_nodes)
                                    for _ in range(self.path_length - 2)] + [target]

            # Generate images
            images = np.stack([self._generate_node_image(n) for n in nodes])

            self.data.append({
                'images': images,
                'result': label,
                'text': [str(n) for n in nodes],
                'source': source,
                'target': target,
                'graph_edges': self.edges,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
