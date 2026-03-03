"""
STAGE 3 — Scene Graph Construction
====================================
Builds a structured graph representation from normalized scene semantics.

Nodes: objects with attributes
Edges: spatial/semantic relationships

This is the reasoning backbone of the system.
"""

import logging
import json
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from pathlib import Path

from src.stages.stage1_text_understanding import SceneSemantics

logger = logging.getLogger(__name__)


@dataclass
class SceneGraph:
    """Structured scene graph representation."""

    scene_type: str = "unknown"
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    objects: List[str] = field(default_factory=list)
    relationships: List[Tuple[str, str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        nodes = []
        for node in self.graph.nodes(data=True):
            nodes.append({
                "id": node[0],
                "attributes": node[1].get("attributes", []),
            })
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "relation": data.get("relation", ""),
            })
        return {
            "scene_type": self.scene_type,
            "nodes": nodes,
            "edges": edges,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class SceneGraphBuilder:
    """
    Constructs a directed graph from SceneSemantics.

    Nodes represent objects (with attribute metadata).
    Edges represent spatial/semantic relationships.
    """

    def __init__(self):
        pass

    def build(self, semantics: SceneSemantics) -> SceneGraph:
        """
        Build a scene graph from structured semantics.

        Args:
            semantics: Output from Stage 1/2 (text understanding + normalization).

        Returns:
            SceneGraph with populated networkx DiGraph.
        """
        logger.info("=" * 50)
        logger.info("STAGE 3: Scene Graph Construction")
        logger.info("=" * 50)

        sg = SceneGraph()
        sg.scene_type = semantics.scene_type
        sg.objects = list(semantics.objects)
        sg.relationships = list(semantics.relationships)

        G = nx.DiGraph()

        # Add nodes (objects with attributes)
        for obj in semantics.objects:
            attrs = semantics.attributes.get(obj, [])
            G.add_node(obj, attributes=attrs, node_type="object")
            logger.info(f"  Node: {obj} | attrs={attrs}")

        # Add edges (relationships)
        for subj, pred, obj in semantics.relationships:
            # Ensure both endpoints exist as nodes
            if subj not in G:
                G.add_node(subj, attributes=[], node_type="object")
            if obj not in G:
                G.add_node(obj, attributes=[], node_type="object")

            G.add_edge(subj, obj, relation=pred)
            logger.info(f"  Edge: {subj} --[{pred}]--> {obj}")

        sg.graph = G

        logger.info(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return sg

    def visualize(self, scene_graph: SceneGraph, output_path: str) -> str:
        """
        Render the scene graph as a PNG image.

        Args:
            scene_graph: The SceneGraph to visualize.
            output_path: File path for the output image.

        Returns:
            Path to saved image.
        """
        G = scene_graph.graph
        if G.number_of_nodes() == 0:
            logger.warning("Empty graph, skipping visualization")
            return ""

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.set_title(f"Scene Graph — {scene_graph.scene_type}", fontsize=14, fontweight="bold")

        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

        # Draw nodes
        node_labels = {}
        for node, data in G.nodes(data=True):
            attrs = data.get("attributes", [])
            label = node
            if attrs:
                label += f"\n({', '.join(attrs)})"
            node_labels[node] = label

        nx.draw_networkx_nodes(G, pos, node_color="#4ECDC4", node_size=2000,
                               alpha=0.9, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9,
                                font_weight="bold", ax=ax)

        # Draw edges with labels
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            edge_labels[(u, v)] = data.get("relation", "")

        nx.draw_networkx_edges(G, pos, edge_color="#556270", arrows=True,
                               arrowsize=20, width=2, ax=ax,
                               connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=8, font_color="#C44D58", ax=ax)

        ax.axis("off")
        plt.tight_layout()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)

        logger.info(f"  Scene graph saved: {output_path}")
        return output_path
