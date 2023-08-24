"""Test graph to simplicial complex transformation."""

import networkx as nx
import pytest

from toponetx.transform.graph_to_simplicial_complex import (
    graph_2_clique_complex,
    graph_2_neighbor_complex,
    graph_to_clique_complex,
    graph_to_neighbor_complex,
    graph_to_independent_set_complex,
    graph_to_power_complex
)


class TestGraphToSimplicialComplex:
    """Test graph to simplicial complex transformation."""

    def test_graph_to_power_complex(self):
        """Test graph_to_power_complex by evaluating 
        a triangle (does nothing) and a square graph"""
        # create triangle graph
        G = nx.Graph()
        G.add_edge(0,1)
        G.add_edge(1,2)
        G.add_edge(2,0)

        with pytest.raises(ValueError):
            # test power > 0 assertion
            graph_to_power_complex(G, 0)

        sc1 = graph_to_power_complex(G, 1)
        assert sc1.dim == 2

        sc2 = graph_to_power_complex(G, 2)
        assert sc2.dim == 2

        sc3 = graph_to_power_complex(G, 2)
        assert sc3.dim == 2

        assert (sc1.adjacency_matrix(1).todense() == \
                 sc2.adjacency_matrix(1).todense()).all()
        
        assert (sc1.adjacency_matrix(1).todense() == \
                 sc3.adjacency_matrix(1).todense()).all()
        
        # new graph, square
        G = nx.Graph()
        G.add_edge(0,1)
        G.add_edge(1,2)
        G.add_edge(2,3)
        G.add_edge(3,0)

        sc1 = graph_to_power_complex(G, 1)
        assert sc1.dim == 1

        sc2 = graph_to_power_complex(G, 2)
        assert sc2.dim == 3

        # should result in the same as pow=2
        sc3 = graph_to_power_complex(G, 3)
        assert sc3.dim == 3

        assert (sc2.adjacency_matrix(1).todense() == \
                sc3.adjacency_matrix(1).todense()).all()
            
    def test_graph_to_independent_set_complex(self):
        """Test graph_to_independent_set_complex by evaluating 
        1,2, and 3-edge parallel graphs and a 4-node cycle."""
        G = nx.Graph()
        G.add_edge(0,1)

        sc1 = graph_to_independent_set_complex(G)
        assert sc1.dim == 0

        G.add_edge(2,3)
        sc2 = graph_to_independent_set_complex(G)
        assert sc2.dim == 1

        G.add_edge(4,5)
        sc3 = graph_to_independent_set_complex(G)
        assert sc3.dim == 2

        # new graph
        G = nx.Graph()
        G.add_edge(0,1)
        G.add_edge(1,2)
        G.add_edge(2,3)
        G.add_edge(3,0)

        sc = graph_to_independent_set_complex(G)
        assert sc.dim == 1
        assert (0, 2) in sc
        assert (1, 3) in sc

    def test_graph_to_neighbor_complex(self):
        """Test graph_to_neighbor_complex."""
        G = nx.Graph()

        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 0)

        sc = graph_to_neighbor_complex(G)

        assert sc.dim == 2
        assert (0, 1) in sc
        assert (0, 2) in sc

    def test_graph_to_clique_complex(self):
        """Test graph_to_clique_complex."""
        G = nx.Graph()
        G.graph["label"] = 12

        G.add_node(0, label=5)
        G.add_edge(0, 1, weight=10)
        G.add_edge(1, 2)
        G.add_edge(2, 0)
        G.add_edge(2, 3)
        G.add_edge(3, 0)

        sc = graph_to_clique_complex(G)

        assert sc.dim == 2
        assert (0, 2, 3) in sc
        assert (0, 1, 2) in sc

        # attributes are copied
        assert sc.complex["label"] == 12
        assert sc[(0,)]["label"] == 5
        assert sc[(0, 1)]["weight"] == 10

        sc = graph_to_clique_complex(G, max_dim=2)

        assert sc.dim == 1
        assert (0, 2, 3) not in sc
        assert (0, 1, 2) not in sc

    def test_graph_2_neighbor_complex(self):
        """Test graph_2_neighbor_complex."""
        G = nx.Graph()

        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 0)

        with pytest.deprecated_call():
            sc = graph_2_neighbor_complex(G)

        assert sc.dim == 2
        assert (0, 1) in sc
        assert (0, 2) in sc

    def test_graph_2_clique_complex(self):
        """Test graph_2_clique_complex."""
        G = nx.Graph()

        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 0)
        G.add_edge(2, 3)
        G.add_edge(3, 0)

        with pytest.deprecated_call():
            sc = graph_2_clique_complex(G)

        assert sc.dim == 2
        assert (0, 2, 3) in sc
        assert (0, 1, 2) in sc

        with pytest.deprecated_call():
            sc = graph_2_clique_complex(G, max_dim=2)

        assert sc.dim == 1
        assert (0, 2, 3) not in sc
        assert (0, 1, 2) not in sc
