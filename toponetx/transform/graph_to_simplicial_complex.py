"""Methods to lift a graph to a simplicial complex."""
from itertools import takewhile
from warnings import warn

import networkx as nx

from toponetx.classes.simplicial_complex import SimplicialComplex

__all__ = [
    "graph_to_clique_complex",
    "graph_to_neighbor_complex",
]


def graph_to_neighbor_complex(G: nx.Graph) -> SimplicialComplex:
    """Get the neighbor complex of a graph.

    Parameters
    ----------
    G : networkx graph
        Input graph.

    Returns
    -------
    SimplicialComplex
        The neighbor complex of the graph.

    Notes
    -----
    This type of simplicial complexes can have very large dimension ( dimension = max_i(len (G.neighbors(i))) )
    and it is a function of the distribution of the valency of the graph.
    """
    simplices = []
    for node in G:
        # each simplex is the node and its n-hop neighbors
        simplices.append(list(G.neighbors(node)) + [node])
    return SimplicialComplex(simplices)


def graph_to_clique_complex(
    G: nx.Graph, max_dim: int | None = None
) -> SimplicialComplex:
    """Get the clique complex of a graph.

    Parameters
    ----------
    G : networkx graph
        Input graph.
    max_dim : int, optional
        The max dimension of the cliques in
        the output clique complex.
        The default is None indicate max dimension.

    Returns
    -------
    SimplicialComplex
        The clique simplicial complex of dimension dim of the graph G.
    """
    cliques = nx.enumerate_all_cliques(G)

    # `nx.enumerate_all_cliques` returns cliques in ascending order of size. Abort calling the generator once we reach
    # cliques larger than the requested max dimension.
    if max_dim is not None:
        cliques = takewhile(lambda clique: len(clique) <= max_dim, cliques)

    SC = SimplicialComplex(cliques)

    # copy attributes of the input graph
    for node in G.nodes:
        SC[[node]].update(G.nodes[node])
    for edge in G.edges:
        SC[edge].update(G.edges[edge])
    SC.complex.update(G.graph)

    return SC

def graph_to_independent_set_complex(
    G: nx.Graph, max_dim: int | None = None
) -> SimplicialComplex:
    """Get the independent set complex of a graph.
    This simplicial complex is equivalent to the clique complex
    of the complement of the original graph. 
    Note that this complex will ignore any edge information in G. 

    Parameters
    ----------
    G : networkx graph
        Input graph.
    max_dim : int, optional
        The max dimension of the cliques in
        the output clique complex of the complement.
        The default is None indicates max dimension.

    Returns
    -------
    SimplicialComplex
        The independent set simplicial complex of the graph G.
    """
    return graph_to_clique_complex(nx.complement(G), max_dim=max_dim)

def graph_to_power_complex(
    G: nx.Graph, pow: int, max_dim: int | None = None
) -> SimplicialComplex:
    """Get the power complex of a graph by taking pow graph powers
    then computing the clique complex of the resulting power graph.
    Note that this complex will ignore any edge information in G. 

    Parameters
    ----------
    G : networkx graph
        Input graph.
    pow: int
        The graph power to compute. Must be greater than 0.
    max_dim : int, optional
        The max dimension of the cliques in
        the output clique complex of the complement.
        The default is None indicates max dimension.

    Returns
    -------
    SimplicialComplex
        The graph power complex of the graph G.
    """
    if pow < 1:
        raise ValueError(f'Graph power {pow} must be greater than 0.')
    return graph_to_clique_complex(nx.power(G, pow), max_dim=max_dim)

def graph_2_neighbor_complex(G) -> SimplicialComplex:
    warn(
        "`graph_2_neighbor_complex` is deprecated and will be removed in a future version, use `graph_to_neighbor_complex` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return graph_to_neighbor_complex(G)


def graph_2_clique_complex(
    G: nx.Graph, max_dim: int | None = None
) -> SimplicialComplex:
    warn(
        "`graph_2_clique_complex` is deprecated and will be removed in a future version, use `graph_to_clique_complex` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return graph_to_clique_complex(G, max_dim)
