import json
from typing import List, Tuple, Any, Dict, Callable
import os

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from fitness.game_theory_game import HawkAndDove


def plot_hawk_and_dove(
    sentences: List[Tuple[float, float]],
    histories: List[Tuple[str, str]],
    out_path: str,
    name: str = "hd_test.pdf",
) -> None:
    """
    Plot the choices and payoffs for each iteration of iterated prisoners dilemma.

    The first iteration is at the bottom, the choice, payoff and total are connected with edges,
    and color indicates value:

    - Circle is choice

    - Triangle is payoff for a choice

    - Square is the total payoff
    """
    assert os.path.exists(out_path)

    graph = nx.Graph()

    # Add edges and nodes to the graph
    history_node_name: Callable[[int, int], str] = "h_{}_{}".format
    sentence_node_name: Callable[[int, int], str] = "s_{}_{}".format
    total_node_name: Callable[[int], str] = "t_{}".format
    total: float = 0
    history_colors: List[str] = []
    SENTENCE_COLOR_MAP = plt.cm.RdYlGn(np.linspace(0, 1, len(HawkAndDove.PAYOFF) + 1))
    sentence_colors: List[str] = []
    total_colors: List[str] = []
    total_max: float = sum([sum(_) for _ in sentences])
    total_color_map = plt.cm.RdYlGn(np.linspace(0, 1, int(total_max) + 1))
    assert len(histories) == len(sentences)
    for i in range(len(histories)):
        history = histories[i]
        sentence = sentences[i]
        total = total + sum(sentence)
        graph.add_node(total_node_name(i), action="{}: {}".format(i, total))
        total_colors.append(total_color_map[int(total)])
        assert len(history) == len(sentence)
        for j in range(len(history)):
            graph.add_node(history_node_name(i, j), action=history[j])
            history_colors.append(get_history_color(history[j]))
            graph.add_node(sentence_node_name(i, j), action=sentence[j])
            sentence_colors.append(SENTENCE_COLOR_MAP[int(sentence[j])])
            graph.add_edge(history_node_name(i, j), sentence_node_name(i, j))
            graph.add_edge(sentence_node_name(i, j), total_node_name(i))

    history_nodes: List[Any] = [_ for _ in graph.nodes if _.startswith("h")]
    sentence_nodes: List[Any] = sorted([_ for _ in graph.nodes if _.startswith("s")])
    total_nodes: List[Any] = sorted([_ for _ in graph.nodes if _.startswith("t")])

    # Set positions of the graph
    positions: Dict[str, Tuple[float, float]] = {}
    get_position(x=0.15, data=history_nodes, positions=positions)
    get_position(x=0.35, data=sentence_nodes, positions=positions)
    get_position(x=0.75, data=total_nodes, positions=positions)

    # Draw nodes
    _, ax = plt.subplots()
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=history_nodes,
        node_color=history_colors,
        alpha=0.5,
        node_size=100,
        node_shape="o",
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=sentence_nodes,
        node_color=sentence_colors,
        alpha=0.5,
        node_size=400,
        node_shape="v",
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=total_nodes,
        node_color=total_colors,
        alpha=0.5,
        node_size=500,
        node_shape="s",
    )

    nx.draw_networkx_edges(graph, positions, alpha=0.5)

    # Label graph
    label_data = dict(graph.nodes.data())
    for key, value in label_data.items():
        label_data[key] = "{}".format(value["action"])

    nx.draw_networkx_labels(graph, positions, label_data, font_size=6, font_type="bold")
    plt.title("Iterated Hawk And Dove")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(0, 1)
    plt.savefig(os.path.join(out_path, "{}".format(name)))


def get_position(x: float, data: List[Any], positions: Dict[str, Tuple[float, float]]) -> None:
    """Helper for assigning x, y position
    """
    ys: List[float] = [_ for _ in np.linspace(0, 0.9, num=len(data))]
    for i, node in enumerate(data):
        positions[node] = (x, ys[i])


def get_history_color(choice: str) -> str:
    """Return choice color from the history"""
    if choice == HawkAndDove.HAWK:
        _color = "r"
    elif choice == HawkAndDove.DOVE:
        _color = "g"
    else:
        raise Exception("Bad history value: {}".format(choice))

    return _color


def plot_hd_from_file(in_file_name: str, out_path: str = ".", name: str = "hd_test.pdf") -> None:
    """Plot from a Prisoners Dilemma statistics file"""
    with open(in_file_name, "r") as in_file:
        json_data = json.load(in_file)

    for i, data in enumerate(json_data):
        _name = "{}_{}".format(i, name)
        plot_hawk_and_dove(
            histories=data["history"], sentences=data["payoffs"], out_path=out_path, name=_name
        )
