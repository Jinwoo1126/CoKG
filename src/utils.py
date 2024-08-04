import os
import random
import networkx as nx
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from collections import Counter


def print_random_article(ds):
    random_index = random.randint(0, len(ds) - 1)
    article = ds[random_index]
    print(f"document: {article['document']}\n")
    print(f"summary: {article['summary']}")


def random_article(ds):
    random_index = random.randint(0, len(ds) - 1)
    return ds[random_index]


def print_scores(entities:dict):
    n = len(entities)
    keys = list(entities.keys())
    table = PrettyTable(keys)
    if (n == 0):
        n = 1
    table.add_row(
        [round(entities[key], 4) for key in keys]
    )
    print(table)

    return table.get_string()


def draw_knowledge_graph(data, file_name=None):
    G = nx.DiGraph()

    node_frequency = Counter()
    for head, relations in data.items():
        node_frequency[head] += len(relations)
        for relation, tail, weight in relations:
            node_frequency[tail] += 1

    for head, relations in data.items():
        for relation, tail, weight in relations:
            G.add_edge(head, tail, label=relation, weight=weight)

    pos = nx.spring_layout(G, k=5, iterations=100)

    plt.figure(figsize=(14, 10))
    
    node_size = [node_frequency[node] * 1000 for node in G.nodes()]

    def rescale_weights(weights, min_width=1, max_width=8):
        min_weight = min(weights)
        max_weight = max(weights)
        return [min_width + (w - min_weight) / (max_weight - min_weight) * (max_width - min_width) for w in weights]

    edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
    rescaled_edge_weights = rescale_weights(edge_weights)
    
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue', alpha=0.8)
    
    edges = G.edges(data=True)
    for (u, v, d), width in zip(edges, rescaled_edge_weights):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], arrowstyle='-|>', arrowsize=20, edge_color='gray',
            connectionstyle='arc3,rad=0.1', width=width, alpha=0.3  # 강도에 따라 선의 굵기 및 투명도 설정
        )
    
    edge_labels = {(u, v): f"{d['label']}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')

    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    
    plt.title('Knowledge Graph')
    
    file_path=os.path.join('kg_imgs', file_name if file_name else 'knowledge_graph.png')
    
    plt.savefig(file_path)

    