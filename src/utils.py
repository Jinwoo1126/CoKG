import os
import random
import networkx as nx
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from collections import Counter
import numpy as np


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
    G = nx.DiGraph()  # 방향 그래프를 사용

    # 노드 빈도수 계산
    node_frequency = Counter()
    for head, relations in data.items():
        if isinstance(head, list):
            for h in head:
                node_frequency[h] += len(relations)
        else:
            node_frequency[head] += len(relations)
            
        for relation_data in relations:
            tail = relation_data['Tail']
            if isinstance(tail, list):
                for t in tail:
                    node_frequency[t] += 1
            else:
                node_frequency[tail] += 1

    # 노드 및 엣지 추가
    for head, relations in data.items():
        for relation_data in relations:
            relation = relation_data['Relation']
            tail = relation_data['Tail']
            weight = relation_data['Relation Strength']
            if isinstance(head, list):
                for h in head:
                    if isinstance(tail, list):
                        for t in tail:
                            G.add_edge(h, t, label=relation, weight=weight)
                    else:
                        G.add_edge(h, tail, label=relation, weight=weight)
            else:
                if isinstance(tail, list):
                    for t in tail:
                        G.add_edge(head, t, label=relation, weight=weight)
                else:
                    G.add_edge(head, tail, label=relation, weight=weight)

    pos = nx.spring_layout(G, k=7, iterations=100)  # 그래프 레이아웃, k값을 크게 설정하여 노드 간의 거리 넓히기

    plt.figure(figsize=(14, 10))
    
    # 노드 크기 설정 (빈도수에 따라)
    node_size = [node_frequency[node] * 1000 for node in G.nodes()]

    # 엣지 굵기 조정 함수
    def rescale_weights(weights, min_width=1, max_width=8):
        min_weight = min(weights)
        max_weight = max(weights)
        if min_weight == max_weight:
            return [max_width for _ in weights]
        return [min_width + (w - min_weight) / (max_weight - min_weight) * (max_width - min_width) for w in weights]

    # 간선 굵기 계산
    edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
    rescaled_edge_weights = rescale_weights(edge_weights)
    
    # 노드 그리기 (투명도 추가)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue', alpha=0.8)
    
    # 엣지 그리기 (투명도 및 굵기 추가)
    edges = G.edges(data=True)
    for (u, v, d), width in zip(edges, rescaled_edge_weights):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], arrowstyle='-|>', arrowsize=20, edge_color='gray',
            connectionstyle='arc3,rad=0.1', width=width, alpha=0.3  # 강도에 따라 선의 굵기 및 투명도 설정
        )
    
    # 엣지 레이블 그리기
    edge_labels = {(u, v): f"{d['label']}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')

    # 노드 레이블 그리기
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    
    plt.title('Knowledge Graph')
    
    if file_name:
        directory = 'results/CoKG_Graph'
        if not os.path.exists(directory):
            os.makedirs(directory)    
        file_path = os.path.join(directory, file_name)
        plt.savefig(file_path)
        plt.close()
    else:
        plt.show()
        plt.close()