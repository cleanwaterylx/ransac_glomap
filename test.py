import networkx as nx

G = nx.Graph()  # 无向图
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(3,4)
G.add_edge(4,1)
G.add_edge(2,4)

all_simple_cycles = list(nx.simple_cycles(G))
print(all_simple_cycles)
