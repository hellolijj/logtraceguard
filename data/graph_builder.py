import torch
import networkx as nx
from torch_geometric.data import Data

class TraceGraphBuilder:
    def __init__(self):
        pass

    def build_graph(self, spans, log_embeddings):
        # 假设 spans 是 [(src, dst, timestamp), ...] 格式
        G = nx.DiGraph()
        node_map = {}
        for i, (src, dst, _) in enumerate(spans):
            G.add_edge(src, dst, index=i)
            node_map[src] = log_embeddings[src]
            node_map[dst] = log_embeddings[dst]

        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        x = torch.stack([node_map[n] for n in G.nodes])
        edge_attr = torch.stack([log_embeddings[src] + log_embeddings[dst] for src, dst in G.edges])

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def build_trace_graph(service_nodes, edges, node_logs, edge_logs, embedder):
        # service_nodes: [s1, s2, s3...]
        # edges: [(src_idx, dst_idx)]
        # node_logs: List[str]
        # edge_logs: List[str]（与 edges 对应）
        x = torch.stack([embedder.encode(log).squeeze(0) for log in node_logs])  # shape: [N, 768]
        edge_attr = torch.stack([embedder.encode(text).squeeze(0) for text in edge_logs])  # shape: [E, 768]
        edge_index = torch.tensor(edges, dtype=torch.long).T
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)