# tools/process.py
# Unified processor to handle multiple datasets with different parsing logic

import os
import torch
import json
from torch_geometric.data import Data, InMemoryDataset


class DatasetProcessor:
    def __init__(self, name, raw_dir, out_file):
        self.name = name
        self.raw_dir = raw_dir
        self.out_file = out_file

    def process(self):
        if self.name == 'dataset1':
            return self._process_dataset1()
        elif self.name == 'dataset2':
            return self._process_dataset2()
        else:
            raise ValueError(f"Unsupported dataset name: {self.name}")

    def _process_dataset1(self):
        graphs = []
        service_vocab = set()
        trace_files = [f for f in os.listdir(self.raw_dir) if f.endswith('.json')]

        for f in trace_files:
            spans = self._load_trace(os.path.join(self.raw_dir, f))
            service_vocab |= self._collect_services(spans)

        service_list = sorted(service_vocab)
        for f in trace_files:
            spans = self._load_trace(os.path.join(self.raw_dir, f))
            g = self._convert_trace_to_graph(spans, service_list)
            graphs.append(g)

        self._save(graphs)
        return graphs

    def _process_dataset2(self):
        # Support for dataset2: synthetic log-based graphs
        graphs = []
        dataset_path = os.path.join(self.raw_dir, 'dataset2.json')
        with open(dataset_path) as f:
            entries = json.load(f)  # list of traces

        for entry in entries:
            x = torch.tensor(entry['x'], dtype=torch.float)
            edge_index = torch.tensor(entry['edge_index'], dtype=torch.long)
            edge_attr = torch.tensor(entry['edge_attr'], dtype=torch.float)
            y = torch.tensor([entry['label']], dtype=torch.long)
            graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

        self._save(graphs)
        return graphs

    def _load_trace(self, path):
        with open(path) as f:
            return json.load(f)["spans"]

    def _collect_services(self, spans):
        names = set()
        for s in spans:
            for tag in s.get("tags", []):
                if tag["key"] == "http.url":
                    names.add(self._extract_service(tag["value"]))
        return names

    def _extract_service(self, url):
        return url.split("/")[2].split(":")[0]

    def _convert_trace_to_graph(self, spans, service_list):
        node_idx = {}
        node_feat = []
        edge_index = [[], []]
        edge_attr = []

        for span in spans:
            svc = self._extract_service_from_span(span)
            if svc not in node_idx:
                node_idx[svc] = len(node_idx)
                onehot = [0.0] * len(service_list)
                onehot[service_list.index(svc)] = 1.0
                node_feat.append(onehot)

        for span in spans:
            child = self._extract_service_from_span(span)
            parent = None
            for ref in span.get("references", []):
                ref_id = ref["spanID"]
                for other in spans:
                    if other["spanID"] == ref_id:
                        parent = self._extract_service_from_span(other)
            if parent in node_idx and child in node_idx:
                edge_index[0].append(node_idx[parent])
                edge_index[1].append(node_idx[child])
                edge_attr.append([1.0])

        x = torch.tensor(node_feat, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        y = torch.tensor([1])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def _extract_service_from_span(self, span):
        for tag in span.get("tags", []):
            if tag["key"] == "http.url":
                return self._extract_service(tag["value"])
        return "unknown"

    def _save(self, graphs):
        data, slices = InMemoryDataset.collate(None, graphs)
        torch.save((data, slices), self.out_file)
        print(f"[✓] Saved {len(graphs)} graphs to {self.out_file}")


if __name__ == '__main__':
    # Example usage
    processor = DatasetProcessor(
        name='dataset1',
        raw_dir='datasets/raw/dataset1/traces',
        out_file='datasets/processed/dataset1.pt'
    )
    processor.process()