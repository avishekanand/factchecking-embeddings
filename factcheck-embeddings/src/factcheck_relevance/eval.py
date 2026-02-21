import argparse
import pandas as pd
import numpy as np
import logging
from factcheck_relevance.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_metrics(qrels_path, run_path):
    # Load qrels: query_id, 0, docid, 1
    qrels = {}
    with open(qrels_path, 'r') as f:
        for line in f:
            qid, _, docid, label = line.strip().split('\t')
            if int(label) > 0:
                if qid not in qrels:
                    qrels[qid] = set()
                qrels[qid].add(docid)
    
    # Load run: qid, docid, rank, score
    run = {}
    with open(run_path, 'r') as f:
        for line in f:
            qid, docid, rank, score = line.strip().split('\t')
            if qid not in run:
                run[qid] = []
            run[qid].append(docid)
            
    # Metrics
    recalls = {5: [], 10: [], 20: [], 50: []}
    mrrs = []
    ndcgs = []
    
    for qid, pos_docs in qrels.items():
        retrieved = run.get(qid, [])
        
        # Recall@k
        for k in recalls.keys():
            topk = set(retrieved[:k])
            recall = len(topk.intersection(pos_docs)) / len(pos_docs) if pos_docs else 0
            recalls[k].append(recall)
            
        # MRR@10
        mrr = 0
        for i, docid in enumerate(retrieved[:10]):
            if docid in pos_docs:
                mrr = 1 / (i + 1)
                break
        mrrs.append(mrr)
        
        # nDCG@10 (simplified binary)
        dcg = 0
        for i, docid in enumerate(retrieved[:10]):
            if docid in pos_docs:
                dcg += 1 / np.log2(i + 2)
        
        idcg = 0
        for i in range(min(len(pos_docs), 10)):
            idcg += 1 / np.log2(i + 2)
            
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
        
    results = {
        "MRR@10": np.mean(mrrs),
        "nDCG@10": np.mean(ndcgs)
    }
    for k, v in recalls.items():
        results[f"Recall@{k}"] = np.mean(v)
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config yaml")
    parser.add_argument("--qrels", help="Path to qrels tsv")
    parser.add_argument("--run", help="Path to run tsv")
    args = parser.parse_args()
    
    if args.config:
        config = load_config(args.config)
        qrels_path = config.get('qrels_path')
        run_path = config.get('run_path')
    else:
        qrels_path = args.qrels
        run_path = args.run
        
    if not qrels_path or not run_path:
        parser.error("Must provide --config or both --qrels and --run")
        
    metrics = compute_metrics(qrels_path, run_path)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
