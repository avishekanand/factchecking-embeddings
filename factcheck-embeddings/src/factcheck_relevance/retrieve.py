import argparse
import numpy as np
import faiss
import logging
import pickle
import os
from tqdm import tqdm
from factcheck_relevance.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_retrieval(config):
    query_reps_path = config['query_out_path']
    corpus_reps_path = config['corpus_out_path']
    save_path = config['run_path']
    topk = config.get('topk', 100)
    
    logger.info(f"Loading query representations from {query_reps_path}")
    # Tevatron saves as pickle maps or multiple files. We assume standard glob or single file here.
    # For simplicity, we handle the case where tevatron saves as list of pickles (default behavior)
    
    def load_reps(path):
        # Handle directory or file
        if os.path.isdir(path):
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pkl')]
        else:
            files = [path]
        
        all_reps = []
        all_ids = []
        for f in sorted(files):
            with open(f, 'rb') as reader:
                data = pickle.load(reader)
                # data is usually (reps, ids) in Tevatron
                if isinstance(data[0], np.ndarray) and data[0].ndim > 1:
                    reps, ids = data[0], data[1]
                elif isinstance(data[1], np.ndarray) and data[1].ndim > 1:
                    ids, reps = data[0], data[1]
                else:
                    reps, ids = data[0], data[1]
                
                all_ids.extend(ids)
                if isinstance(reps, list):
                    reps = np.array(reps)
                all_reps.append(reps)
        return all_ids, np.concatenate(all_reps, axis=0)

    query_ids, query_reps = load_reps(query_reps_path)
    corpus_ids, corpus_reps = load_reps(corpus_reps_path)
    
    dim = query_reps.shape[1]
    index = faiss.IndexFlatIP(dim)
    
    logger.info("Building FAISS index...")
    index.add(corpus_reps)
    
    logger.info(f"Searching for top-{topk}...")
    scores, indices = index.search(query_reps, topk)
    
    logger.info(f"Saving results to {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as writer:
        for i, qid in enumerate(query_ids):
            for j in range(topk):
                docid = corpus_ids[indices[i][j]]
                score = scores[i][j]
                writer.write(f"{qid}\t{docid}\t{j+1}\t{score}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_retrieval(config)
