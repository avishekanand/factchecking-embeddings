import argparse
import random
import math
import logging
import os
from tqdm import tqdm
from factcheck_relevance.utils import load_config, load_json, save_jsonl, save_tsv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_data(config):
    raw_data = load_json(config['input_path'])
    seed = config.get('seed', 42)
    random.seed(seed)
    
    # Shuffle and split claims
    random.shuffle(raw_data)
    num_claims = len(raw_data)
    dev_size = int(num_claims * config.get('dev_ratio', 0.05))
    dev_claims = raw_data[:dev_size]
    train_claims = raw_data[dev_size:]
    
    logger.info(f"Total claims: {num_claims}, Train: {len(train_claims)}, Dev: {len(dev_claims)}")
    
    corpus = {}  # docid -> {text, title}
    
    def process_claims(claims, split_name):
        instances = []
        qrels = [] if split_name == 'dev' else None
        queries = [] if split_name == 'dev' else None
        
        skipped_no_pos = 0
        
        for i, c_data in enumerate(tqdm(claims, desc=f"Processing {split_name}")):
            claim_id = f"{split_name}_c{i:06d}"
            claim_text = c_data['claim']
            
            evidences = c_data.get('evidence', [])
            
            positives = []
            negatives = []
            
            seen_snippets = set()
            
            for j, ev in enumerate(evidences):
                snippet = ev['snippet']
                label = ev['relevance_label']
                sim = ev.get('cosine_similarity', 0.0)
                
                # Deduplication within claim
                if snippet in seen_snippets:
                    continue
                seen_snippets.add(snippet)
                
                docid = f"d_{claim_id}_{j:04d}"
                mapped_label = config['label_mapping'].get(label, 'drop')
                
                doc_obj = {"docid": docid, "title": "", "text": snippet}
                
                if mapped_label == 'positive':
                    positives.append((docid, snippet, sim))
                    corpus[docid] = {"title": "", "text": snippet}
                elif mapped_label == 'negative':
                    negatives.append((docid, snippet, sim))
                    corpus[docid] = {"title": "", "text": snippet}
            
            if split_name == 'dev':
                queries.append({"query_id": claim_id, "query": claim_text})
                for pid, _, _ in positives:
                    qrels.append([claim_id, 0, pid, 1])
            
            if not positives:
                skipped_no_pos += 1
                continue
                
            # Training instance generation (only for train split)
            if split_name == 'train':
                for pid, p_snippet, _ in positives:
                    instance = {
                        "query": claim_text,
                        "positives": [p_snippet],
                        "negatives": []
                    }
                    
                    # Sample negatives
                    k_neg = config.get('k_neg', 3)
                    if len(negatives) < k_neg:
                        # Not enough negatives in this claim, just take all available
                        sampled_negs = negatives
                    else:
                        # Hard negative sampling
                        hard_pool_size = config.get('hard_pool_size', 20)
                        hard_frac = config.get('hard_frac', 0.67)
                        
                        # Sort by similarity descending
                        sorted_negs = sorted(negatives, key=lambda x: x[2], reverse=True)
                        n_hard_pool = sorted_negs[:hard_pool_size]
                        n_rand_pool = sorted_negs[hard_pool_size:]
                        
                        k_hard = math.ceil(hard_frac * k_neg)
                        k_rand = k_neg - k_hard
                        
                        sampled_hard = random.sample(n_hard_pool, min(len(n_hard_pool), k_hard))
                        remaining_rand = k_neg - len(sampled_hard)
                        
                        # Combine remaining pool
                        rest_pool = [n for n in negatives if n[0] not in [s[0] for s in sampled_hard]]
                        sampled_rand = random.sample(rest_pool, min(len(rest_pool), remaining_rand))
                        
                        sampled_negs = sampled_hard + sampled_rand
                        
                    for nid, n_snippet, _ in sampled_negs:
                        instance["negatives"].append(n_snippet)
                    
                    if len(instance["negatives"]) >= k_neg:
                        instances.append(instance)

        logger.info(f"{split_name} split: processed {len(claims)} claims, skipped {skipped_no_pos} due to no positives.")
        return instances, queries, qrels

    train_instances, _, _ = process_claims(train_claims, 'train')
    dev_instances, dev_queries, dev_qrels = process_claims(dev_claims, 'dev')
    
    # Save outputs
    out_dir = config['out_dir']
    save_jsonl(train_instances, os.path.join(out_dir, "train", "train.jsonl"))
    save_jsonl(dev_instances, os.path.join(out_dir, "dev", "dev.jsonl"))
    
    # Corpus
    corpus_list = [{"text_id": k, "text": v["text"]} for k, v in corpus.items()]
    save_jsonl(corpus_list, os.path.join(out_dir, "corpus.jsonl"))
    
    # Dev artifacts
    if dev_queries:
        # Format queries for encoding
        query_list = [{"text_id": q['query_id'], "text": q['query']} for q in dev_queries]
        save_jsonl(query_list, os.path.join(out_dir, "dev_queries.jsonl"))
    if dev_qrels:
        save_tsv(dev_qrels, os.path.join(out_dir, "dev_qrels.tsv"))

    logger.info("Data building complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    build_data(config)
