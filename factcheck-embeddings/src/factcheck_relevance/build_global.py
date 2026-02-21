import argparse
import logging
import os
from factcheck_relevance.utils import load_config, load_json, save_jsonl, save_tsv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_global(config):
    # input_paths should be a list in global_data_build.yaml
    input_paths = config['input_paths']
    out_dir = config['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    global_corpus = {}  # text -> docid (for dedup across files)
    final_corpus = []   # list of {text_id, text}
    
    all_queries = []
    all_qrels = []
    
    doc_counter = 0
    
    for file_path in input_paths:
        logger.info(f"Processing {file_path}...")
        data = load_json(file_path)
        file_basename = os.path.basename(file_path).replace(".json", "")
        
        for i, item in enumerate(tqdm(data, desc=f"Extracting from {file_basename}")):
            claim_text = item['claim']
            claim_id = f"{file_basename}_c{i:06d}"
            
            # Record query
            all_queries.append({"text_id": claim_id, "text": claim_text})
            
            evidences = item.get('evidence', [])
            for ev in evidences:
                snippet = ev['snippet']
                label = ev['relevance_label']
                mapped_label = config['label_mapping'].get(label, 'drop')
                
                # Global Deduplication
                if snippet not in global_corpus:
                    doc_id = f"g_doc_{doc_counter:08d}"
                    global_corpus[snippet] = doc_id
                    final_corpus.append({"text_id": doc_id, "text": snippet})
                    doc_counter += 1
                
                doc_id = global_corpus[snippet]
                
                # If positive, add to qrels
                if mapped_label == 'positive':
                    all_qrels.append([claim_id, 0, doc_id, 1])

    # Save outputs
    logger.info(f"Saving global corpus ({len(final_corpus)} unique snippets)...")
    save_jsonl(final_corpus, os.path.join(out_dir, "corpus.jsonl"))
    
    logger.info(f"Saving all queries ({len(all_queries)} total)...")
    save_jsonl(all_queries, os.path.join(out_dir, "queries.jsonl"))
    
    logger.info(f"Saving all qrels ({len(all_qrels)} instances)...")
    save_tsv(all_qrels, os.path.join(out_dir, "qrels.tsv"))
    
    logger.info("Global data build complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    build_global(config)
