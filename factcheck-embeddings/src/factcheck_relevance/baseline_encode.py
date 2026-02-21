import argparse
import sys
import os
import json
import logging
import pickle
import glob
from factcheck_relevance.utils import load_config, save_jsonl
from tevatron.driver.encode import main as encode_main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_prefixes(in_path, out_path, prefix):
    logger.info(f"Applying prefix '{prefix}' to {in_path} -> {out_path}")
    with open(in_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    prefixed_data = []
    for item in data:
        # Tevatron expects 'text_id' and 'text'
        prefixed_item = {
            "text_id": item['text_id'],
            "text": f"{prefix}{item['text']}"
        }
        prefixed_data.append(prefixed_item)
    
    save_jsonl(prefixed_data, out_path)

def check_cache(output_dir, output_path, current_model):
    metadata_path = os.path.join(output_dir, "model_info.json")
    if os.path.exists(output_path) and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                info = json.load(f)
            if info.get('model_name_or_path') == current_model:
                # Check if pkl exists
                if os.path.isdir(output_path):
                    if glob.glob(os.path.join(output_path, "*.pkl")):
                        return True
                elif output_path.endswith('.pkl'):
                    return True
        except:
            return False
    return False

def run_encode_with_cache(config, is_query=False):
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    target_path = config['query_out_path'] if is_query else config['corpus_out_path']
    model_name = config['model_name_or_path']
    
    if check_cache(output_dir, target_path, model_name):
        logger.info(f"Cache hit for {target_path}. Skipping encoding.")
        return

    # Prepare prefixed data
    prefix = config.get('query_prefix', '') if is_query else config.get('document_prefix', '')
    original_in_path = config['query_in_path'] if is_query else config['corpus_in_path']
    
    prefixed_in_path = os.path.join(output_dir, f"prefixed_{'query' if is_query else 'corpus'}.jsonl")
    apply_prefixes(original_in_path, prefixed_in_path, prefix)

    # Prepare arguments for Tevatron driver
    tevatron_args = [
        "--output_dir", output_dir,
        "--model_name_or_path", config['model_name_or_path'],
        "--p_max_len", str(config.get('p_max_len', 192)),
        "--q_max_len", str(config.get('q_max_len', 64)),
        "--per_device_eval_batch_size", str(config.get('per_device_eval_batch_size', 32)),
        "--encode_in_path", prefixed_in_path,
        "--encoded_save_path", target_path,
    ]
    
    if is_query:
        tevatron_args.append("--encode_is_qry")

    # Set CUDA_VISIBLE_DEVICES="" to force CPU if needed, 
    # but here we allow it if available unless forced
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    logger.info(f"Running Tevatron encode for {'query' if is_query else 'corpus'}...")
    sys.argv = [sys.argv[0]] + tevatron_args
    encode_main()

    # Save model info after successful encoding
    metadata_path = os.path.join(output_dir, "model_info.json")
    with open(metadata_path, 'w') as f:
        json.dump({'model_name_or_path': model_name}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--is_query", action="store_true", help="Whether to encode queries")
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_encode_with_cache(config, is_query=args.is_query)
