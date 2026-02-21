import argparse
import sys
import os
import glob
import json
from factcheck_relevance.utils import load_config
from tevatron.driver.encode import main as encode_main

def check_cache(output_dir, output_path, current_model):
    metadata_path = os.path.join(output_dir, "model_info.json")
    if os.path.exists(output_path) and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                info = json.load(f)
            if info.get('model_name_or_path') == current_model:
                # Also check if pkl exists
                if os.path.isdir(output_path):
                    if glob.glob(os.path.join(output_path, "*.pkl")):
                        return True
                elif output_path.endswith('.pkl'):
                    return True
        except:
            return False
    return False

def run_encode(config, is_query=False):
    # Prepare arguments for Tevatron driver
    tevatron_args = [
        "--output_dir", config['output_dir'],
        "--model_name_or_path", config['model_name_or_path'],
        "--p_max_len", str(config.get('p_max_len', 192)),
        "--q_max_len", str(config.get('q_max_len', 64)),
        "--per_device_eval_batch_size", str(config.get('per_device_eval_batch_size', 32)),
    ]
    
    if config.get('dataset_name'):
        tevatron_args.extend(["--dataset_name", config['dataset_name']])
        
    model_name = config['model_name_or_path']
    if is_query:
        target_path = config['query_out_path']
        if check_cache(config['output_dir'], target_path, model_name):
            print(f"Cache hit for {target_path}. Skipping encoding.")
            return
        in_path = os.path.abspath(config['query_in_path'])
        tevatron_args.append("--encode_is_qry")
        tevatron_args.extend(["--encode_in_path", in_path])
        tevatron_args.extend(["--encoded_save_path", target_path])
    else:
        target_path = config['corpus_out_path']
        if check_cache(config['output_dir'], target_path, model_name):
            print(f"Cache hit for {target_path}. Skipping encoding.")
            return
        in_path = os.path.abspath(config['corpus_in_path'])
        tevatron_args.extend(["--encode_in_path", in_path])
        tevatron_args.extend(["--encoded_save_path", target_path])

    # Set CUDA_VISIBLE_DEVICES="" to force CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    sys.argv = [sys.argv[0]] + tevatron_args
    encode_main()
    
    # Save model info after successful encoding
    metadata_path = os.path.join(config['output_dir'], "model_info.json")
    with open(metadata_path, 'w') as f:
        json.dump({'model_name_or_path': model_name}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--is_query", action="store_true", help="Whether to encode queries")
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_encode(config, is_query=args.is_query)
