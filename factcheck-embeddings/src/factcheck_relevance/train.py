import argparse
import sys
import os
from factcheck_relevance.utils import load_config
from tevatron.driver.train import main as train_main
from tevatron.trainer import DenseTrainer

# Monkeypatch DenseTrainer.compute_loss for compatibility with newer transformers
def compute_loss_patched(self, model, inputs, return_outputs=False, **kwargs):
    query, passage = inputs
    outputs = model(query=query, passage=passage)
    loss = outputs.loss
    return (loss, outputs) if return_outputs else loss

DenseTrainer.compute_loss = compute_loss_patched

def run_train(config):
    # Prepare arguments for Tevatron driver
    train_dir = os.path.abspath(config['train_dir'])
    tevatron_args = [
        "--output_dir", config['output_dir'],
        "--model_name_or_path", config['model_name_or_path'],
        "--train_dir", train_dir,
        "--do_train",
        "--overwrite_output_dir",
        "--per_device_train_batch_size", str(config['per_device_train_batch_size']),
        "--train_n_passages", str(config['train_n_passages']),
        "--learning_rate", str(config['learning_rate']),
        "--q_max_len", str(config['q_max_len']),
        "--p_max_len", str(config['p_max_len']),
        "--num_train_epochs", str(config['num_train_epochs']),
        "--save_steps", str(config['save_steps']),
    ]
    
    if config.get('dataset_name'):
        tevatron_args.extend(["--dataset_name", config['dataset_name']])
    
    if not config.get('fp16', False):
        # Default is usually false but good to be explicit if they had it
        pass 
    else:
        tevatron_args.append("--fp16")

    # Set CUDA_VISIBLE_DEVICES="" to force CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Mock sys.argv to pass to Tevatron
    sys.argv = [sys.argv[0]] + tevatron_args
    
    train_main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training config")
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_train(config)
