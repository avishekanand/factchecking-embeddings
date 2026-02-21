import os
import pandas as pd
from factcheck_relevance.eval import compute_metrics
from factcheck_relevance.utils import load_config

def main():
    runs_dir = "runs"
    all_results = []
    
    # Mapping of run file names to their likely qrels
    qrels_map = {
        "dev.run": "data/tevatron/dev_qrels.tsv",
        "test.run": "data/tevatron_test/dev_qrels.tsv",
        "global.run": "data/global/qrels.tsv"
    }
    
    for run_name in os.listdir(runs_dir):
        run_dir = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_dir):
            continue
        
        # Look for any .run files in the subdirectory
        run_files = [f for f in os.listdir(run_dir) if f.endswith(".run")]
        
        for rf in run_files:
            run_file_path = os.path.join(run_dir, rf)
            qrels_path = qrels_map.get(rf, "data/tevatron/dev_qrels.tsv")
            
            if not os.path.exists(qrels_path):
                continue

            try:
                metrics = compute_metrics(qrels_path, run_file_path)
                metrics['Model/Run'] = f"{run_name} ({rf})"
                all_results.append(metrics)
            except Exception as e:
                print(f"Could not evaluate {run_name}/{rf}: {e}")
            
    if not all_results:
        print("No results found.")
        return
        
    df = pd.DataFrame(all_results)
    # Reorder columns to put Model/Run first
    cols = ['Model/Run'] + [c for c in df.columns if c != 'Model/Run']
    df = df[cols]
    
    print("\n--- Comparison Table ---")
    print(df.to_markdown(index=False))
    print("\n")

if __name__ == "__main__":
    main()
