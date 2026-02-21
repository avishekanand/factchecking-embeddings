import pytest
import os
import json
from factcheck_relevance.build_data import build_data

@pytest.fixture
def mock_config(tmp_path):
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    tevatron_dir = data_dir / "tevatron"
    raw_dir.mkdir(parents=True)
    tevatron_dir.mkdir(parents=True)
    
    input_path = raw_dir / "test_input.json"
    raw_data = [
        {
            "claim": "Claim 1",
            "evidence": [
                {"snippet": "Pos 1", "relevance_label": "RELEVANT", "cosine_similarity": 0.9},
                {"snippet": "Pos 1", "relevance_label": "RELEVANT", "cosine_similarity": 0.9}, # Dupe
                {"snippet": "Neg 1", "relevance_label": "NOT_RELEVANT", "cosine_similarity": 0.1},
                {"snippet": "Neg 2", "relevance_label": "NOT_RELEVANT", "cosine_similarity": 0.2},
                {"snippet": "Neg 3", "relevance_label": "NOT_RELEVANT", "cosine_similarity": 0.3},
                {"snippet": "Neg 4", "relevance_label": "NOT_RELEVANT", "cosine_similarity": 0.8}, # Hard neg
                {"snippet": "Err 1", "relevance_label": "ERROR", "cosine_similarity": 0.5}
            ]
        }
    ]
    with open(input_path, 'w') as f:
        json.dump(raw_data, f)
        
    return {
        "input_path": str(input_path),
        "out_dir": str(tevatron_dir),
        "dev_ratio": 0.0,
        "seed": 42,
        "k_neg": 2,
        "hard_pool_size": 1,
        "hard_frac": 1.0,
        "label_mapping": {
            "RELEVANT": "positive",
            "PARTIALLY_RELEVANT": "positive",
            "NOT_RELEVANT": "negative",
            "ERROR": "drop"
        }
    }

def test_build_data_logic(mock_config):
    build_data(mock_config)
    
    out_dir = mock_config['out_dir']
    train_path = os.path.join(out_dir, "train.jsonl")
    corpus_path = os.path.join(out_dir, "corpus.jsonl")
    
    assert os.path.exists(train_path)
    assert os.path.exists(corpus_path)
    
    with open(train_path, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 1 # 1 positive after dedup
        item = json.loads(lines[0])
        assert item['query'] == "Claim 1"
        assert len(item['positive_passages']) == 1
        assert len(item['negative_passages']) == 2
        # Hard neg should be Neg 4 (sim 0.8)
        assert item['negative_passages'][0]['text'] == "Neg 4"
        
    with open(corpus_path, 'r') as f:
        corpus = [json.loads(l) for l in f]
        docids = [c['docid'] for c in corpus]
        assert len(docids) == len(set(docids)) # Unique docids
        assert len(corpus) == 5 # 1 pos, 4 negs (Err dropped, Pos dupe dropped)
