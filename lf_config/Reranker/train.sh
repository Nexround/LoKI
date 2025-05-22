echo '{
  "reranker_continuous_filt_max7_train": {
    "hf_hub_url": "lightblue/reranker_continuous_filt_max7_train",
    "formatting": "sharegpt"
  }
}' > /workspace/LLaMA-Factory/data/dataset_info.json

cd /workspace/LLaMA-Factory/ && llamafactory-cli train /home/ubuntu/LoKI/lf_config/Reranker/reranker.yaml
