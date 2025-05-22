echo '{
  "ToolACE": {
    "hf_hub_url": "/workspace/ToolACE",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "system": "system"
  }
  }
}' > /workspace/LLaMA-Factory/data/dataset_info.json

cd /workspace/LLaMA-Factory/ && llamafactory-cli train /home/ubuntu/LoKI/lf_config/ToolACE/toolace.yaml