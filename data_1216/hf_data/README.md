---
configs:
  - config_name: v1.0
    description: "Version 1.0 of the dataset: (1) added id; (2) removed items with empty reference; (3) each subset label balanced"
    data_files:
      - split: train
        path: "train_all_subset_balanced.jsonl"
      - split: dev
        path: "dev_all_subset_balanced.jsonl"
      - split: test
        path: "test_all_subset_balanced.jsonl"
      - split: test_ood
        path: "test_ood_all_subset_balanced.jsonl"
  - config_name: v1.1
    description: "Version 1.1 of the dataset: (1) added id; (2) removed items with empty reference; (3) overall label balanced"
    data_files:
      - split: train
        path: "train_overall_balanced.jsonl"
      - split: dev
        path: "dev_all_subset_balanced.jsonl"
      - split: test
        path: "test_all_subset_balanced.jsonl"
      - split: test_ood
        path: "test_ood_all_subset_balanced.jsonl"
  - config_name: v1.2
    description: "Version 1.2 of the dataset: (1) added id; (2) removed items with empty reference; (3) all data included"
    data_files:
      - split: train
        path: "merged_train_sampled.jsonl"
      - split: dev
        path: "dev_all_subset_balanced.jsonl"
      - split: test
        path: "test_all_subset_balanced.jsonl"
      - split: test_ood
        path: "test_ood_all_subset_balanced.jsonl"
  - config_name: v1.3
    description: "Version 1.3 of the dataset: (1) added id; (2) removed items with empty reference; (3) all data included"
    data_files:
      - split: train
        path: "merged_train.jsonl"
      - split: dev
        path: "dev_all_subset_balanced.jsonl"
      - split: test
        path: "test_all_subset_balanced.jsonl"
      - split: test_ood
        path: "test_ood_all_subset_balanced.jsonl"

# description: |
#   Your general dataset description goes here.

# citation: |
#   If there's any citation related to your dataset, put it here.
# ... any other relevant sections ...
---