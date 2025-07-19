from datasets import load_dataset, DatasetDict, concatenate_datasets

# Load datasets
humaneval_dataset = load_dataset("HeyixInn0/Reorganized-humaneval")
mbpp_dataset = load_dataset("mbpp", "sanitized", split="test")  # Using 'train' split for MBPP

mbpp_dataset = mbpp_dataset.map(lambda x: {'solution': x['prompt'] + '\n' + x['code']})

# Split each dataset into two halves
humaneval_train = humaneval_dataset["train"]
humaneval_split_1, humaneval_split_2 = humaneval_train.train_test_split(test_size=0.5).values()

mbpp_split_1, mbpp_split_2 = mbpp_dataset.train_test_split(test_size=0.5).values()

# Combine one half from each dataset
combined_dataset_1 = concatenate_datasets([humaneval_split_1, mbpp_split_1])
combined_dataset_2 = concatenate_datasets([humaneval_split_2, mbpp_split_2])

# Create a dictionary of combined datasets for easy access
combined_datasets = DatasetDict({
    "dataset_1": combined_dataset_1,
    "dataset_2": combined_dataset_2
})

# Print dataset sizes
print(f"Combined dataset 1 size: {len(combined_dataset_1)}")
print(f"Combined dataset 2 size: {len(combined_dataset_2)}")

# Save each combined dataset as a file
combined_dataset_1.save_to_disk("combined_dataset_1")
combined_dataset_2.save_to_disk("combined_dataset_2")

print("Datasets saved to disk.")