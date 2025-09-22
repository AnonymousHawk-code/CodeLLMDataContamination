import vllm
from transformers import AutoTokenizer
from datasets import load_dataset


def fine_tune_model(model_name, output_dir, batch_size=8, num_epochs=3, learning_rate=1e-5):
    # Load the HumanEval dataset from HuggingFace Datasets
    dataset = load_dataset("openai/openai_humaneval", split='test')

    # Load the tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = vllm.LLM(model_name)

    # Prepare the data by combining the prompt and solution
    def preprocess_function(examples):
        # Concatenate the prompt and solution with a separator
        input_texts = [
            prompt + "\n" + solution
            for prompt, solution in zip(examples['prompt'], examples['canonical_solution'])
        ]
        return tokenizer(input_texts, padding="max_length", truncation=True, max_length=512)

    # Tokenize the dataset
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Set up the training configuration for vllm
    training_args = vllm.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,  # Not used, but required by Trainer
        num_train_epochs=num_epochs,
        logging_dir='./logs',
        logging_steps=500,
        learning_rate=learning_rate,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    # Fine-tuning using vllm
    trainer = vllm.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        # No eval_dataset included
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Example model, replace with any compatible model, e.g., "gpt-neo"
    output_dir = "/finetuned_models"

    fine_tune_model(model_name, output_dir)