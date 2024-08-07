from transformers import (
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataset import dataset
from huggingface_hub import notebook_login

# Load pre-trained model and tokenizer
model_name = "facebook/mbart-large-50"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="km_KH", tgt_lang="vi_VN")
model = MBartForConditionalGeneration.from_pretrained(model_name)

source_lang = "km"
target_lang = "vi"

def preprocess_function(examples):
    inputs = [example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



if __name__ == "__main__":
    # Preprocess the dataset
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Initialize the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True
    )

    # Initialize the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Login to huggingface hub
    notebook_login()

    # Save model and tokenizer
    model.save_pretrained('fine-tuned-mbart-50-km-vi')
    tokenizer.save_pretrained('fine-tuned-mbart-50-km-vi')
    model.push_to_hub('fine-tuned-mbart-50-km-vi')
    tokenizer.push_to_hub('fine-tuned-mbart-50-km-vi')