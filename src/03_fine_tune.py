# src/03_fine_tune.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import config


# ... останалата част от файла остава същата ...

def fine_tune_model():
    model_name = config.BASE_MODEL_NAME
    train_file = config.CORPUS_PATH
    output_dir = config.FINETUNED_MODEL_PATH

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    print(f"Зареждане на данни от текстов файл: {train_file}")
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=config.TRAIN_EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_steps=5000,
        save_total_limit=5,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    latest_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
            latest_checkpoint = os.path.join(output_dir, latest_checkpoint)
            print(f"Намерен е checkpoint! Обучението ще продължи от: {latest_checkpoint}")

    print("=" * 50)
    print("СТАРТИРАНЕ НА ФИНО-НАСТРОЙВАНЕ...")
    print("=" * 50)

    trainer.train(resume_from_checkpoint=latest_checkpoint)

    print("=" * 50)
    print("Фино-настройването завърши УСПЕШНО!")
    print("=" * 50)

    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Финалният модел е запазен в {output_dir}")


if __name__ == '__main__':
    fine_tune_model()