# src/04_inference.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch
import config


# ... останалата част от файла остава същата ...

def test_model_generation(prompt_text, max_length=100):
    """
    Зарежда фино-настроения модел и генерира текст по подаден prompt.
    """
    print("Зареждане на модела за тестване...")
    try:
        model_path = config.FINETUNED_MODEL_PATH
        device = 0 if torch.cuda.is_available() else -1

        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)

        generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        print("Моделът е зареден успешно.")
    except Exception as e:
        print(f"Грешка при зареждане на модела: {e}")
        return

    print("-" * 30)
    print(f"Входен текст (Prompt): {prompt_text}")
    print("-" * 30)

    results = generator(prompt_text, max_length=max_length, num_return_sequences=1,
                        pad_token_id=generator.tokenizer.eos_token_id)
    generated_text = results[0]['generated_text']

    print("Генериран отговор:")
    print(generated_text)
    print("-" * 30)


if __name__ == '__main__':
    # Променяме тестовите въпроси да са на английски
    test_prompt = "Question: What is Cristiano Ronaldo's preferred foot?"
    test_model_generation(test_prompt, max_length=120)

    print("\n\n")

    test_prompt_2 = "Question: Who won the match between Real Madrid C.F. and FC Barcelona in the 2015/2016 season?"
    test_model_generation(test_prompt_2, max_length=120)