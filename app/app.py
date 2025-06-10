# app/app.py

# --- НАЧАЛО НА КОРЕКЦИЯТА ---
import sys
import os

# Добавяме основната папка на проекта (която е една папка над текущата 'app') към пътя на Python
# Това гарантира, че 'import config' ще намери правилния файл, който е в основната папка
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- КРАЙ НА КОРЕКЦИЯТА ---


from flask import Flask, render_template, request
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch
import config  # Този импорт вече ще работи правилно


app = Flask(__name__)

try:
    print("Зареждане на фино-настроения модел...")
    device = 0 if torch.cuda.is_available() else -1

    model_path = config.FINETUNED_MODEL_PATH
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    print("Моделът е зареден успешно!")
except Exception as e:
    print(f"Грешка при зареждане на модела: {e}")
    generator = None


@app.route('/', methods=['GET', 'POST'])
def home():
    generated_text = ""
    prompt_text = ""

    if request.method == 'POST':
        prompt_text = request.form.get('prompt', '')

        if generator and prompt_text:
            try:
                results = generator(prompt_text, max_length=150, num_return_sequences=1,
                                    pad_token_id=generator.tokenizer.eos_token_id)
                generated_text = results[0]['generated_text']
            except Exception as e:
                generated_text = f"Възникна грешка при генериране на текст: {e}"
        else:
            if not generator:
                generated_text = "Грешка: Моделът не е зареден правилно. Проверете конзолата."

    return render_template('index.html', generated_text=generated_text, prompt_text=prompt_text)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)