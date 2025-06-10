# src/config.py

# --- Пътища до данни (КОРИГИРАНИ) ---
DB_PATH = 'data/database.sqlite'
PROCESSED_MATCHES_CSV = 'data/processed_matches.csv'
PROCESSED_PLAYERS_CSV = 'data/processed_players.csv'
# ПРОМЯНА ТУК: от .txt на .csv
CORPUS_PATH = 'data/processed_corpus.txt'


# --- Настройки на модела (КОРИГИРАНИ) ---
BASE_MODEL_NAME = 'gpt2-medium'
FINETUNED_MODEL_PATH = 'models/gpt2-football-finetuned'


# --- Настройки на обучението (опционално) ---
TRAIN_EPOCHS = 3
BATCH_SIZE = 4