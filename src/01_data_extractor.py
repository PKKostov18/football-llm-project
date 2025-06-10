# src/01_data_extractor.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sqlite3
import pandas as pd
import config


def extract_all_data():
    """
    Извлича цялата необходима информация от базата данни в отделни, лесни за обработка CSV файлове.
    """
    try:
        con = sqlite3.connect(config.DB_PATH)
        print("Свързване с базата данни...")

        # 1. Извличане на мачовете
        print("Извличане на данни за мачове...")
        match_query = "SELECT * FROM Match"
        matches_df = pd.read_sql_query(match_query, con)

        # 2. Извличане на играчите (КОРИГИРАНА ЗАЯВКА - добавени height и weight)
        print("Извличане на данни за играчи...")
        player_query = "SELECT player_api_id, player_name, height, weight FROM Player"
        players_df = pd.read_sql_query(player_query, con)

        # 3. Извличане на атрибутите на играчите (КОРИГИРАНА ЗАЯВКА - премахнати height и weight)
        print("Извличане на атрибути на играчи...")
        player_attr_query = "SELECT player_api_id, date, overall_rating, potential, preferred_foot FROM Player_Attributes"
        player_attrs_df = pd.read_sql_query(player_attr_query, con)
        # Взимаме само последния запис за всеки играч
        player_attrs_df = player_attrs_df.sort_values('date', ascending=False).drop_duplicates('player_api_id')

        # 4. Извличане на отборите
        print("Извличане на данни за отбори...")
        team_query = "SELECT team_api_id, team_long_name FROM Team"
        teams_df = pd.read_sql_query(team_query, con)

        # 5. Извличане на лигите
        print("Извличане на данни за лиги...")
        league_query = "SELECT id, name FROM League"
        leagues_df = pd.read_sql_query(league_query, con)

        # Запазване на всичко в CSV файлове
        print("Запазване на данните в CSV файлове...")
        matches_df.to_csv(config.PROCESSED_MATCHES_CSV, index=False)
        players_df.to_csv(config.PROCESSED_PLAYERS_CSV, index=False)
        player_attrs_df.to_csv('data/processed_player_attributes.csv', index=False)
        teams_df.to_csv('data/processed_teams.csv', index=False)
        leagues_df.to_csv('data/processed_leagues.csv', index=False)

        print("Всички данни са извлечени успешно.")

    except Exception as e:
        print(f"Възникна грешка при извличане на данните: {e}")
    finally:
        if 'con' in locals() and con:
            con.close()


if __name__ == '__main__':
    extract_all_data()