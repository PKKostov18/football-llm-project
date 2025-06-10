# src/02_corpus_creator.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import random
import config


def generate_qa_corpus():
    print("Loading all data files...")
    try:
        matches = pd.read_csv(config.PROCESSED_MATCHES_CSV)
        players = pd.read_csv(config.PROCESSED_PLAYERS_CSV)
        player_attrs = pd.read_csv('data/processed_player_attributes.csv')
        teams = pd.read_csv('data/processed_teams.csv')
        leagues = pd.read_csv('data/processed_leagues.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please run 01_data_extractor.py first.")
        return

    print("Preparing data structures for Q&A generation...")
    # Създаваме речници за бързо търсене
    player_id_to_name = players.set_index('player_api_id')['player_name'].to_dict()
    team_id_to_name = teams.set_index('team_api_id')['team_long_name'].to_dict()
    league_id_to_name = leagues.set_index('id')['name'].to_dict()
    player_attrs_map = player_attrs.set_index('player_api_id').to_dict('index')

    # Отваряме файла за запис
    with open(config.CORPUS_PATH, 'w', encoding='utf-8') as f:
        # --- 1. Генериране на въпроси за мачове ---
        print("Generating questions about matches...")
        for _, match in matches.head(5000).iterrows():  # Ограничаваме до 5000 мача за по-бърза работа
            home_team_name = team_id_to_name.get(match['home_team_api_id'])
            away_team_name = team_id_to_name.get(match['away_team_api_id'])
            league_name = league_id_to_name.get(match['league_id'])

            if not all([home_team_name, away_team_name, league_name]):
                continue

            # Въпрос за резултат
            q = f"Question: What was the score of the {league_name} match between {home_team_name} and {away_team_name} on {match['date'][:10]}?"
            a = f"Answer: The score was {home_team_name} {int(match['home_team_goal'])} - {int(match['away_team_goal'])} {away_team_name}."
            f.write(q + "\n" + a + "\n\n")

        # --- 2. Генериране на въпроси за играчи (атрибути, отбор, съотборници) ---
        print("Generating questions about players...")
        for player_id, player_info in player_attrs_map.items():
            player_name = player_id_to_name.get(player_id)
            if not player_name:
                continue

            # Въпрос за рейтинг
            if pd.notna(player_info['overall_rating']):
                q = f"Question: What is the overall rating of {player_name}?"
                a = f"Answer: The overall rating of {player_name} is {int(player_info['overall_rating'])}."
                f.write(q + "\n" + a + "\n\n")

            # Намиране на последния отбор и съотборници на играча
            player_matches = matches[
                (matches['home_player_1'] == player_id) | (matches['home_player_2'] == player_id) | (
                            matches['home_player_3'] == player_id) |
                (matches['home_player_4'] == player_id) | (matches['home_player_5'] == player_id) | (
                            matches['home_player_6'] == player_id) |
                (matches['home_player_7'] == player_id) | (matches['home_player_8'] == player_id) | (
                            matches['home_player_9'] == player_id) |
                (matches['home_player_10'] == player_id) | (matches['home_player_11'] == player_id) |
                (matches['away_player_1'] == player_id) | (matches['away_player_2'] == player_id) | (
                            matches['away_player_3'] == player_id) |
                (matches['away_player_4'] == player_id) | (matches['away_player_5'] == player_id) | (
                            matches['away_player_6'] == player_id) |
                (matches['away_player_7'] == player_id) | (matches['away_player_8'] == player_id) | (
                            matches['away_player_9'] == player_id) |
                (matches['away_player_10'] == player_id) | (matches['away_player_11'] == player_id)
                ].sort_values('date', ascending=False)

            if not player_matches.empty:
                last_match = player_matches.iloc[0]
                is_home_player = player_id in last_match[
                    ['home_player_1', 'home_player_2', 'home_player_3', 'home_player_4', 'home_player_5',
                     'home_player_6', 'home_player_7', 'home_player_8', 'home_player_9', 'home_player_10',
                     'home_player_11']].values

                team_id = last_match['home_team_api_id'] if is_home_player else last_match['away_team_api_id']
                team_name = team_id_to_name.get(team_id)
                league_name = league_id_to_name.get(last_match['league_id'])

                if team_name and league_name:
                    # Въпрос за отбор и лига
                    q = f"Question: Which team did {player_name} play for in the {last_match['season']} season?"
                    a = f"Answer: {player_name} played for {team_name} in the {league_name} during the {last_match['season']} season."
                    f.write(q + "\n" + a + "\n\n")

                    # Въпрос за съотборници
                    if is_home_player:
                        teammate_ids = [pid for pid in last_match[
                            ['home_player_1', 'home_player_2', 'home_player_3', 'home_player_4', 'home_player_5',
                             'home_player_6', 'home_player_7', 'home_player_8', 'home_player_9', 'home_player_10',
                             'home_player_11']].values if pd.notna(pid) and pid != player_id]
                    else:
                        teammate_ids = [pid for pid in last_match[
                            ['away_player_1', 'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5',
                             'away_player_6', 'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10',
                             'away_player_11']].values if pd.notna(pid) and pid != player_id]

                    if len(teammate_ids) > 2:
                        # Взимаме до 3-ма случайни съотборници
                        random_teammates = random.sample(teammate_ids, k=min(3, len(teammate_ids)))
                        teammate_names = [player_id_to_name.get(tid) for tid in random_teammates if
                                          player_id_to_name.get(tid)]
                        if teammate_names:
                            q = f"Question: Who were some of {player_name}'s teammates at {team_name}?"
                            a = f"Answer: Some of {player_name}'s teammates were {', '.join(teammate_names)}."
                            f.write(q + "\n" + a + "\n\n")

        # --- 3. Генериране на въпроси за лиги ---
        print("Generating questions about leagues...")
        for season, season_matches in matches.groupby('season'):
            for league_id, league_matches in season_matches.groupby('league_id'):
                league_name = league_id_to_name.get(league_id)
                if not league_name:
                    continue

                team_ids = pd.concat([league_matches['home_team_api_id'], league_matches['away_team_api_id']]).unique()
                team_names = [team_id_to_name.get(tid) for tid in team_ids if team_id_to_name.get(tid)]

                if len(team_names) > 5:
                    q = f"Question: Which teams were in the {league_name} during the {season} season?"
                    a = f"Answer: Some of the teams in the {league_name} during the {season} season were {', '.join(team_names[:5])}."
                    f.write(q + "\n" + a + "\n\n")

    print(f"The new corpus has been successfully created at {config.CORPUS_PATH}")


if __name__ == '__main__':
    generate_qa_corpus()