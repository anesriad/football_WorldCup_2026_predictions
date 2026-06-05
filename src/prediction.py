import pandas as pd
import numpy as np
from scipy.stats import poisson

from src.data_loader import load_models, build_lookups


def build_match_features(home_team, away_team, neutral=True, tournament_weight=5):
    """
    Build one model-ready feature row.

    Current production model:
    - no confederation features
    - clean Elo setup
    - features:
        const
        home_elo_pre
        away_elo_pre
        tournament_weight
        neutral
    """

    _, _, feature_columns = load_models()
    lookups = build_lookups()

    team_to_elo = lookups["team_to_elo"]

    home_elo = team_to_elo.get(home_team)
    away_elo = team_to_elo.get(away_team)

    if home_elo is None:
        raise ValueError(f"Missing Elo rating for {home_team}")

    if away_elo is None:
        raise ValueError(f"Missing Elo rating for {away_team}")

    row = pd.DataFrame([{
        "const": 1.0,
        "home_elo_pre": home_elo,
        "away_elo_pre": away_elo,
        "tournament_weight": tournament_weight,
        "neutral": int(neutral),
    }])

    X = row.reindex(columns=feature_columns, fill_value=0)
    X = X.astype(float)

    return X


def predict_match(home_team, away_team, neutral=True, tournament_weight=5, max_goals=10):
    """Predict expected goals and win/draw/loss probabilities."""

    model_home, model_away, _ = load_models()

    X = build_match_features(
        home_team=home_team,
        away_team=away_team,
        neutral=neutral,
        tournament_weight=tournament_weight,
    )

    home_xg = float(model_home.predict(X)[0])
    away_xg = float(model_away.predict(X)[0])

    score_probs = []

    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob = float(
                poisson.pmf(home_goals, home_xg)
                * poisson.pmf(away_goals, away_xg)
            )

            score_probs.append({
                "home_goals": home_goals,
                "away_goals": away_goals,
                "probability": prob,
            })

            if home_goals > away_goals:
                home_win_prob += prob
            elif home_goals == away_goals:
                draw_prob += prob
            else:
                away_win_prob += prob

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "home_win_prob": home_win_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_win_prob,
        "score_probs": pd.DataFrame(score_probs),
    }


def get_h2h_score(team_a, team_b):
    """Return H2H score from team_a's point of view."""

    lookups = build_lookups()
    df_h2h = lookups["df_h2h"]

    direct_match = df_h2h[
        (df_h2h["team_a"] == team_a)
        & (df_h2h["team_b"] == team_b)
    ]

    if len(direct_match) > 0:
        return float(direct_match["h2h_score"].iloc[0])

    reverse_match = df_h2h[
        (df_h2h["team_a"] == team_b)
        & (df_h2h["team_b"] == team_a)
    ]

    if len(reverse_match) > 0:
        return -float(reverse_match["h2h_score"].iloc[0])

    return 0.0


def simulate_match(home_team, away_team, neutral=True, tournament_weight=5):
    """Simulate one group-stage match. Draws are allowed."""

    pred = predict_match(
        home_team=home_team,
        away_team=away_team,
        neutral=neutral,
        tournament_weight=tournament_weight,
    )

    home_goals = int(np.random.poisson(pred["home_xg"]))
    away_goals = int(np.random.poisson(pred["away_xg"]))

    if home_goals > away_goals:
        result = "H"
        winner = home_team
    elif away_goals > home_goals:
        result = "A"
        winner = away_team
    else:
        result = "D"
        winner = None

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_xg": pred["home_xg"],
        "away_xg": pred["away_xg"],
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": result,
        "winner": winner,
    }