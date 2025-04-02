import pandas as pd
import joblib

# Load all models
rf = joblib.load("models/random_forest.pkl")
lr = joblib.load("models/log_reg.pkl")
xgb = joblib.load("models/xgb.pkl")

def predict_all_models(fight_data):
    features = build_features(fight_data)
    X_input = pd.DataFrame([features], columns=[
        "age_diff", "height_diff", "weight_diff", "reach_diff",
        "SLpM_total_diff", "SApM_total_diff", "sig_str_acc_total_diff",
        "str_def_total_diff", "td_avg_diff", "td_acc_total_diff",
        "td_def_total_diff", "sub_avg_diff"
    ])

    preds = {
        "Random Forest": rf.predict_proba(X_input)[0],
        "Logistic Regression": lr.predict_proba(X_input)[0],
        "XGBoost": xgb.predict_proba(X_input)[0]
    }

    avg = sum(preds.values()) / len(preds)
    preds["Average"] = avg

    return {
        model: {
            "Red Win": round(prob[1], 3),
            "Blue Win": round(prob[0], 3)
        } for model, prob in preds.items()
    }

def get_value_bets(model_probs, bookmaker_odds):
    results = []
    for outcome, prob in model_probs.items():
        if outcome in bookmaker_odds:
            ev = (prob * bookmaker_odds[outcome]) - (1 - prob)
            if ev > 0:
                results.append({
                    "Bet": outcome,
                    "Model Chance": round(prob, 2),
                    "Bookmaker Odds": bookmaker_odds[outcome],
                    "EV": round(ev, 2)
                })
    return pd.DataFrame(results)


def build_features(fight_data):
    fighter_a, fighter_b = fight_data.split(" vs ")

    df = pd.read_csv("data/fighter_stats.csv")
    df = df.dropna(subset=["name"])

    f1 = df[df["name"].str.contains(fighter_a, case=False)].iloc[0]
    f2 = df[df["name"].str.contains(fighter_b, case=False)].iloc[0]

    feature_vector = {
        "age_diff": f1["age"] - f2["age"],
        "height_diff": f1["height"] - f2["height"],
        "weight_diff": f1["weight"] - f2["weight"],
        "reach_diff": f1["reach"] - f2["reach"],
        "SLpM_total_diff": f1["SLpM"] - f2["SLpM"],
        "SApM_total_diff": f1["SApM"] - f2["SApM"],
        "sig_str_acc_total_diff": f1["sig_str_acc"] - f2["sig_str_acc"],
        "str_def_total_diff": f1["str_def"] - f2["str_def"],
        "td_avg_diff": f1["td_avg"] - f2["td_avg"],
        "td_acc_total_diff": f1["td_acc"] - f2["td_acc"],
        "td_def_total_diff": f1["td_def"] - f2["td_def"],
        "sub_avg_diff": f1["sub_avg"] - f2["sub_avg"],
    }
    return list(feature_vector.values())

method_model = joblib.load("models/method_model.pkl")
round_model = joblib.load("models/round_model.pkl")

def predict_method_and_round(fight_data):
    features = build_features(fight_data)
    X_input = pd.DataFrame([features], columns=[
        "age_diff", "height_diff", "weight_diff", "reach_diff",
        "SLpM_total_diff", "SApM_total_diff", "sig_str_acc_total_diff",
        "str_def_total_diff", "td_avg_diff", "td_acc_total_diff",
        "td_def_total_diff", "sub_avg_diff"
    ])
    method = method_model.predict(X_input)[0]
    finish_round = round_model.predict(X_input)[0]
    return method, finish_round
