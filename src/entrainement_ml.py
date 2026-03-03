"""
=============================================================
 SYSTÈME D'IRRIGATION INTELLIGENTE — CÔTE D'IVOIRE
 Étape 2 : Entraînement ML (Random Forest + XGBoost)
=============================================================
 Dataset   : 2022-2024 (1096 jours)
 Modèles   :
   A — RandomForestClassifier  → décision OUI/NON
   B — XGBClassifier           → décision OUI/NON (comparaison)
   C — RandomForestRegressor   → volume (litres)
   D — XGBRegressor            → volume (litres) (comparaison)
=============================================================
"""

import os, sys, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import (accuracy_score, f1_score,
                                     mean_absolute_error, r2_score,
                                     mean_squared_error, classification_report,
                                     confusion_matrix)
warnings.filterwarnings("ignore")

# ─── Chemins ────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)

CSV_CLEAN = os.path.join(DATA_DIR, "yamoussoukro_dataset_ML.csv")

# ─── XGBoost optionnel ──────────────────────────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_OK = True
    print("✅ XGBoost disponible")
except ImportError:
    XGBOOST_OK = False
    print("⚠️  XGBoost absent → pip install xgboost\n   RF uniquement.")

# ─── Features ───────────────────────────────────────────────
FEATURES = [
    "humidite_sol_moy_pct", "humidite_sol_min_pct", "humidite_sol_0_7_moy",
    "temp_max_C", "temp_min_C", "temp_moy_C", "humidite_air_moy_pct",
    "vent_u2_ms", "rayonnement_Rs_MJ", "ET0_reference_mm", "ETc_mm",
    "deficit_hydrique_mm", "pluie_totale_mm", "pluie_effective_mm",
    "jour_annee", "mois",
]
TARGET_C = "irriguer"
TARGET_R = "volume_litres"
KC_MOYEN = 1.05


# ══════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ══════════════════════════════════════════════════════════════
def charger():
    print("📂 Chargement du dataset 2022-2024...")
    df = pd.read_csv(CSV_CLEAN, parse_dates=["date"])

    # Recalcul des colonnes dérivées (compatibilité anciens CSV)
    df["ETc_mm"]              = (df["ET0_reference_mm"] * KC_MOYEN).round(2)
    df["pluie_effective_mm"]  = (df["pluie_totale_mm"]  * 0.80).round(2)
    df["deficit_hydrique_mm"] = (df["ETc_mm"] - df["pluie_effective_mm"]).round(2)
    if "annee" not in df.columns:
        df["annee"] = pd.to_datetime(df["date"]).dt.year
    if "mois" not in df.columns:
        df["mois"] = pd.to_datetime(df["date"]).dt.month
    if "jour_annee" not in df.columns:
        df["jour_annee"] = pd.to_datetime(df["date"]).dt.dayofyear
    if "saison" not in df.columns:
        df["saison"] = df["mois"].map(lambda m:
            "seche" if m in [11,12,1,2,3] else
            "grande_pluie" if m in [6,7,8,9] else "petite_pluie")

    manquantes = [f for f in FEATURES if f not in df.columns]
    if manquantes:
        print(f"❌ Colonnes manquantes : {manquantes}"); sys.exit(1)

    df.dropna(subset=FEATURES + [TARGET_C, TARGET_R], inplace=True)
    print(f"   ✅ {len(df)} observations | "
          f"{sorted(df['annee'].unique().tolist())}")
    return df


# ══════════════════════════════════════════════════════════════
# 2. CLASSIFICATION
# ══════════════════════════════════════════════════════════════
def entrainer_classification(df):
    print("\n" + "═"*58)
    print("  MODÈLE A — CLASSIFICATION : Irriguer OUI/NON")
    print("═"*58)

    X = df[FEATURES]
    y = df[TARGET_C].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"  Train : {len(X_tr)} | Test : {len(X_te)}")
    print(f"  NON : {(y==0).sum()} | OUI : {(y==1).sum()}")

    resultats = {}

    # ── Random Forest ──────────────────────────────────────
    print("\n  [1/2] Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    yp = rf.predict(X_te)
    f1 = f1_score(y_te, yp, average="weighted")
    ac = accuracy_score(y_te, yp)
    cv = cross_val_score(rf, X, y, cv=5, scoring="f1_weighted")
    print(f"     Accuracy : {ac:.4f} | F1 : {f1:.4f} | CV : {cv.mean():.4f}±{cv.std():.4f}")
    print(classification_report(y_te, yp, target_names=["NON","OUI"]))
    resultats["RF_Classifier"] = {"model":rf,"f1":f1,"accuracy":ac,"cv_mean":cv.mean(),"y_pred":yp}

    # ── XGBoost ────────────────────────────────────────────
    if XGBOOST_OK:
        print("\n  [2/2] XGBoost Classifier...")
        ratio = (y==0).sum()/(y==1).sum()
        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=ratio, random_state=42,
            eval_metric="logloss", verbosity=0)
        xgb.fit(X_tr, y_tr)
        yp2 = xgb.predict(X_te)
        f12 = f1_score(y_te, yp2, average="weighted")
        ac2 = accuracy_score(y_te, yp2)
        cv2 = cross_val_score(xgb, X, y, cv=5, scoring="f1_weighted")
        print(f"     Accuracy : {ac2:.4f} | F1 : {f12:.4f} | CV : {cv2.mean():.4f}±{cv2.std():.4f}")
        resultats["XGB_Classifier"] = {"model":xgb,"f1":f12,"accuracy":ac2,"cv_mean":cv2.mean(),"y_pred":yp2}

    best_nom = max(resultats, key=lambda k: resultats[k]["f1"])
    best     = resultats[best_nom]
    print(f"\n   Meilleur : {best_nom} | F1={best['f1']:.4f}")

    chemin = os.path.join(MODEL_DIR, "modele_classification.joblib")
    joblib.dump(best["model"], chemin)
    print(f"   Sauvegardé → {chemin}")

    sauver_importance(best["model"], FEATURES, "importance_classification.png",
                      "Classification — OUI/NON Irriguer")
    sauver_confusion(y_te, best["y_pred"], best_nom)
    return best["model"], X_te, y_te, resultats


# ══════════════════════════════════════════════════════════════
# 3. RÉGRESSION
# ══════════════════════════════════════════════════════════════
def entrainer_regression(df):
    print("\n" + "═"*58)
    print("  MODÈLE B — RÉGRESSION : Volume d'eau (litres)")
    print("═"*58)

    sub = df[df[TARGET_R] > 0].copy()
    print(f"  Observations (jours irrigués) : {len(sub)}")

    X = sub[FEATURES]; y = sub[TARGET_R]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

    resultats = {}

    # ── Random Forest ──────────────────────────────────────
    print("\n  [1/2] Random Forest Regressor...")
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=2,
        random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    yp  = rf.predict(X_te)
    mae = mean_absolute_error(y_te, yp)
    r2  = r2_score(y_te, yp)
    rmse= np.sqrt(mean_squared_error(y_te, yp))
    print(f"     MAE={mae:.1f}L | RMSE={rmse:.1f}L | R²={r2:.4f}")
    resultats["RF_Regressor"] = {"model":rf,"mae":mae,"r2":r2,"y_pred":yp}

    # ── XGBoost ────────────────────────────────────────────
    if XGBOOST_OK:
        print("\n  [2/2] XGBoost Regressor...")
        xgb = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=42, verbosity=0)
        xgb.fit(X_tr, y_tr)
        yp2  = xgb.predict(X_te)
        mae2 = mean_absolute_error(y_te, yp2)
        r22  = r2_score(y_te, yp2)
        rmse2= np.sqrt(mean_squared_error(y_te, yp2))
        print(f"     MAE={mae2:.1f}L | RMSE={rmse2:.1f}L | R²={r22:.4f}")
        resultats["XGB_Regressor"] = {"model":xgb,"mae":mae2,"r2":r22,"y_pred":yp2}

    best_nom = max(resultats, key=lambda k: resultats[k]["r2"])
    best     = resultats[best_nom]
    print(f"\n   Meilleur : {best_nom} | R²={best['r2']:.4f} | MAE={best['mae']:.1f}L")

    chemin = os.path.join(MODEL_DIR, "modele_regression.joblib")
    joblib.dump(best["model"], chemin)
    print(f"   Sauvegardé → {chemin}")

    sauver_importance(best["model"], FEATURES, "importance_regression.png",
                      "Régression — Volume (litres)")
    sauver_pred_reel(y_te, best["y_pred"], best_nom)
    return best["model"], resultats


# ══════════════════════════════════════════════════════════════
# 4. VISUALISATIONS
# ══════════════════════════════════════════════════════════════
def sauver_importance(model, features, nom, titre):
    imp  = model.feature_importances_
    idx  = np.argsort(imp)
    cols = [features[i] for i in idx]
    vals = imp[idx]
    colors = ["#2E86AB" if v >= 0.05 else "#ADB5BD" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(cols, vals, color=colors, edgecolor="white")
    ax.axvline(0.05, color="red", linestyle="--", alpha=0.6, label="Seuil 5%")
    ax.set_title(f"Importance des variables — {titre}", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, nom), dpi=150); plt.close()
    print(f"   {nom}")


def sauver_confusion(y_true, y_pred, nom):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im  = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Prédit NON","Prédit OUI"])
    ax.set_yticklabels(["Réel NON","Réel OUI"])
    labels = [["VN","FP"],["FN","VP"]]
    for i in range(2):
        for j in range(2):
            c = "white" if cm[i,j] > cm.max()/2 else "black"
            ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]}", ha="center",
                    va="center", fontsize=14, fontweight="bold", color=c)
    ax.set_title(f"Confusion — {nom}", fontweight="bold")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "matrice_confusion.png"), dpi=150)
    plt.close(); print("   matrice_confusion.png")


def sauver_pred_reel(y_true, y_pred, nom):
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, color="#2E86AB", s=25)
    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0,lim],[0,lim],"r--", label="Parfait")
    ax.text(0.05, 0.92, f"R²={r2:.4f}\nMAE={mae:.0f}L",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(facecolor="white", alpha=0.85))
    ax.set_xlabel("Réel (L)"); ax.set_ylabel("Prédit (L)")
    ax.set_title(f"Prédit vs Réel — {nom}", fontweight="bold")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pred_vs_reel_regression.png"), dpi=150)
    plt.close(); print("   pred_vs_reel_regression.png")


# ══════════════════════════════════════════════════════════════
# 5. BILAN FINAL
# ══════════════════════════════════════════════════════════════
def bilan(res_c, res_r):
    print("\n" + "═"*58)
    print("  BILAN FINAL")
    print("═"*58)
    print("\n  Classification :")
    for n, r in res_c.items():
        print(f"    {n:<25} F1={r['f1']:.4f}  Acc={r['accuracy']:.4f}  CV={r['cv_mean']:.4f}")
    print("\n  Régression :")
    for n, r in res_r.items():
        print(f"    {n:<25} R²={r['r2']:.4f}  MAE={r['mae']:.1f}L")

    with open(os.path.join(OUT_DIR, "rapport_entrainement.txt"), "w") as f:
        f.write("RAPPORT ENTRAÎNEMENT — IRRIGATION CI (2022-2024)\n")
        f.write("="*50 + "\nCLASSIFICATION :\n")
        for n, r in res_c.items():
            f.write(f"  {n}: F1={r['f1']:.4f} Acc={r['accuracy']:.4f} CV={r['cv_mean']:.4f}\n")
        f.write("\nRÉGRESSION :\n")
        for n, r in res_r.items():
            f.write(f"  {n}: R²={r['r2']:.4f} MAE={r['mae']:.1f}L\n")

    print("\n  📄 rapport_entrainement.txt")
    print("═"*58)
    print("   Entraînement terminé → lancez : 03_prediction.py")
    print("═"*58)


# ══════════════════════════════════════════════════════════════
# EXÉCUTION
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = charger()
    clf, X_te, y_te, res_c = entrainer_classification(df)
    reg, res_r              = entrainer_regression(df)
    bilan(res_c, res_r)