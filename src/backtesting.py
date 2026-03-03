"""
=============================================================
 SYSTÈME D'IRRIGATION INTELLIGENTE — CÔTE D'IVOIRE
 Étape 4 : Backtesting sur données historiques 2022-2024
=============================================================
 Simule le système sur n'importe quelle date du dataset.
 Compare la prédiction ML à la réalité connue.

 Usage :
   python src/04_backtesting.py                         ← 23/04/2024 (défaut)
   python src/04_backtesting.py 2022-07-15              ← date choisie
   python src/04_backtesting.py 2023-11-01              ← saison sèche
   python src/04_backtesting.py 2024-04-23 full         ← rapport annuel
   python src/04_backtesting.py 2022-01-01 full         ← rapport 2022
=============================================================
"""

import os, sys, joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─── Chemins ────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")

CSV_CLEAN = os.path.join(DATA_DIR, "yamoussoukro_dataset_ML.csv")
CLF_PATH  = os.path.join(MODEL_DIR, "modele_classification.joblib")
REG_PATH  = os.path.join(MODEL_DIR, "modele_regression.joblib")

KC_MOYEN   = 1.05
SURFACE_M2 = 200
EFFICACITE = 0.90

FEATURES = [
    "humidite_sol_moy_pct", "humidite_sol_min_pct", "humidite_sol_0_7_moy",
    "temp_max_C", "temp_min_C", "temp_moy_C", "humidite_air_moy_pct",
    "vent_u2_ms", "rayonnement_Rs_MJ", "ET0_reference_mm", "ETc_mm",
    "deficit_hydrique_mm", "pluie_totale_mm", "pluie_effective_mm",
    "jour_annee", "mois",
]


# ══════════════════════════════════════════════════════════════
# 1. RÈGLES + DÉCISION
# ══════════════════════════════════════════════════════════════
def appliquer_regles(hs, pluie, deficit):
    if hs > 70:
        return 0, f"Sol humide ({hs:.1f}%)", True
    if pluie > 10:
        return 0, f"Forte pluie ({pluie:.1f}mm)", True
    if pluie > 5 and hs > 50:
        return 0, "Pluie modérée + sol ok", True
    if deficit <= 0:
        return 0, f"Pas de déficit ({deficit:.2f}mm)", True
    return 1, "Délégué au ML", False


def predire_jour(clf, reg, row: pd.Series) -> dict:
    row = row.copy()
    row["ETc_mm"]              = round(float(row["ET0_reference_mm"]) * KC_MOYEN, 2)
    row["pluie_effective_mm"]  = round(float(row["pluie_totale_mm"])  * 0.80, 2)
    row["deficit_hydrique_mm"] = round(row["ETc_mm"] - row["pluie_effective_mm"], 2)

    hs, pluie, deficit = (float(row["humidite_sol_moy_pct"]),
                          float(row["pluie_totale_mm"]),
                          float(row["deficit_hydrique_mm"]))

    dec_r, raison, court = appliquer_regles(hs, pluie, deficit)
    if court:
        return {"irriguer_predit": dec_r, "volume_predit_L": 0.0,
                "raison": raison, "source": "Règle agronomique",
                "confiance": 100.0, "ETc_mm": row["ETc_mm"],
                "deficit_mm": deficit}

    X         = pd.DataFrame([row])[FEATURES]
    proba     = clf.predict_proba(X)[0]
    decision  = int(clf.predict(X)[0])
    confiance = round(proba[decision] * 100, 1)

    volume = 0.0
    if decision == 1:
        vol_ml  = float(reg.predict(X)[0])
        facteur = max(0.0, (65.0 - hs) / 25.0)
        vol_fo  = deficit * facteur * SURFACE_M2 / EFFICACITE
        volume  = round(max(0.60 * vol_ml + 0.40 * vol_fo, 0), 1)

    return {"irriguer_predit": decision, "volume_predit_L": volume,
            "raison": f"ML confiance {confiance}%",
            "source": "Modèle ML", "confiance": confiance,
            "ETc_mm": row["ETc_mm"], "deficit_mm": deficit}


# ══════════════════════════════════════════════════════════════
# 2. BACKTESTING SUR UNE DATE + 3 JOURS
# ══════════════════════════════════════════════════════════════
def recuperer_jours(date_str: str, df: pd.DataFrame) -> list:
    df["date_dt"] = pd.to_datetime(df["date"]).dt.date
    dates         = df["date_dt"].tolist()
    date_cible    = pd.to_datetime(date_str).date()

    if date_cible not in dates:
        debut = str(min(dates)); fin = str(max(dates))
        print(f"❌ Date {date_str} non trouvée.")
        print(f"   Période disponible : {debut} → {fin}")
        sys.exit(1)

    jours = []
    for i in range(4):
        d = date_cible + timedelta(days=i)
        if d in dates:
            jours.append(df[df["date_dt"] == d].iloc[0])
    return jours


# ══════════════════════════════════════════════════════════════
# 3. AFFICHAGE DES RÉSULTATS
# ══════════════════════════════════════════════════════════════
def afficher_resultats(jours: list, clf, reg, date_str: str):
    print("\n" + "═"*64)
    print(f"  🌱 BACKTESTING — Simulation à partir du {date_str}")
    print(f"  📍 Yamoussoukro, CI | 🍅 Tomate | Dataset : 2022-2024")
    print("═"*64)

    labels     = ["AUJOURD'HUI (J)  ", "DEMAIN    (J+1)  ",
                  "APRÈS-DEM (J+2)  ", "J+3              "]
    preds, reels = [], []

    for i, row in enumerate(jours):
        lbl        = labels[i] if i < len(labels) else f"J+{i}            "
        date_aff   = str(row["date"])[:10]
        res        = predire_jour(clf, reg, row)
        vrai_irr   = int(row["irriguer"])
        vrai_vol   = float(row["volume_litres"])
        correct    = (res["irriguer_predit"] == vrai_irr)
        ip         = "✅" if res["irriguer_predit"] else "❌"
        ir         = "✅" if vrai_irr else "❌"
        iv         = "🎯" if correct else "⚠️ "

        print(f"\n  ┌{'─'*60}┐")
        print(f"  │  {lbl} — {date_aff}  (saison : {row['saison']}){'':>12}│")
        print(f"  ├{'─'*60}┤")
        print(f"  │  📊 DONNÉES RÉELLES DU JOUR :{'':>31}│")
        print(f"  │     Humidité sol   : {row['humidite_sol_moy_pct']:.1f}%{'':>39}│")
        print(f"  │     Température    : {row['temp_max_C']:.1f}°C max / {row['temp_min_C']:.1f}°C min{'':>24}│")
        print(f"  │     Pluie          : {row['pluie_totale_mm']:.1f} mm{'':>38}│")
        print(f"  │     ET₀ référence  : {row['ET0_reference_mm']:.2f} mm/j{'':>36}│")
        print(f"  ├{'─'*60}┤")
        print(f"  │  🤖 PRÉDICTION MODÈLE ML :{'':>34}│")
        dec_txt = "ARROSER        " if res["irriguer_predit"] else "NE PAS ARROSER "
        print(f"  │     {ip} {dec_txt}  ({res['source']}){'':>14}│")
        if res["irriguer_predit"]:
            print(f"  │        Volume prédit   : {res['volume_predit_L']:.0f} L{'':>33}│")
        print(f"  │        ETc             : {res['ETc_mm']:.2f} mm/j{'':>31}│")
        print(f"  │        Déficit         : {res['deficit_mm']:.2f} mm{'':>32}│")
        print(f"  │        Raison          : {res['raison'][:35]:<35}{'':>1}│")
        print(f"  ├{'─'*60}┤")
        print(f"  │  📋 RÉALITÉ (dataset 2022-2024) :{'':>27}│")
        reel_txt = "ARROSER        " if vrai_irr else "NE PAS ARROSER "
        print(f"  │     {ir} {reel_txt}{'':>40}│")
        if vrai_irr:
            print(f"  │        Volume réel     : {vrai_vol:.0f} L{'':>34}│")
        print(f"  ├{'─'*60}┤")
        verdict = "CORRECT  ✅" if correct else "INCORRECT ⚠️"
        print(f"  │  {iv} VERDICT : {verdict:<50}│")
        if correct and res["irriguer_predit"] and vrai_vol > 0:
            ecart     = abs(res["volume_predit_L"] - vrai_vol)
            ecart_pct = ecart / vrai_vol * 100
            print(f"  │     Écart volume  : {ecart:.0f} L ({ecart_pct:.1f}%){'':>32}│")
        print(f"  └{'─'*60}┘")

        preds.append(res["irriguer_predit"])
        reels.append(vrai_irr)

    n_ok  = sum(p == r for p, r in zip(preds, reels))
    score = n_ok / len(preds) * 100
    print(f"\n  {'═'*60}")
    print(f"  📈 SCORE : {n_ok}/{len(preds)} correct ({score:.0f}%)")
    print(f"  {'═'*60}")
    return preds, reels


# ══════════════════════════════════════════════════════════════
# 4. RAPPORT COMPLET PAR ANNÉE (mode full)
# ══════════════════════════════════════════════════════════════
def rapport_complet(df: pd.DataFrame, clf, reg, annee: int = None):
    """
    Teste le modèle sur tous les jours du dataset.
    Si annee est fourni, filtre sur cette année uniquement.
    """
    if annee:
        df = df[df["annee"] == annee].copy()
        titre = f"Backtesting année {annee}"
    else:
        titre = "Backtesting 2022-2024 (complet)"

    print("\n" + "═"*64)
    print(f"  📊 RAPPORT COMPLET — {titre}")
    print("═"*64)

    preds, reels, ecarts_vol = [], [], []

    for _, row in df.iterrows():
        res      = predire_jour(clf, reg, row)
        vrai_irr = int(row["irriguer"])
        vrai_vol = float(row["volume_litres"])
        preds.append(res["irriguer_predit"])
        reels.append(vrai_irr)
        if res["irriguer_predit"] == 1 and vrai_irr == 1:
            ecarts_vol.append(abs(res["volume_predit_L"] - vrai_vol))

    preds = np.array(preds); reels = np.array(reels)
    acc   = (preds == reels).mean() * 100
    VP    = ((preds==1)&(reels==1)).sum()
    VN    = ((preds==0)&(reels==0)).sum()
    FP    = ((preds==1)&(reels==0)).sum()
    FN    = ((preds==0)&(reels==1)).sum()
    prec  = VP/(VP+FP)*100 if (VP+FP)>0 else 0
    rec   = VP/(VP+FN)*100 if (VP+FN)>0 else 0
    f1    = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0

    print(f"\n  Période testée : {df['date'].min()} → {df['date'].max()}")
    print(f"  Observations  : {len(df)} jours\n")
    print(f"  CLASSIFICATION :")
    print(f"  {'─'*45}")
    print(f"  Accuracy   : {acc:.2f}%")
    print(f"  Précision  : {prec:.2f}%")
    print(f"  Rappel     : {rec:.2f}%")
    print(f"  F1-score   : {f1:.2f}%")
    print(f"\n  Matrice de confusion :")
    print(f"               Prédit NON   Prédit OUI")
    print(f"  Réel NON  :      {VN:>4}         {FP:>4}")
    print(f"  Réel OUI  :      {FN:>4}         {VP:>4}")

    if ecarts_vol:
        print(f"\n  RÉGRESSION (volume en litres) :")
        print(f"  {'─'*45}")
        print(f"  MAE    : {np.mean(ecarts_vol):.0f} L")
        print(f"  Médiane: {np.median(ecarts_vol):.0f} L")
        print(f"  Max    : {np.max(ecarts_vol):.0f} L")

    print(f"\n  🔴 Oublis d'irrigation (FN) : {FN} jours")
    print(f"  ⚠️  Irrigations inutiles (FP) : {FP} jours")

    # Rapport par saison
    print(f"\n  PAR SAISON :")
    print(f"  {'─'*50}")
    df_r = df.copy()
    df_r["predit"] = preds
    df_r["reel"]   = reels
    for sais, grp in df_r.groupby("saison"):
        acc_s = (grp["predit"] == grp["reel"]).mean() * 100
        print(f"  {sais:<20} : {len(grp):>4} j | "
              f"irrig={( grp.reel==1).sum():>3} | acc={acc_s:.1f}%")

    print("═"*64)


# ══════════════════════════════════════════════════════════════
# EXÉCUTION
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    date_str = sys.argv[1] if len(sys.argv) > 1 else "2024-04-23"
    mode     = sys.argv[2] if len(sys.argv) > 2 else "normal"

    print("📂 Chargement...")
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    df  = pd.read_csv(CSV_CLEAN, parse_dates=["date"])
    # Recalcul colonnes dérivées si absentes (compatibilité anciens CSV)
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
    print(f"   ✅ Dataset 2022-2024 : {len(df)} jours")

    jours = recuperer_jours(date_str, df, )
    afficher_resultats(jours, clf, reg, date_str)

    if mode == "full":
        annee = pd.to_datetime(date_str).year
        rapport_complet(df, clf, reg, annee=annee)
        # rapport sur les 3 ans complets
        rapport_complet(df, clf, reg, annee=None)

    print(f"\n💡 Autres tests possibles :")
    print(f"   python src/04_backtesting.py 2022-07-15")
    print(f"   python src/04_backtesting.py 2023-01-10")
    print(f"   python src/04_backtesting.py 2024-09-05 full")