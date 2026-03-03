"""
=============================================================
 SYSTÈME D'IRRIGATION INTELLIGENTE — CÔTE D'IVOIRE
 Étape 5 : Évaluation complète des métriques ML
=============================================================
 Dataset : 2022-2024 (1096 jours)
 Produit :
   - Métriques classification et régression
   - Analyse par année et par saison CI
   - Analyse des erreurs critiques
   - Courbes ROC / Précision-Rappel
   - Graphique complet → outputs/evaluation_complete.png
=============================================================
"""

import os, sys, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
warnings.filterwarnings("ignore")

# ─── Chemins ────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_CLEAN = os.path.join(DATA_DIR, "yamoussoukro_dataset_ML.csv")
CLF_PATH  = os.path.join(MODEL_DIR, "modele_classification.joblib")
REG_PATH  = os.path.join(MODEL_DIR, "modele_regression.joblib")

KC_MOYEN   = 1.05
SURFACE_M2 = 200
EFFICACITE = 0.90
COULEURS   = {"bleu":"#1D3557","orange":"#E76F51","vert":"#2D6A4F",
              "clair":"#A8DADC","gris":"#ADB5BD","jaune":"#F4A261"}

FEATURES = [
    "humidite_sol_moy_pct", "humidite_sol_min_pct", "humidite_sol_0_7_moy",
    "temp_max_C", "temp_min_C", "temp_moy_C", "humidite_air_moy_pct",
    "vent_u2_ms", "rayonnement_Rs_MJ", "ET0_reference_mm", "ETc_mm",
    "deficit_hydrique_mm", "pluie_totale_mm", "pluie_effective_mm",
    "jour_annee", "mois",
]


# ══════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ══════════════════════════════════════════════════════════════
def charger():
    print(" Chargement données et modèles (2022-2024)...")
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    df  = pd.read_csv(CSV_CLEAN)

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

    X     = df[FEATURES]
    y_clf = df["irriguer"].astype(int)
    X_irr = df[df["volume_litres"] > 0][FEATURES]
    y_irr = df[df["volume_litres"] > 0]["volume_litres"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_clf, test_size=0.20, random_state=42, stratify=y_clf)
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_irr, y_irr, test_size=0.20, random_state=42)

    annees = sorted(df["annee"].unique().tolist())
    print(f"    {len(df)} jours | années : {annees}")
    print(f"    Test clf : {len(X_te)} obs | Test reg : {len(X_te_r)} obs\n")
    return clf, reg, df, X, y_clf, X_te, y_te, X_irr, X_te_r, y_te_r


# ══════════════════════════════════════════════════════════════
# 2. CLASSIFICATION
# ══════════════════════════════════════════════════════════════
def evaluer_classification(clf, X, y_clf, X_te, y_te):
    print("═"*60)
    print("  ÉVALUATION — CLASSIFICATION (OUI/NON Irriguer)")
    print("═"*60)

    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]
    acc     = accuracy_score(y_te, y_pred)
    prec    = precision_score(y_te, y_pred, zero_division=0)
    rec     = recall_score(y_te, y_pred, zero_division=0)
    f1      = f1_score(y_te, y_pred, average="weighted")
    f1b     = f1_score(y_te, y_pred, average="binary")
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    roc_auc = auc(fpr, tpr)

    print(f"\n  Métriques sur le jeu de TEST ({len(y_te)} obs) :")
    print(f"  {'─'*45}")
    print(f"  Accuracy            : {acc*100:.2f}%")
    print(f"  Précision           : {prec*100:.2f}%")
    print(f"  Rappel              : {rec*100:.2f}%")
    print(f"  F1-score (binaire)  : {f1b*100:.2f}%")
    print(f"  F1-score (pondéré)  : {f1*100:.2f}%")
    print(f"  AUC-ROC             : {roc_auc:.4f}")

    print(f"\n  Validation croisée 5-fold (1096 jours) :")
    print(f"  {'─'*45}")
    for m in ["accuracy","f1_weighted","precision","recall"]:
        sc = cross_val_score(clf, X, y_clf, cv=5, scoring=m)
        print(f"  {m:<20}: {sc.mean()*100:.2f}% ± {sc.std()*100:.2f}%")

    cm = confusion_matrix(y_te, y_pred)
    VP,VN,FP,FN = cm[1,1],cm[0,0],cm[0,1],cm[1,0]
    print(f"\n  Matrice de confusion :")
    print(f"                  Prédit NON   Prédit OUI")
    print(f"  Réel NON   :       {VN:>4}         {FP:>4}")
    print(f"  Réel OUI   :       {FN:>4}         {VP:>4}")
    print(f"\n   VP (irrig. prédit et réel)     : {VP}")
    print(f"   VN (pas irrig. prédit et réel) : {VN}")
    print(f"    FP (gaspillage eau)             : {FP}")
    print(f"   FN (oubli irrigation)            : {FN}")
    print(f"\n{classification_report(y_te, y_pred, target_names=['NON','OUI'])}")
    return y_pred, y_proba, cm, acc, prec, rec, f1, roc_auc, fpr, tpr


# ══════════════════════════════════════════════════════════════
# 3. RÉGRESSION
# ══════════════════════════════════════════════════════════════
def evaluer_regression(reg, X_te_r, y_te_r):
    print("\n" + "═"*60)
    print("  ÉVALUATION — RÉGRESSION (Volume en litres)")
    print("═"*60)

    y_pred_r = reg.predict(X_te_r)
    mae      = mean_absolute_error(y_te_r, y_pred_r)
    rmse     = np.sqrt(mean_squared_error(y_te_r, y_pred_r))
    r2       = r2_score(y_te_r, y_pred_r)
    mape     = np.mean(np.abs((y_te_r - y_pred_r) / y_te_r.clip(lower=1))) * 100
    erreurs  = np.abs(y_te_r.values - y_pred_r)

    print(f"\n  Métriques sur le jeu de TEST ({len(y_te_r)} obs) :")
    print(f"  {'─'*45}")
    print(f"  MAE  (Erreur absolue moyenne)   : {mae:.1f} L")
    print(f"  RMSE (Racine erreur quadratique): {rmse:.1f} L")
    print(f"  R²   (Variance expliquée)       : {r2:.4f} ({r2*100:.2f}%)")
    print(f"  MAPE (Erreur % moyenne)         : {mape:.1f}%")
    print(f"\n  Distribution des erreurs absolues :")
    for seuil in [50, 100, 150, 200]:
        pct = (erreurs <= seuil).mean() * 100
        print(f"  Erreur ≤ {seuil:>3}L : {pct:.1f}%")
    print(f"  Erreur médiane : {np.median(erreurs):.0f}L | Max : {erreurs.max():.0f}L")

    interp = ("🟢 Excellent" if r2>=0.95 else "🟡 Bon" if r2>=0.85 else "🔴 À améliorer")
    print(f"\n  Interprétation R² : {interp} "
          f"(modèle explique {r2*100:.1f}% de la variance)")
    return y_pred_r, mae, rmse, r2, mape


# ══════════════════════════════════════════════════════════════
# 4. ANALYSE PAR ANNÉE ET PAR SAISON
# ══════════════════════════════════════════════════════════════
def analyse_annee_saison(clf, df):
    print("\n" + "═"*60)
    print("  ANALYSE PAR ANNÉE — 2022 / 2023 / 2024")
    print("═"*60)

    df2 = df.copy()
    df2["ETc_mm"]              = df2["ET0_reference_mm"] * KC_MOYEN
    df2["pluie_effective_mm"]  = df2["pluie_totale_mm"] * 0.80
    df2["deficit_hydrique_mm"] = df2["ETc_mm"] - df2["pluie_effective_mm"]

    print(f"\n  {'Année':<8} {'Jours':>6} {'Irrig%':>7} {'Acc%':>7} "
          f"{'Vol moy':>9} {'Pluie/an':>10}")
    print(f"  {'─'*52}")
    for yr, grp in df2.groupby("annee"):
        X_y  = grp[FEATURES]
        y_y  = grp["irriguer"].astype(int)
        y_p  = clf.predict(X_y)
        acc  = accuracy_score(y_y, y_p) * 100
        irr  = y_y.mean() * 100
        vol  = grp[grp["volume_litres"]>0]["volume_litres"].mean()
        pluie= grp["pluie_totale_mm"].sum()
        print(f"  {yr:<8} {len(grp):>6} {irr:>6.1f}% {acc:>6.1f}% "
              f"{vol:>8.0f}L {pluie:>9.0f}mm")

    print(f"\n  ANALYSE PAR SAISON CI :")
    print(f"  {'─'*52}")
    print(f"  {'Saison':<22} {'Jours':>6} {'Irrig%':>7} {'Acc%':>7} {'Vol moy':>9}")
    print(f"  {'─'*52}")
    noms = {"seche":"Sèche (nov–mars)",
            "grande_pluie":"Grande pluie (juin–sept)",
            "petite_pluie":"Petite pluie (avr–mai,oct)"}
    for code, label in noms.items():
        grp = df2[df2["saison"]==code]
        if len(grp)==0: continue
        X_s = grp[FEATURES]; y_s = grp["irriguer"].astype(int)
        y_p = clf.predict(X_s)
        acc = accuracy_score(y_s, y_p)*100
        irr = y_s.mean()*100
        vol = grp[grp["volume_litres"]>0]["volume_litres"].mean()
        print(f"  {label:<22} {len(grp):>6} {irr:>6.1f}% {acc:>6.1f}% {vol:>8.0f}L")


# ══════════════════════════════════════════════════════════════
# 5. ERREURS CRITIQUES
# ══════════════════════════════════════════════════════════════
def analyser_erreurs(clf, df):
    print("\n" + "═"*60)
    print("  ANALYSE DES ERREURS CRITIQUES")
    print("═"*60)

    df2 = df.copy()
    df2["ETc_mm"]              = df2["ET0_reference_mm"] * KC_MOYEN
    df2["pluie_effective_mm"]  = df2["pluie_totale_mm"] * 0.80
    df2["deficit_hydrique_mm"] = df2["ETc_mm"] - df2["pluie_effective_mm"]

    y_real = df2["irriguer"].astype(int)
    y_pred = clf.predict(df2[FEATURES])

    FN_mask = (y_pred==0) & (y_real==1)
    FP_mask = (y_pred==1) & (y_real==0)
    faux_neg = df2[FN_mask]
    faux_pos = df2[FP_mask]

    print(f"\n   FAUX NÉGATIFS (oubli irrigation) : {len(faux_neg)} jours")
    if len(faux_neg):
        print(f"  {'Date':<12} {'Année':>5} {'Sol%':>6} {'Pluie':>7} "
              f"{'ET₀':>6} {'Déficit':>9} {'Saison'}")
        print(f"  {'─'*60}")
        for _, r in faux_neg.iterrows():
            print(f"  {str(r['date'])[:10]:<12} {int(r['annee']):>5} "
                  f"{r['humidite_sol_moy_pct']:>5.1f}% "
                  f"{r['pluie_totale_mm']:>6.1f}mm "
                  f"{r['ET0_reference_mm']:>5.2f} "
                  f"{r['deficit_hydrique_mm']:>8.2f}mm  {r['saison']}")

    print(f"\n    FAUX POSITIFS (irrigation inutile) : {len(faux_pos)} jours")
    if len(faux_pos):
        print(f"  {'Date':<12} {'Année':>5} {'Sol%':>6} {'Pluie':>7} "
              f"{'ET₀':>6} {'Déficit':>9} {'Saison'}")
        print(f"  {'─'*60}")
        for _, r in faux_pos.iterrows():
            print(f"  {str(r['date'])[:10]:<12} {int(r['annee']):>5} "
                  f"{r['humidite_sol_moy_pct']:>5.1f}% "
                  f"{r['pluie_totale_mm']:>6.1f}mm "
                  f"{r['ET0_reference_mm']:>5.2f} "
                  f"{r['deficit_hydrique_mm']:>8.2f}mm  {r['saison']}")
    else:
        print("  Aucun gaspillage d'eau détecté !")

    return faux_neg, faux_pos


# ══════════════════════════════════════════════════════════════
# 6. GRAPHIQUES COMPLETS
# ══════════════════════════════════════════════════════════════
def generer_graphiques(clf, reg, df, y_te, y_pred_clf,
                        y_proba, y_te_r, y_pred_r, cm,
                        fpr, tpr, roc_auc):
    print("\n" + "═"*60)
    print("  GÉNÉRATION DES GRAPHIQUES")
    print("═"*60)

    fig = plt.figure(figsize=(22, 26))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
    C   = COULEURS

    # ── 1. Matrice de confusion ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    im  = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
    ax1.set_xticklabels(["Prédit NON","Prédit OUI"], fontsize=10)
    ax1.set_yticklabels(["Réel NON","Réel OUI"],    fontsize=10)
    labs = [["VN","FP"],["FN","VP"]]
    for i in range(2):
        for j in range(2):
            col = "white" if cm[i,j]>cm.max()/2 else "black"
            ax1.text(j,i,f"{labs[i][j]}\n{cm[i,j]}",ha="center",
                     va="center",fontsize=14,fontweight="bold",color=col)
    ax1.set_title("Matrice de confusion", fontweight="bold", fontsize=12)
    plt.colorbar(im, ax=ax1, shrink=0.8)

    # ── 2. Courbe ROC ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fpr, tpr, color=C["bleu"], lw=2.5, label=f"AUC = {roc_auc:.4f}")
    ax2.plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Aléatoire")
    ax2.fill_between(fpr, tpr, alpha=0.15, color=C["bleu"])
    ax2.set_xlabel("Taux Faux Positifs", fontsize=10)
    ax2.set_ylabel("Taux Vrais Positifs", fontsize=10)
    ax2.set_title("Courbe ROC", fontweight="bold", fontsize=12)
    ax2.legend(fontsize=10); ax2.grid(alpha=0.3)

    # ── 3. Courbe Précision-Rappel ────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    prec_c, rec_c, _ = precision_recall_curve(y_te, y_proba)
    pr_auc = auc(rec_c, prec_c)
    ax3.plot(rec_c, prec_c, color=C["orange"], lw=2.5,
             label=f"AUC-PR = {pr_auc:.4f}")
    ax3.fill_between(rec_c, prec_c, alpha=0.15, color=C["orange"])
    ax3.set_xlabel("Rappel", fontsize=10); ax3.set_ylabel("Précision", fontsize=10)
    ax3.set_title("Courbe Précision–Rappel", fontweight="bold", fontsize=12)
    ax3.legend(fontsize=10); ax3.grid(alpha=0.3)

    # ── 4. Prédit vs Réel (régression) ───────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(y_te_r, y_pred_r, alpha=0.5, color=C["bleu"], s=25)
    lim = max(y_te_r.max(), y_pred_r.max()) * 1.05
    ax4.plot([0,lim],[0,lim],"r--",lw=1.5,label="Parfait")
    r2  = r2_score(y_te_r, y_pred_r)
    mae = mean_absolute_error(y_te_r, y_pred_r)
    ax4.text(0.05,0.92,f"R²={r2:.4f}\nMAE={mae:.0f}L",
             transform=ax4.transAxes, fontsize=10,
             bbox=dict(facecolor="white", alpha=0.85, edgecolor=C["bleu"]))
    ax4.set_xlabel("Réel (L)"); ax4.set_ylabel("Prédit (L)")
    ax4.set_title("Prédit vs Réel — Régression", fontweight="bold", fontsize=12)
    ax4.legend(fontsize=10); ax4.grid(alpha=0.3)

    # ── 5. Distribution erreurs régression ────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    erreurs = y_pred_r - y_te_r.values
    ax5.hist(erreurs, bins=30, color=C["clair"],
             edgecolor=C["bleu"], lw=0.8)
    ax5.axvline(0, color="red", lw=2, linestyle="--", label="Erreur=0")
    ax5.axvline(erreurs.mean(), color=C["orange"], lw=2, linestyle="-.",
                label=f"Moy={erreurs.mean():.0f}L")
    ax5.set_xlabel("Erreur (prédit−réel) en L", fontsize=10)
    ax5.set_ylabel("Fréquence", fontsize=10)
    ax5.set_title("Distribution des erreurs", fontweight="bold", fontsize=12)
    ax5.legend(fontsize=10); ax5.grid(alpha=0.3)

    # ── 6. Importance features (classification) ───────────
    ax6 = fig.add_subplot(gs[1, 2])
    imp = clf.feature_importances_; idx = np.argsort(imp)
    cols_f  = [FEATURES[i] for i in idx]
    vals_f  = imp[idx]
    colors6 = [C["vert"] if v>=0.05 else C["gris"] for v in vals_f]
    ax6.barh(cols_f, vals_f, color=colors6, edgecolor="white")
    ax6.axvline(0.05,color="red",lw=1.5,linestyle="--",alpha=0.7,label="Seuil 5%")
    ax6.set_title("Importance variables\n(Classification)",
                  fontweight="bold",fontsize=12)
    ax6.legend(fontsize=9); ax6.grid(axis="x",alpha=0.3)

    # ── 7. Importance features (régression) ───────────────
    ax7 = fig.add_subplot(gs[2, 0])
    imp2 = reg.feature_importances_; idx2 = np.argsort(imp2)
    cols7   = [FEATURES[i] for i in idx2]
    vals7   = imp2[idx2]
    colors7 = [C["orange"] if v>=0.05 else C["gris"] for v in vals7]
    ax7.barh(cols7, vals7, color=colors7, edgecolor="white")
    ax7.axvline(0.05,color="red",lw=1.5,linestyle="--",alpha=0.7,label="Seuil 5%")
    ax7.set_title("Importance variables\n(Régression)",
                  fontweight="bold",fontsize=12)
    ax7.legend(fontsize=9); ax7.grid(axis="x",alpha=0.3)

    # ── 8. Volume mensuel moyen (3 ans) ───────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    df["mois_num"] = pd.to_datetime(df["date"]).dt.month
    vol_mois = df.groupby("mois_num")["volume_litres"].mean()
    mois_noms = ["Jan","Fév","Mar","Avr","Mai","Jun",
                 "Jul","Aoû","Sep","Oct","Nov","Déc"]
    bar_c = [C["orange"] if v>vol_mois.mean() else C["clair"] for v in vol_mois]
    ax8.bar(range(1,13), vol_mois.values, color=bar_c, edgecolor="white")
    ax8.axhline(vol_mois.mean(), color="red", lw=1.5, linestyle="--",
                label=f"Moy. {vol_mois.mean():.0f}L")
    ax8.set_xticks(range(1,13)); ax8.set_xticklabels(mois_noms,fontsize=9)
    ax8.set_ylabel("Volume moyen (L)", fontsize=10)
    ax8.set_title("Volume irrigation mensuel\n(moy. 2022-2024)",
                  fontweight="bold",fontsize=12)
    ax8.legend(fontsize=10); ax8.grid(axis="y",alpha=0.3)

    # ── 9. Accuracy par année ─────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    annees_list = sorted(df["annee"].unique())
    accs_yr = []
    for yr in annees_list:
        sub_yr = df[df["annee"]==yr]
        sub_yr = sub_yr.copy()
        sub_yr["ETc_mm"]              = sub_yr["ET0_reference_mm"] * KC_MOYEN
        sub_yr["pluie_effective_mm"]  = sub_yr["pluie_totale_mm"] * 0.80
        sub_yr["deficit_hydrique_mm"] = sub_yr["ETc_mm"] - sub_yr["pluie_effective_mm"]
        yp_yr  = clf.predict(sub_yr[FEATURES])
        accs_yr.append(accuracy_score(sub_yr["irriguer"].astype(int), yp_yr)*100)
    bars9 = ax9.bar([str(y) for y in annees_list], accs_yr,
                    color=[C["vert"],C["bleu"],C["orange"]], edgecolor="white")
    for bar, val in zip(bars9, accs_yr):
        ax9.text(bar.get_x()+bar.get_width()/2, val-1.5,
                 f"{val:.1f}%", ha="center", va="top",
                 fontsize=12, fontweight="bold", color="white")
    ax9.set_ylim(90, 101)
    ax9.set_ylabel("Accuracy (%)", fontsize=10)
    ax9.set_title("Accuracy par année",
                  fontweight="bold",fontsize=12)
    ax9.grid(axis="y",alpha=0.3)

    # ── 10. Tableau récapitulatif ─────────────────────────
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis("off")
    acc_v  = accuracy_score(y_te, y_pred_clf)*100
    prec_v = precision_score(y_te, y_pred_clf, zero_division=0)*100
    rec_v  = recall_score(y_te, y_pred_clf, zero_division=0)*100
    f1_v   = f1_score(y_te, y_pred_clf, average="weighted")*100
    mae_v  = mean_absolute_error(y_te_r, y_pred_r)
    r2_v   = r2_score(y_te_r, y_pred_r)
    rmse_v = np.sqrt(mean_squared_error(y_te_r, y_pred_r))

    data_tab = [
        ["Métrique","Valeur","Interprétation","Seuil idéal"],
        ["Accuracy",        f"{acc_v:.2f}%",  "Taux de décisions correctes",      "> 95%"],
        ["Précision",       f"{prec_v:.2f}%", "OUI prédits qui sont vrais",        "> 90%"],
        ["Rappel",          f"{rec_v:.2f}%",  "OUI réels correctement détectés",   "> 90%"],
        ["F1-score",        f"{f1_v:.2f}%",   "Équilibre précision/rappel",         "> 90%"],
        ["AUC-ROC",         f"{roc_auc:.4f}", "Capacité discriminante (1=parfait)", "> 0.95"],
        ["R² régression",   f"{r2_v:.4f}",    "Variance du volume expliquée",       "> 0.90"],
        ["MAE régression",  f"{mae_v:.1f} L", "Erreur absolue moyenne (litres)",    "< 100 L"],
        ["RMSE régression", f"{rmse_v:.1f} L","Sensibilité aux grandes erreurs",    "< 150 L"],
    ]
    table = ax10.table(cellText=data_tab[1:], colLabels=data_tab[0],
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2.2)
    for (row,col), cell in table.get_celld().items():
        if row==0:
            cell.set_facecolor(C["bleu"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row%2==0:
            cell.set_facecolor("#EEF2FF")
        cell.set_edgecolor("white")
    ax10.set_title(" Tableau récapitulatif — Métriques complètes",
                   fontweight="bold", fontsize=13, pad=20)

    fig.suptitle(
        "Évaluation complète — Système d'Irrigation Intelligente CI\n"
        "Tomate | 200m² | Yamoussoukro | Dataset 2022-2024 (1096 jours)",
        fontsize=14, fontweight="bold", y=0.98, color=C["bleu"])

    chemin = os.path.join(OUT_DIR, "evaluation_complete.png")
    plt.savefig(chemin, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"    evaluation_complete.png → {chemin}")
    return chemin


# ══════════════════════════════════════════════════════════════
# 7. BILAN FINAL CONSOLE
# ══════════════════════════════════════════════════════════════
def bilan_final(acc, prec, rec, f1, roc_auc,
                mae, rmse, r2, mape, faux_neg, faux_pos):
    def badge(v, ok, bon):
        return "🟢 Excellent" if v>=bon else "🟡 Bon" if v>=ok else "🔴 À améliorer"

    print("\n" + "═"*60)
    print("   BILAN FINAL — MÉTRIQUES COMPLÈTES (2022-2024)")
    print("═"*60)
    print(f"\n  CLASSIFICATION :")
    print(f"  {'─'*50}")
    print(f"  Accuracy   : {acc*100:>6.2f}%  {badge(acc,0.90,0.95)}")
    print(f"  Précision  : {prec*100:>6.2f}%  {badge(prec,0.85,0.95)}")
    print(f"  Rappel     : {rec*100:>6.2f}%  {badge(rec,0.85,0.95)}")
    print(f"  F1-score   : {f1*100:>6.2f}%  {badge(f1,0.88,0.95)}")
    print(f"  AUC-ROC    : {roc_auc:>6.4f}   {badge(roc_auc,0.90,0.97)}")
    print(f"\n  RÉGRESSION :")
    print(f"  {'─'*50}")
    print(f"  R²         : {r2:>6.4f}   {badge(r2,0.85,0.95)}")
    print(f"  MAE        : {mae:>6.1f} L  {'🟢 Excellent' if mae<80 else '🟡 Bon' if mae<120 else '🔴 À améliorer'}")
    print(f"  RMSE       : {rmse:>6.1f} L")
    print(f"  MAPE       : {mape:>6.1f}%")
    print(f"\n  SÉCURITÉ AGRONOMIQUE :")
    print(f"  {'─'*50}")
    print(f"  Oublis irrigation (FN) : {len(faux_neg):>3} j  "
          f"{'🟢' if len(faux_neg)<=5 else '🔴'}")
    print(f"  Gaspillages eau (FP)   : {len(faux_pos):>3} j  "
          f"{'🟢' if len(faux_pos)==0 else '🟡'}")
    print(f"\n  Fichiers générés :")
    print(f"     outputs/evaluation_complete.png")
    print("═"*60)


# ══════════════════════════════════════════════════════════════
# EXÉCUTION
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    (clf, reg, df, X, y_clf,
     X_te, y_te, X_irr, X_te_r, y_te_r) = charger()

    y_pred_clf, y_proba, cm, acc, prec, rec, f1, roc_auc, fpr, tpr = \
        evaluer_classification(clf, X, y_clf, X_te, y_te)

    y_pred_r, mae, rmse, r2, mape = \
        evaluer_regression(reg, X_te_r, y_te_r)

    analyse_annee_saison(clf, df)

    faux_neg, faux_pos = analyser_erreurs(clf, df)

    generer_graphiques(clf, reg, df, y_te, y_pred_clf,
                       y_proba, y_te_r, y_pred_r, cm, fpr, tpr, roc_auc)

    bilan_final(acc, prec, rec, f1, roc_auc,
                mae, rmse, r2, mape, faux_neg, faux_pos)