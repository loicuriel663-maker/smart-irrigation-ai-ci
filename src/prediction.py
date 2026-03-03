"""
=============================================================
 SYSTÈME D'IRRIGATION INTELLIGENTE — CÔTE D'IVOIRE
 Étape 3 : Prédiction & Moteur de décision
=============================================================
 Dataset entraînement : 2022-2024 (1096 jours)
 Deux modes :
   --mode sim  → simulation avec données exemple (défaut)
   --mode api  → données réelles Open-Meteo (internet requis)
=============================================================
"""

import os, sys, joblib, argparse
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime

# ─── Chemins ────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CLF_PATH = os.path.join(MODEL_DIR, "modele_classification.joblib")
REG_PATH = os.path.join(MODEL_DIR, "modele_regression.joblib")

# ─── Constantes ─────────────────────────────────────────────
LATITUDE   =  6.8205
LONGITUDE  = -5.2767
TIMEZONE   = "Africa/Abidjan"
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
# 1. CHARGEMENT DES MODÈLES
# ══════════════════════════════════════════════════════════════
def charger_modeles():
    if not os.path.exists(CLF_PATH) or not os.path.exists(REG_PATH):
        print(" Modèles introuvables → lancez d'abord : 02_entrainement_ml.py")
        sys.exit(1)
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    print(" Modèles ML chargés (entraînés sur 2022-2024)")
    return clf, reg


# ══════════════════════════════════════════════════════════════
# 2. RÈGLES AGRONOMIQUES (couche prioritaire)
# ══════════════════════════════════════════════════════════════
def appliquer_regles(hs, pluie, deficit):
    """Retourne (décision, raison, court_circuit)."""
    if hs > 70:
        return 0, f"Sol suffisamment humide ({hs:.1f}% > 70%)", True
    if pluie > 10:
        return 0, f"Forte pluie prévue ({pluie:.1f} mm > 10mm)", True
    if pluie > 5 and hs > 50:
        return 0, f"Pluie modérée ({pluie:.1f}mm) + sol ok ({hs:.1f}%)", True
    if deficit <= 0:
        return 0, f"Pas de déficit hydrique ({deficit:.2f} mm)", True
    return 1, "Délégué au modèle ML", False


# ══════════════════════════════════════════════════════════════
# 3. MOTEUR DE DÉCISION COMPLET
# ══════════════════════════════════════════════════════════════
def decider(clf, reg, donnees: dict) -> dict:
    """
    Applique règles → ML → retourne décision + volume.
    donnees doit contenir toutes les clés de FEATURES
    plus : pluie_totale_mm, humidite_sol_moy_pct.
    """
    # Calculs dérivés
    ETc     = round(donnees["ET0_reference_mm"] * KC_MOYEN, 2)
    pluie_e = round(donnees["pluie_totale_mm"]  * 0.80, 2)
    deficit = round(ETc - pluie_e, 2)
    donnees.update({"ETc_mm": ETc, "pluie_effective_mm": pluie_e,
                    "deficit_hydrique_mm": deficit})

    hs    = donnees["humidite_sol_moy_pct"]
    pluie = donnees["pluie_totale_mm"]

    # Couche 1 : règles agronomiques
    dec_r, raison, court = appliquer_regles(hs, pluie, deficit)
    if court:
        return {"irriguer": dec_r, "volume_L": 0.0, "raison": raison,
                "source": "Règle agronomique", "confiance": 100.0,
                "ETc_mm": ETc, "deficit_mm": deficit}

    # Couche 2 : modèle ML
    X         = pd.DataFrame([donnees])[FEATURES]
    proba     = clf.predict_proba(X)[0]
    decision  = int(clf.predict(X)[0])
    confiance = round(proba[decision] * 100, 1)

    volume = 0.0
    if decision == 1:
        vol_ml  = float(reg.predict(X)[0])
        facteur = max(0.0, (65.0 - hs) / 25.0)
        vol_fo  = deficit * facteur * SURFACE_M2 / EFFICACITE
        volume  = round(max(0.60 * vol_ml + 0.40 * vol_fo, 0), 1)

    raison = (f"ML confiance {confiance}% — "
              f"Déficit {deficit:.2f}mm, Sol {hs:.1f}%")
    return {"irriguer": decision, "volume_L": volume, "raison": raison,
            "source": "Modèle ML", "confiance": confiance,
            "ETc_mm": ETc, "deficit_mm": deficit}


# ══════════════════════════════════════════════════════════════
# 4. PRÉVISION 3 JOURS (simulation depuis données exemples)
# ══════════════════════════════════════════════════════════════
def prevoir_3_jours(clf, reg, donnees_j0: dict,
                    previsions: list) -> list:
    """
    previsions : liste de 3 dicts (J+1, J+2, J+3)
    Simule l'évolution de l'humidité du sol jour après jour.
    """
    resultats    = []
    humidite_sol = donnees_j0["humidite_sol_moy_pct"]

    for i, meteo in enumerate(previsions, 1):
        # Évolution humidité sol
        if i > 1 and resultats[i-2]["irriguer"]:
            apport_mm    = resultats[i-2]["volume_L"] * EFFICACITE / SURFACE_M2
            humidite_sol = min(80, humidite_sol + apport_mm * 2)

        humidite_sol += meteo.get("pluie_totale_mm", 0) * 0.8 * 0.1
        humidite_sol  = min(90, max(10, humidite_sol))

        d_j = date.today() + timedelta(days=i)
        donnees_j = {**meteo,
                     "humidite_sol_moy_pct": round(humidite_sol, 1),
                     "humidite_sol_min_pct": round(humidite_sol - 3, 1),
                     "humidite_sol_0_7_moy": round(humidite_sol - 2, 1),
                     "jour_annee": d_j.timetuple().tm_yday,
                     "mois":       d_j.month}

        res = decider(clf, reg, donnees_j)
        res["jour"]         = f"J+{i} ({d_j.strftime('%d/%m')})"
        res["humidite_sol"] = round(humidite_sol, 1)
        resultats.append(res)

        # Sol sèche par ETc
        ETc_j        = meteo.get("ET0_reference_mm", 4) * KC_MOYEN
        humidite_sol = max(10, humidite_sol - ETc_j * 0.5)

    return resultats


# ══════════════════════════════════════════════════════════════
# 5. AFFICHAGE
# ══════════════════════════════════════════════════════════════
def afficher(label: str, res: dict, date_str: str = ""):
    icone = " ARROSER       " if res["irriguer"] else " NE PAS ARROSER"
    print(f"\n  ┌{'─'*58}┐")
    print(f"  │  {label:<54}  │")
    if date_str:
        print(f"  │  Date     : {date_str:<45}│")
    print(f"  ├{'─'*58}┤")
    print(f"  │  Décision : {icone:<45}│")
    if res["irriguer"]:
        print(f"  │  Volume   : {res['volume_L']:.0f} L (surface {SURFACE_M2}m²)"
              f"{'':>28}│")
    print(f"  │  Source   : {res['source']:<45}│")
    raison = res["raison"]
    print(f"  │  Raison   : {raison[:45]:<45}│")
    if len(raison) > 45:
        print(f"  │             {raison[45:90]:<45}│")
    print(f"  │  ETc      : {res['ETc_mm']:.2f} mm/j  |  Déficit : {res['deficit_mm']:.2f} mm"
          f"{'':>20}│")
    print(f"  └{'─'*58}┘")


# ══════════════════════════════════════════════════════════════
# 6. SMS
# ══════════════════════════════════════════════════════════════
def generer_sms(res_auj: dict, previsions: list,
                donnees_auj: dict) -> str:
    d   = date.today().strftime("%d/%m/%Y")
    dec = "ARROSER " if res_auj["irriguer"] else "PAS D'ARROSAGE "
    vol = f"{res_auj['volume_L']:.0f}L" if res_auj["irriguer"] else "0L"

    sms  = f"🌱 AgroIrri CI — {d}\n"
    sms += f"Décision : {dec}\n"
    sms += f"Volume   : {vol}\n"
    sms += f"Sol      : {donnees_auj['humidite_sol_moy_pct']:.1f}% | "
    sms += f"Pluie : {donnees_auj['pluie_totale_mm']:.1f}mm | "
    sms += f"ET₀ : {donnees_auj['ET0_reference_mm']:.2f}mm\n\n"
    sms += "Prévisions :\n"
    for p in previsions:
        e = "" if p["irriguer"] else ""
        v = f"{p['volume_L']:.0f}L" if p["irriguer"] else "0L"
        sms += f"  {p['jour']} : {e} {v}\n"
    sms += f"\nSource : Modèle IA (2022-2024) | {datetime.now():%d/%m/%Y %H:%M}"
    return sms


# ══════════════════════════════════════════════════════════════
# 7. DONNÉES EXEMPLES (mode simulation)
# ══════════════════════════════════════════════════════════════
def donnees_exemple_auj() -> dict:
    """
    Données représentatives d'une journée typique de saison sèche
    à Yamoussoukro (janvier-mars).
    Remplacées automatiquement par l'API en mode --mode api.
    """
    d = date.today()
    return {
        # Capteur DHT22 (lecture 06h00)
        "humidite_sol_moy_pct" : 38.0,
        "humidite_sol_min_pct" : 35.0,
        "humidite_sol_0_7_moy" : 36.0,
        # API météo Open-Meteo
        "temp_max_C"           : 34.5,
        "temp_min_C"           : 22.0,
        "temp_moy_C"           : 28.0,
        "humidite_air_moy_pct" : 60.0,
        "vent_u2_ms"           : 2.0,
        "rayonnement_Rs_MJ"    : 18.0,
        "ET0_reference_mm"     : 5.0,
        "pluie_totale_mm"      : 0.0,
        # Contexte
        "jour_annee"           : d.timetuple().tm_yday,
        "mois"                 : d.month,
    }


def previsions_exemple() -> list:
    """Prévisions 3 jours exemples (saison sèche → pluie → sec)."""
    return [
        {"temp_max_C":35.0,"temp_min_C":22.5,"temp_moy_C":28.5,
         "humidite_air_moy_pct":58.0,"vent_u2_ms":2.1,
         "rayonnement_Rs_MJ":19.0,"ET0_reference_mm":5.3,"pluie_totale_mm":0.0},
        {"temp_max_C":29.0,"temp_min_C":21.0,"temp_moy_C":25.0,
         "humidite_air_moy_pct":80.0,"vent_u2_ms":1.5,
         "rayonnement_Rs_MJ":11.0,"ET0_reference_mm":3.5,"pluie_totale_mm":14.0},
        {"temp_max_C":31.5,"temp_min_C":22.0,"temp_moy_C":26.5,
         "humidite_air_moy_pct":65.0,"vent_u2_ms":1.8,
         "rayonnement_Rs_MJ":16.5,"ET0_reference_mm":4.5,"pluie_totale_mm":1.0},
    ]


# ══════════════════════════════════════════════════════════════
# EXÉCUTION
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sim","api"], default="sim",
                        help="sim=données exemples | api=Open-Meteo temps réel")
    parser.add_argument("--port", default="COM3",
                        help="Port série Arduino (ex: COM3 ou /dev/ttyUSB0)")
    args = parser.parse_args()

    clf, reg = charger_modeles()

    print("\n" + "═"*62)
    print("   SYSTÈME D'IRRIGATION INTELLIGENTE — M. KOFFI")
    print(f"   Yamoussoukro, CI |  Tomate | {SURFACE_M2}m²")
    print(f"    {date.today():%d/%m/%Y}  |  Mode : {args.mode.upper()}")
    print(f"   Modèles entraînés sur 2022-2024 (1096 jours)")
    print("═"*62)

    if args.mode == "api":
        # Import du module API
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from api_openmeteo import appeler_api, preparer_features_jour, lire_capteur_arduino

        df_h, df_q        = appeler_api()
        humidite_capteur  = lire_capteur_arduino(args.port)

        donnees_auj       = preparer_features_jour(df_h, df_q, 0)
        if humidite_capteur:
            donnees_auj["humidite_sol_moy_pct"] = humidite_capteur
            donnees_auj["humidite_sol_min_pct"] = humidite_capteur - 2
            donnees_auj["humidite_sol_0_7_moy"] = humidite_capteur - 1.5
            print(f"   Capteur DHT22 : {humidite_capteur:.1f}%")

        previsions_3j = [preparer_features_jour(df_h, df_q, i) for i in range(1, 4)]
    else:
        print("    Mode simulation — données exemples utilisées")
        print("      (lancez avec --mode api pour les données réelles)\n")
        donnees_auj   = donnees_exemple_auj()
        previsions_3j = previsions_exemple()

    # Décision du jour
    res_auj = decider(clf, reg, donnees_auj.copy())
    afficher("AUJOURD'HUI", res_auj, date.today().strftime("%d/%m/%Y"))

    # Prévisions 3 jours
    print("\n   PRÉVISIONS 3 JOURS :")
    previsions_res = prevoir_3_jours(clf, reg, donnees_auj, previsions_3j)
    for p in previsions_res:
        afficher(p["jour"], p)

    # SMS
    sms = generer_sms(res_auj, previsions_res, donnees_auj)
    print("\n   MESSAGE SMS POUR M. KOFFI :")
    print("  " + "─"*50)
    for l in sms.split("\n"):
        print(f"  {l}")
    print("  " + "─"*50)

    # Log
    log = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "date": str(date.today()),
           "mode": args.mode,
           "irriguer": res_auj["irriguer"],
           "volume_L": res_auj["volume_L"],
           "source": res_auj["source"],
           "humidite_sol": donnees_auj["humidite_sol_moy_pct"],
           "ET0": donnees_auj["ET0_reference_mm"],
           "pluie": donnees_auj["pluie_totale_mm"],
           "deficit_mm": res_auj["deficit_mm"]}
    lp = os.path.join(OUT_DIR, "historique_decisions.csv")
    pd.DataFrame([log]).to_csv(lp, mode="a",
                               header=not os.path.exists(lp), index=False)
    print(f"\n   Décision enregistrée → historique_decisions.csv")
    print("═"*62)
    print("\n Pipeline terminé → lancez : 04_backtesting.py ou 05_evaluation.py")