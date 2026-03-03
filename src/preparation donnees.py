"""
=============================================================
 SYSTÈME D'IRRIGATION INTELLIGENTE — CÔTE D'IVOIRE
 Étape 1 : Préparation et nettoyage des données
=============================================================
 Source      : Open-Meteo (Yamoussoukro, 2022-2024)
 Coordonnées : 6.82°N, 5.28°W, altitude 212m
 Culture     : Tomate (Solanum lycopersicum)
 Surface     : 200 m²  |  Période : 3 ans (1096 jours)
=============================================================
"""

import os, math
import pandas as pd
import numpy as np

# ─── Chemins ────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_BRUT  = os.path.join(DATA_DIR, "open_meteo_brut.csv")
CSV_CLEAN = os.path.join(DATA_DIR, "yamoussoukro_dataset_ML.csv")

# ─── Constantes agronomiques ────────────────────────────────
KC_MOYEN   = 1.05   # Kc tomate moyen annuel (FAO-56)
SURFACE_M2 = 200    # Surface champ M. Koffi
EFFICACITE = 0.90   # Efficacité goutte-à-goutte


# ══════════════════════════════════════════════════════════════
# 1. LECTURE DU CSV OPEN-METEO (double bloc)
# ══════════════════════════════════════════════════════════════
def detecter_separation(filepath: str) -> int:
    with open(filepath, "r", encoding="utf-8") as f:
        for i, ligne in enumerate(f):
            if i > 5 and ligne.startswith("time,temperature_2m_mean"):
                return i
    raise ValueError("Séparation horaire/quotidien introuvable dans le CSV.")


def lire_open_meteo(filepath: str):
    print(" Lecture du fichier brut Open-Meteo...")
    sep      = detecter_separation(filepath)
    n_h      = sep - 4
    df_h     = pd.read_csv(filepath, skiprows=3, nrows=n_h,      parse_dates=["time"])
    df_q     = pd.read_csv(filepath, skiprows=sep - 1,            parse_dates=["time"])
    print(f"    Horaires   : {len(df_h):,} lignes "
          f"({df_h['time'].min().date()} → {df_h['time'].max().date()})")
    print(f"    Quotidiens : {len(df_q):,} lignes "
          f"({df_q['time'].min().date()} → {df_q['time'].max().date()})")
    return df_h, df_q


# ══════════════════════════════════════════════════════════════
# 2. RENOMMAGE
# ══════════════════════════════════════════════════════════════
def renommer(df_h, df_q):
    df_h.rename(columns={
        "time"                            : "datetime",
        "temperature_2m (°C)"             : "temp_C",
        "rain (mm)"                       : "pluie_mm",
        "wind_speed_10m (km/h)"           : "vent_10m_kmh",
        "soil_temperature_0_to_7cm (°C)"  : "temp_sol_0_7cm",
        "soil_moisture_7_to_28cm (m³/m³)" : "humidite_sol_7_28cm_m3",
        "relative_humidity_2m (%)"        : "humidite_air_pct",
        "soil_moisture_0_to_7cm (m³/m³)"  : "humidite_sol_0_7cm_m3",
    }, inplace=True)
    df_q.rename(columns={
        "time"                            : "date",
        "temperature_2m_mean (°C)"        : "temp_moy_C",
        "temperature_2m_max (°C)"         : "temp_max_C",
        "temperature_2m_min (°C)"         : "temp_min_C",
        "wind_speed_10m_max (km/h)"       : "vent_max_kmh",
        "sunrise (iso8601)"               : "lever_soleil",
        "sunset (iso8601)"                : "coucher_soleil",
        "shortwave_radiation_sum (MJ/m²)" : "rayonnement_Rs_MJ",
        "et0_fao_evapotranspiration (mm)" : "ET0_reference_mm",
        "precipitation_sum (mm)"          : "pluie_totale_mm",
    }, inplace=True)
    return df_h, df_q


# ══════════════════════════════════════════════════════════════
# 3. CONVERSION HUMIDITÉ SOL m³/m³ → %
# ══════════════════════════════════════════════════════════════
def convertir_sol(df_h):
    df_h["humidite_sol_0_7cm_pct"]  = (df_h["humidite_sol_0_7cm_m3"]  * 100).round(1)
    df_h["humidite_sol_7_28cm_pct"] = (df_h["humidite_sol_7_28cm_m3"] * 100).round(1)
    return df_h


# ══════════════════════════════════════════════════════════════
# 4. AGRÉGATION HORAIRE → QUOTIDIEN
# ══════════════════════════════════════════════════════════════
def agreger(df_h):
    df_h["date"] = df_h["datetime"].dt.date
    return df_h.groupby("date").agg(
        humidite_sol_moy_pct = ("humidite_sol_7_28cm_pct", "mean"),
        humidite_sol_min_pct = ("humidite_sol_7_28cm_pct", "min"),
        humidite_sol_max_pct = ("humidite_sol_7_28cm_pct", "max"),
        humidite_sol_0_7_moy = ("humidite_sol_0_7cm_pct",  "mean"),
        pluie_horaire_sum_mm = ("pluie_mm",                 "sum"),
        humidite_air_moy_pct = ("humidite_air_pct",         "mean"),
        vent_moy_kmh         = ("vent_10m_kmh",             "mean"),
    ).reset_index()


# ══════════════════════════════════════════════════════════════
# 5. PENMAN-MONTEITH (FAO-56)
# ══════════════════════════════════════════════════════════════
def penman_monteith(T_max, T_min, T_moy, HR, u2, Rs):
    try:
        e_s   = (0.6108*math.exp(17.27*T_max/(T_max+237.3)) +
                 0.6108*math.exp(17.27*T_min/(T_min+237.3))) / 2
        e_a   = HR/100 * e_s
        delta = 4098*0.6108*math.exp(17.27*T_moy/(T_moy+237.3)) / (T_moy+237.3)**2
        gamma = 0.0665
        Rns   = 0.77 * Rs
        Rnl   = (4.903e-9
                 * ((T_max+273.16)**4 + (T_min+273.16)**4) / 2
                 * (0.34 - 0.14*math.sqrt(max(e_a, 0.001)))
                 * 0.8)
        Rn    = Rns - Rnl
        num   = 0.408*delta*Rn + gamma*(900/(T_moy+273))*u2*(e_s-e_a)
        den   = delta + gamma*(1+0.34*u2)
        return round(max(num/den, 0), 2)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
# 6. RÈGLES AGRONOMIQUES + LABELS
# ══════════════════════════════════════════════════════════════
def decision(row) -> int:
    hs, p, d = row["humidite_sol_moy_pct"], row["pluie_totale_mm"], row["deficit_hydrique_mm"]
    if hs > 70:              return 0
    if p > 10:               return 0
    if p > 5 and hs > 50:   return 0
    if d <= 0:               return 0
    return 1


def volume(row) -> float:
    if row["irriguer"] == 0: return 0.0
    facteur = max(0.0, (65.0 - row["humidite_sol_moy_pct"]) / 25.0)
    besoin  = row["deficit_hydrique_mm"] * facteur
    return round(besoin * SURFACE_M2 / EFFICACITE, 1) if besoin > 0 else 0.0


# ══════════════════════════════════════════════════════════════
# 7. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════
def preparer_dataset(source: str = "auto") -> pd.DataFrame:
    """
    source='auto'  → charge le nettoyé si existe, sinon relit le brut
    source='brut'  → relit toujours le CSV Open-Meteo
    source='clean' → charge uniquement le nettoyé
    """
    if source != "brut" and os.path.exists(CSV_CLEAN):
        print(" Dataset nettoyé (2022-2024) trouvé, chargement...")
        df = pd.read_csv(CSV_CLEAN, parse_dates=["date"])
        # Recalcul des colonnes dérivées si absentes (compatibilité anciens CSV)
        if "annee" not in df.columns:
            df["annee"] = pd.to_datetime(df["date"]).dt.year
        if "mois" not in df.columns:
            df["mois"] = pd.to_datetime(df["date"]).dt.month
        if "saison" not in df.columns:
            df["saison"] = df["mois"].map(lambda m:
                "seche" if m in [11,12,1,2,3] else
                "grande_pluie" if m in [6,7,8,9] else "petite_pluie")
        if "ETc_mm" not in df.columns:
            df["ETc_mm"]              = (df["ET0_reference_mm"] * KC_MOYEN).round(2)
            df["pluie_effective_mm"]  = (df["pluie_totale_mm"]  * 0.80).round(2)
            df["deficit_hydrique_mm"] = (df["ETc_mm"] - df["pluie_effective_mm"]).round(2)
        print(f"    {len(df)} jours | "
              f"{df['date'].min().date()} → {df['date'].max().date()}")
        return df

    df_h, df_q = lire_open_meteo(CSV_BRUT)
    df_h, df_q = renommer(df_h, df_q)
    df_h       = convertir_sol(df_h)
    df_sol     = agreger(df_h)

    df_q["date"] = pd.to_datetime(df_q["date"]).dt.date
    df = df_q.merge(df_sol, on="date", how="left")

    # Vent u2 (10m → 2m, formule FAO)
    df["vent_u2_ms"] = (df["vent_max_kmh"] / 3.6 * 0.748).round(3)

    # ET₀ Penman-Monteith
    print(" Calcul ET₀ Penman-Monteith...")
    df["ET0_calcule_mm"] = df.apply(
        lambda r: penman_monteith(r.temp_max_C, r.temp_min_C, r.temp_moy_C,
                                  r.humidite_air_moy_pct, r.vent_u2_ms,
                                  r.rayonnement_Rs_MJ), axis=1)
    corr = df[["ET0_reference_mm","ET0_calcule_mm"]].corr().iloc[0,1]
    print(f"    Corrélation ET₀ calculé vs référence : {corr:.3f}")

    # Déficit hydrique
    df["ETc_mm"]              = (df["ET0_reference_mm"] * KC_MOYEN).round(2)
    df["pluie_effective_mm"]  = (df["pluie_totale_mm"]  * 0.80).round(2)
    df["deficit_hydrique_mm"] = (df["ETc_mm"] - df["pluie_effective_mm"]).round(2)

    # Labels
    df["irriguer"]      = df.apply(decision, axis=1)
    df["volume_litres"] = df.apply(volume,   axis=1)

    # Contexte CI
    df["mois"]       = pd.to_datetime(df["date"]).dt.month
    df["jour_annee"] = pd.to_datetime(df["date"]).dt.dayofyear
    df["annee"]      = pd.to_datetime(df["date"]).dt.year
    df["saison"]     = df["mois"].map(lambda m:
        "seche" if m in [11,12,1,2,3] else
        "grande_pluie" if m in [6,7,8,9] else "petite_pluie")

    df.to_csv(CSV_CLEAN, index=False)
    print(f"\n Dataset sauvegardé → {CSV_CLEAN}")
    return df


# ══════════════════════════════════════════════════════════════
# 8. RAPPORT DE QUALITÉ
# ══════════════════════════════════════════════════════════════
def rapport_qualite(df: pd.DataFrame):
    # Recalcul des colonnes dérivées si absentes (ancien CSV sans ces colonnes)
    if "annee" not in df.columns:
        df["annee"] = pd.to_datetime(df["date"]).dt.year
    if "mois" not in df.columns:
        df["mois"] = pd.to_datetime(df["date"]).dt.month
    if "saison" not in df.columns:
        df["saison"] = df["mois"].map(lambda m:
            "seche" if m in [11,12,1,2,3] else
            "grande_pluie" if m in [6,7,8,9] else "petite_pluie")

    print("\n" + "═" * 58)
    print("  RAPPORT DE QUALITÉ DU DATASET")
    print("═" * 58)
    print(f"  Période      : {df['date'].min()} → {df['date'].max()}")
    annees = sorted(df['annee'].unique().tolist())
    print(f"  Observations : {len(df)} jours ({len(annees)} années : {annees})")
    print(f"  Colonnes     : {df.shape[1]}")
    print(f"  Valeurs nulles : {df.isnull().sum().sum()}")

    d = df["irriguer"].value_counts()
    print(f"\n  Distribution des décisions :")
    print(f"    OUI : {d.get(1,0):>5} jours ({d.get(1,0)/len(df)*100:.1f}%)")
    print(f"    NON : {d.get(0,0):>5} jours ({d.get(0,0)/len(df)*100:.1f}%)")

    print(f"\n  Par année :")
    for yr, g in df.groupby("annee"):
        v = g[g.volume_litres>0].volume_litres.mean()
        print(f"    {yr} : {len(g)} j | irrig={( g.irriguer==1).sum()} | "
              f"pluie={g.pluie_totale_mm.sum():.0f}mm | vol_moy={v:.0f}L")

    irr = df[df["volume_litres"]>0]["volume_litres"]
    print(f"\n  Volumes : moy={irr.mean():.0f}L | min={irr.min():.0f}L | max={irr.max():.0f}L")
    print(f"  Sol (7-28cm) : moy={df['humidite_sol_moy_pct'].mean():.1f}% | "
          f"min={df['humidite_sol_min_pct'].min():.1f}% | "
          f"max={df['humidite_sol_max_pct'].max():.1f}%")
    print(f"  ET₀ moyen    : {df['ET0_reference_mm'].mean():.2f} mm/j")
    print(f"  Pluie totale : {df['pluie_totale_mm'].sum():.0f} mm "
          f"(~{df['pluie_totale_mm'].sum()/len(annees):.0f} mm/an)")
    print("═" * 58)


if __name__ == "__main__":
    df = preparer_dataset(source="auto")
    rapport_qualite(df)
    print("\n Étape 1 terminée → lancez : entrainement_ml.py")