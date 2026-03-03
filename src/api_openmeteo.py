"""
=============================================================
 SYSTÈME D'IRRIGATION INTELLIGENTE — CÔTE D'IVOIRE
 Module 6 : API Open-Meteo (temps réel) + Décision ML
=============================================================
 Prédiction du jour + 3 jours via données météo réelles
 Intégration capteur DHT22/Arduino (optionnelle)

 Installation :
   pip install openmeteo-requests requests-cache retry-requests pyserial

 Usage :
   python src/06_api_openmeteo.py                  ← décision du jour
   python src/06_api_openmeteo.py --test            ← affiche les données brutes
   python src/06_api_openmeteo.py --port COM4       ← avec capteur Arduino
   python src/06_api_openmeteo.py --port /dev/ttyUSB0  ← Linux/Mac
=============================================================
"""

import os, sys, math, joblib, argparse
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

# ─── Chemins ─────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CLF_PATH = os.path.join(MODEL_DIR, "modele_classification.joblib")
REG_PATH = os.path.join(MODEL_DIR, "modele_regression.joblib")

# ─── Constantes Yamoussoukro ──────────────────────────────────
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
# 1. APPEL API OPEN-METEO (format openmeteo-requests officiel)
# ══════════════════════════════════════════════════════════════
def appeler_api() -> tuple:
    """
    Récupère les données météo via le client officiel openmeteo-requests.
    Utilise le cache 1h pour éviter les appels répétés.
    Retourne (df_horaire, df_quotidien).
    """
    try:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry
    except ImportError:
        print("❌ Bibliothèques manquantes. Exécutez :")
        print("   pip install openmeteo-requests requests-cache retry-requests")
        sys.exit(1)

    print("🌐 Connexion à l'API Open-Meteo...")

    # Cache 1 heure + retry automatique (5 tentatives)
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo     = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude"      : LATITUDE,
        "longitude"     : LONGITUDE,
        "timezone"      : TIMEZONE,
        "forecast_days" : 4,   # Aujourd'hui + J+1, J+2, J+3

        # ── Variables horaires ─────────────────────────────
        "hourly": [
            "temperature_2m",           # 0 — Température air 2m (°C)
            "rain",                     # 1 — Pluie horaire (mm)
            "wind_speed_10m",           # 2 — Vitesse vent 10m (km/h)
            "soil_temperature_0_to_7cm",# 3 — Temp sol 0-7cm (°C)
            "soil_moisture_7_to_28cm",  # 4 — Humidité sol 7-28cm (m³/m³)
            "relative_humidity_2m",     # 5 — Humidité air (%)
            "soil_moisture_0_to_7cm",   # 6 — Humidité sol 0-7cm (m³/m³)
        ],

        # ── Variables quotidiennes ─────────────────────────
        "daily": [
            "temperature_2m_mean",          # 0 — Temp moyenne (°C)
            "temperature_2m_max",           # 1 — Temp max (°C)
            "temperature_2m_min",           # 2 — Temp min (°C)
            "wind_speed_10m_max",           # 3 — Vent max (km/h)
            "sunrise",                      # 4 — Lever du soleil (ISO)
            "sunset",                       # 5 — Coucher du soleil (ISO)
            "shortwave_radiation_sum",      # 6 — Rayonnement Rs (MJ/m²)
            "et0_fao_evapotranspiration",   # 7 — ET₀ FAO (mm)
            "precipitation_sum",            # 8 — Précipitations totales (mm)
        ],
    }

    responses = openmeteo.weather_api(url, params=params)
    response  = responses[0]

    print(f"   ✅ Station   : {response.Latitude():.4f}°N "
          f"{abs(response.Longitude()):.4f}°W")
    print(f"   ✅ Altitude  : {response.Elevation():.0f} m asl")
    print(f"   ✅ Timezone  : {response.Timezone()}")

    # ── Traitement données HORAIRES ────────────────────────
    hourly = response.Hourly()

    dates_h = pd.date_range(
        start     = pd.to_datetime(hourly.Time(),    unit="s", utc=True),
        end       = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq      = pd.Timedelta(seconds=hourly.Interval()),
        inclusive = "left"
    ).tz_convert(TIMEZONE)

    df_h = pd.DataFrame({
        "datetime"               : dates_h,
        "temp_C"                 : hourly.Variables(0).ValuesAsNumpy(),
        "pluie_mm"               : hourly.Variables(1).ValuesAsNumpy(),
        "vent_10m_kmh"           : hourly.Variables(2).ValuesAsNumpy(),
        "temp_sol_0_7cm"         : hourly.Variables(3).ValuesAsNumpy(),
        "humidite_sol_7_28cm_m3" : hourly.Variables(4).ValuesAsNumpy(),
        "humidite_air_pct"       : hourly.Variables(5).ValuesAsNumpy(),
        "humidite_sol_0_7cm_m3"  : hourly.Variables(6).ValuesAsNumpy(),
    })

    # Conversion m³/m³ → % (1 m³/m³ = 100%)
    df_h["humidite_sol_7_28cm_pct"] = (df_h["humidite_sol_7_28cm_m3"] * 100).round(1)
    df_h["humidite_sol_0_7cm_pct"]  = (df_h["humidite_sol_0_7cm_m3"]  * 100).round(1)
    df_h["date_only"] = pd.to_datetime(df_h["datetime"]).dt.date

    # ── Traitement données QUOTIDIENNES ───────────────────
    daily = response.Daily()

    dates_d = pd.date_range(
        start     = pd.to_datetime(daily.Time(),    unit="s", utc=True),
        end       = pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq      = pd.Timedelta(seconds=daily.Interval()),
        inclusive = "left"
    ).tz_convert(TIMEZONE)

    df_q = pd.DataFrame({
        "date"              : dates_d.date,
        "temp_moy_C"        : daily.Variables(0).ValuesAsNumpy(),
        "temp_max_C"        : daily.Variables(1).ValuesAsNumpy(),
        "temp_min_C"        : daily.Variables(2).ValuesAsNumpy(),
        "vent_max_kmh"      : daily.Variables(3).ValuesAsNumpy(),
        "lever_soleil"      : daily.Variables(4).ValuesInt64AsNumpy(),
        "coucher_soleil"    : daily.Variables(5).ValuesInt64AsNumpy(),
        "rayonnement_Rs_MJ" : daily.Variables(6).ValuesAsNumpy(),
        "ET0_reference_mm"  : daily.Variables(7).ValuesAsNumpy(),
        "pluie_totale_mm"   : daily.Variables(8).ValuesAsNumpy(),
    })

    print(f"   ✅ Données   : {len(df_h)} heures | {len(df_q)} jours")
    return df_h, df_q


# ══════════════════════════════════════════════════════════════
# 2. LECTURE CAPTEUR DHT22 VIA ARDUINO
# ══════════════════════════════════════════════════════════════
def lire_capteur_arduino(port: str = "COM3") -> float | None:
    """
    Lit l'humidité du sol depuis le capteur DHT22 via Arduino (Serial).

    Ports communs :
      Windows : COM3, COM4, COM5...
      Linux   : /dev/ttyUSB0, /dev/ttyACM0
      Mac     : /dev/cu.usbmodem...

    L'Arduino doit envoyer le format : "HUMIDITE:45.3\\n"
    (voir fichier arduino_dht22.ino)

    Retourne None si le capteur n'est pas connecté.
    """
    try:
        import serial
        import time

        print(f"   🔌 Connexion Arduino sur {port}...")
        ser = serial.Serial(port, baudrate=9600, timeout=3)
        time.sleep(2.5)  # Attendre l'initialisation Arduino

        # Lire jusqu'à 5 lignes pour trouver une valeur valide
        for _ in range(5):
            ligne = ser.readline().decode("utf-8", errors="ignore").strip()
            if "HUMIDITE:" in ligne:
                valeur = float(ligne.split(":")[1])
                ser.close()
                print(f"   ✅ Capteur DHT22 : humidité sol = {valeur:.1f}%")
                return valeur

        ser.close()
        print(f"   ⚠️  Capteur connecté mais pas de données valides")
        return None

    except ImportError:
        print("   ℹ️  pyserial absent → pip install pyserial")
        return None
    except Exception as e:
        if "could not open port" in str(e).lower() or "no such file" in str(e).lower():
            print(f"   ℹ️  Arduino non connecté sur {port} "
                  f"→ utilisation Open-Meteo")
        else:
            print(f"   ⚠️  Erreur capteur ({e}) → utilisation Open-Meteo")
        return None


# ══════════════════════════════════════════════════════════════
# 3. CONSTRUCTION DES FEATURES PAR JOUR
# ══════════════════════════════════════════════════════════════
def construire_features(df_h: pd.DataFrame,
                         df_q: pd.DataFrame,
                         index_jour: int,
                         humidite_capteur: float = None) -> dict:
    """
    Construit le vecteur de features pour le jour à l'index donné.
    index_jour : 0=aujourd'hui, 1=J+1, 2=J+2, 3=J+3

    Si humidite_capteur est fourni (lecture DHT22), il remplace
    la valeur Open-Meteo pour l'humidité du sol (uniquement J=0).
    """
    row_q  = df_q.iloc[index_jour]
    date_j = row_q["date"]

    # Filtrer les heures du jour concerné
    df_jour = df_h[df_h["date_only"] == date_j]

    if len(df_jour) == 0:
        # Fallback si pas de données horaires pour ce jour
        hs_moy = float(df_h["humidite_sol_7_28cm_pct"].mean())
        hs_min = hs_moy - 3.0
        hs_07  = hs_moy - 1.5
        ha_moy = float(df_h["humidite_air_pct"].mean())
    else:
        hs_moy = float(df_jour["humidite_sol_7_28cm_pct"].mean())
        hs_min = float(df_jour["humidite_sol_7_28cm_pct"].min())
        hs_07  = float(df_jour["humidite_sol_0_7cm_pct"].mean())
        ha_moy = float(df_jour["humidite_air_pct"].mean())

    # Remplacer par la lecture réelle du capteur DHT22 (J=0 uniquement)
    if humidite_capteur is not None and index_jour == 0:
        hs_moy = humidite_capteur
        hs_min = max(10.0, humidite_capteur - 3.0)
        hs_07  = max(10.0, humidite_capteur - 1.5)

    # Calculs agronomiques
    ET0    = float(row_q["ET0_reference_mm"])
    ETc    = round(ET0 * KC_MOYEN, 2)
    pluie  = float(row_q["pluie_totale_mm"])
    pluie_e= round(pluie * 0.80, 2)
    deficit= round(ETc - pluie_e, 2)
    vent_u2= round(float(row_q["vent_max_kmh"]) / 3.6 * 0.748, 3)

    d = pd.to_datetime(str(date_j))

    return {
        # ── Features ML ────────────────────────────────
        "humidite_sol_moy_pct" : round(hs_moy, 1),
        "humidite_sol_min_pct" : round(hs_min, 1),
        "humidite_sol_0_7_moy" : round(hs_07,  1),
        "temp_max_C"           : float(row_q["temp_max_C"]),
        "temp_min_C"           : float(row_q["temp_min_C"]),
        "temp_moy_C"           : float(row_q["temp_moy_C"]),
        "humidite_air_moy_pct" : round(ha_moy, 1),
        "vent_u2_ms"           : vent_u2,
        "rayonnement_Rs_MJ"    : float(row_q["rayonnement_Rs_MJ"]),
        "ET0_reference_mm"     : ET0,
        "ETc_mm"               : ETc,
        "pluie_totale_mm"      : pluie,
        "pluie_effective_mm"   : pluie_e,
        "deficit_hydrique_mm"  : deficit,
        "jour_annee"           : int(d.dayofyear),
        "mois"                 : int(d.month),

        # ── Infos affichage (pas utilisées par ML) ─────
        "_date"         : str(date_j),
        "_ET0"          : ET0,
        "_pluie"        : pluie,
        "_deficit"      : deficit,
        "_humidite_sol" : round(hs_moy, 1),
        "_source_sol"   : "DHT22" if (humidite_capteur and index_jour == 0)
                          else "Open-Meteo",
        "_capteur"      : humidite_capteur is not None and index_jour == 0,
    }


# ══════════════════════════════════════════════════════════════
# 4. MOTEUR DE DÉCISION (Règles → ML)
# ══════════════════════════════════════════════════════════════
def appliquer_regles(f: dict):
    """Couche 1 — règles agronomiques explicites."""
    hs, p, d = (f["humidite_sol_moy_pct"],
                f["pluie_totale_mm"],
                f["deficit_hydrique_mm"])
    if hs > 70:
        return 0, f"Sol humide ({hs:.1f}% > 70%)", True
    if p > 10:
        return 0, f"Forte pluie prévue ({p:.1f} mm)", True
    if p > 5 and hs > 50:
        return 0, f"Pluie modérée + sol ok ({hs:.1f}%)", True
    if d <= 0:
        return 0, f"Pas de déficit hydrique ({d:.2f} mm)", True
    return 1, "Analyse ML requise", False


def decider(clf, reg, f: dict) -> dict:
    """Décision complète : règles agronomiques → modèle ML."""
    dec_r, raison, court = appliquer_regles(f)

    if court:
        return {"irriguer": dec_r, "volume_L": 0.0,
                "raison": raison, "source": "Règle agronomique",
                "confiance": 100.0}

    # Couche 2 — modèle ML
    X         = pd.DataFrame([f])[FEATURES]
    proba     = clf.predict_proba(X)[0]
    decision  = int(clf.predict(X)[0])
    confiance = round(proba[decision] * 100, 1)

    volume = 0.0
    if decision == 1:
        vol_ml  = float(reg.predict(X)[0])
        facteur = max(0.0, (65.0 - f["humidite_sol_moy_pct"]) / 25.0)
        vol_fo  = f["deficit_hydrique_mm"] * facteur * SURFACE_M2 / EFFICACITE
        volume  = round(max(0.60 * vol_ml + 0.40 * vol_fo, 0), 1)

    return {
        "irriguer"  : decision,
        "volume_L"  : volume,
        "raison"    : (f"ML {confiance}% — "
                       f"Déficit {f['deficit_hydrique_mm']:.2f}mm | "
                       f"Sol {f['humidite_sol_moy_pct']:.1f}%"),
        "source"    : "Modèle ML (RF 2022-2024)",
        "confiance" : confiance,
    }


# ══════════════════════════════════════════════════════════════
# 5. AFFICHAGE CONSOLE
# ══════════════════════════════════════════════════════════════
def afficher(label: str, f: dict, res: dict):
    """Affiche une décision avec toutes les données météo du jour."""
    icone = "✅ ARROSER       " if res["irriguer"] else "❌ NE PAS ARROSER"
    capteur_txt = f"🔌 DHT22" if f["_capteur"] else "🌐 API"

    print(f"\n  ┌{'─'*62}┐")
    print(f"  │  {label:<58}  │")
    print(f"  │  📅 {f['_date']}  |  Sol : {capteur_txt:<42}│")
    print(f"  ├{'─'*62}┤")
    print(f"  │  📊 Météo du jour :{'':>43}│")
    print(f"  │     🌡  Temp         : {f['temp_max_C']:.1f}°C max / "
          f"{f['temp_min_C']:.1f}°C min / "
          f"{f['temp_moy_C']:.1f}°C moy{'':>4}│")
    print(f"  │     🌧  Pluie        : {f['_pluie']:.1f} mm{'':>43}│")
    print(f"  │     💧  Humidité sol : {f['_humidite_sol']:.1f}% "
          f"({f['_source_sol']}){'':>32}│")
    print(f"  │     ☁  Humidité air : {f['humidite_air_moy_pct']:.1f}%{'':>40}│")
    print(f"  │     ☀  Rayonnement  : {f['rayonnement_Rs_MJ']:.2f} MJ/m²{'':>40}│")
    print(f"  │     💨  Vent         : {f['vent_u2_ms']:.2f} m/s (u2){'':>39}│")
    print(f"  ├{'─'*62}┤")
    print(f"  │  🌿 Agronomie :{'':>47}│")
    print(f"  │     ET₀ réf.    : {f['ET0_reference_mm']:.2f} mm/j  |  "
          f"ETc : {f['ETc_mm']:.2f} mm/j {'':>23}│")
    print(f"  │     Déficit     : {f['_deficit']:.2f} mm  |  "
          f"Pluie eff. : {f['pluie_effective_mm']:.2f} mm{'':>21}│")
    print(f"  ├{'─'*62}┤")
    print(f"  │  🤖 DÉCISION : {icone:<48}│")
    if res["irriguer"]:
        print(f"  │     Volume   : {res['volume_L']:.0f} L "
              f"(champ {SURFACE_M2}m²){'':>38}│")
    print(f"  │     Source   : {res['source']:<47}│")
    raison = res["raison"]
    print(f"  │     Raison   : {raison[:47]:<47}│")
    if len(raison) > 47:
        print(f"  │               {raison[47:94]:<47}│")
    print(f"  └{'─'*62}┘")


def afficher_donnees_brutes(df_h: pd.DataFrame, df_q: pd.DataFrame):
    """Mode --test : affiche les données brutes de l'API."""
    print("\n" + "═" * 70)
    print("  📊 DONNÉES BRUTES OPEN-METEO (4 jours)")
    print("═" * 70)
    print("\n  QUOTIDIENNES :")
    cols_q = ["date", "temp_max_C", "temp_min_C", "temp_moy_C",
              "ET0_reference_mm", "pluie_totale_mm", "rayonnement_Rs_MJ", "vent_max_kmh"]
    print(df_q[cols_q].to_string(index=False))
    print("\n  HORAIRES — 8 premières heures :")
    cols_h = ["datetime", "temp_C", "humidite_air_pct",
              "pluie_mm", "humidite_sol_7_28cm_pct", "humidite_sol_0_7cm_pct"]
    print(df_h[cols_h].head(8).to_string(index=False))
    print("═" * 70)


# ══════════════════════════════════════════════════════════════
# 6. GÉNÉRATION SMS
# ══════════════════════════════════════════════════════════════
def generer_sms(resultats: list) -> str:
    """Génère le message SMS complet pour M. Koffi."""
    auj = resultats[0]
    f0  = auj["features"]
    r0  = auj["res"]

    dec = "ARROSER ✅" if r0["irriguer"] else "PAS D'ARROSAGE ❌"
    vol = f"{r0['volume_L']:.0f}L" if r0["irriguer"] else "0L"

    sms  = f"🌱 AgroIrri CI — {f0['_date']}\n"
    sms += f"Décision : {dec}\n"
    sms += f"Volume   : {vol}\n"
    sms += f"Sol      : {f0['_humidite_sol']:.1f}% ({f0['_source_sol']})\n"
    sms += f"Pluie    : {f0['_pluie']:.1f}mm | ET₀ : {f0['_ET0']:.2f}mm\n\n"
    sms += "Prévisions :\n"

    for r in resultats[1:]:
        fi = r["features"]; ri = r["res"]
        e  = "✅" if ri["irriguer"] else "❌"
        v  = f"{ri['volume_L']:.0f}L" if ri["irriguer"] else "0L"
        sms += f"  {fi['_date']} : {e} {v}\n"

    sms += f"\nIA entraînée sur 2022-2024 | Open-Meteo\n"
    sms += f"Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}"
    return sms


# ══════════════════════════════════════════════════════════════
# 7. SAUVEGARDE LOG
# ══════════════════════════════════════════════════════════════
def sauvegarder_log(resultats: list):
    """Enregistre toutes les décisions du jour dans l'historique CSV."""
    rows = []
    for r in resultats:
        f = r["features"]; res = r["res"]
        rows.append({
            "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M"),
            "date"         : f["_date"],
            "irriguer"     : res["irriguer"],
            "volume_L"     : res["volume_L"],
            "source"       : res["source"],
            "humidite_sol" : f["_humidite_sol"],
            "source_sol"   : f["_source_sol"],
            "ET0_mm"       : f["_ET0"],
            "pluie_mm"     : f["_pluie"],
            "deficit_mm"   : f["_deficit"],
            "confiance_pct": res["confiance"],
        })
    lp = os.path.join(OUT_DIR, "historique_decisions.csv")
    pd.DataFrame(rows).to_csv(lp, mode="a",
                               header=not os.path.exists(lp), index=False)
    print(f"\n  💾 {len(rows)} décisions enregistrées → historique_decisions.csv")


# ══════════════════════════════════════════════════════════════
# EXÉCUTION PRINCIPALE
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Irrigation intelligente CI")
    parser.add_argument("--test",  action="store_true",
                        help="Affiche données brutes API sans décision ML")
    parser.add_argument("--port",  default="COM3",
                        help="Port série Arduino (COM3, /dev/ttyUSB0...)")
    parser.add_argument("--no-capteur", action="store_true",
                        help="Désactive la lecture du capteur Arduino")
    args = parser.parse_args()

    print("\n" + "═" * 66)
    print("  🌱 SYSTÈME D'IRRIGATION INTELLIGENTE — M. KOFFI")
    print(f"  📍 Yamoussoukro, CI  |  🍅 Tomate  |  {SURFACE_M2} m²")
    print(f"  🕐 {datetime.now().strftime('%A %d/%m/%Y  %H:%M')}")
    print(f"  🧠 Modèles entraînés sur 2022-2024 (1096 jours)")
    print("═" * 66)

    # ── 1. Appel API ──────────────────────────────────────
    df_h, df_q = appeler_api()

    # ── 2. Mode test ──────────────────────────────────────
    if args.test:
        afficher_donnees_brutes(df_h, df_q)
        print("\n  ✅ Test API réussi. Données bien reçues.")
        print("     Lancez sans --test pour obtenir les décisions ML.")
        sys.exit(0)

    # ── 3. Lecture capteur DHT22 ──────────────────────────
    humidite_capteur = None
    if not args.no_capteur:
        humidite_capteur = lire_capteur_arduino(args.port)
    if humidite_capteur is None:
        print("   ℹ️  Humidité sol depuis Open-Meteo (pas de capteur)")

    # ── 4. Chargement modèles ML ──────────────────────────
    if not os.path.exists(CLF_PATH):
        print("\n❌ Modèles introuvables → lancez d'abord : 02_entrainement_ml.py")
        sys.exit(1)
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    print(f"   ✅ Modèles ML chargés")

    # ── 5. Décisions pour 4 jours ─────────────────────────
    labels = [
        "AUJOURD'HUI  (J)  ",
        "DEMAIN       (J+1)",
        "APRÈS-DEMAIN (J+2)",
        "DANS 3 JOURS (J+3)",
    ]
    resultats = []

    print("\n" + "═" * 66)
    for i in range(min(4, len(df_q))):
        f   = construire_features(df_h, df_q, i, humidite_capteur)
        res = decider(clf, reg, f)
        afficher(labels[i], f, res)
        resultats.append({"features": f, "res": res})

    # ── 6. SMS ────────────────────────────────────────────
    sms = generer_sms(resultats)
    print("\n" + "═" * 66)
    print("  📱 MESSAGE SMS — M. KOFFI :")
    print("  " + "─" * 52)
    for ligne in sms.split("\n"):
        print(f"  {ligne}")
    print("  " + "─" * 52)

    # ── 7. Log ────────────────────────────────────────────
    sauvegarder_log(resultats)
    print("═" * 66)