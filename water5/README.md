# Water 5 — Application Flutter
## Irrigation Intelligente par IA — AgroIrri CI

---

## Structure du projet

```
water5/
├── lib/
│   ├── main.dart                    # Point d'entrée — routes — thème
│   ├── theme/
│   │   └── app_theme.dart           # Palette W5Colors + ThemeData
│   ├── models/
│   │   └── models.dart              # IrrigationDecision, DayForecast, HistoryEntry, AppAlert
│   ├── widgets/
│   │   └── common_widgets.dart      # GlassCard, MetricCard, ForecastCard, SoilGauge...
│   └── screens/
│       ├── splash_screen.dart       # Écran d'accueil avec animations
│       ├── loading_screen.dart      # Chargement 5 étapes animées
│       ├── main_screen.dart         # Shell avec AppBar + BottomNav
│       ├── dashboard_screen.dart    # Tableau de bord principal
│       ├── history_screen.dart      # Historique des décisions
│       ├── alerts_screen.dart       # Alertes + SMS M. Koffi
│       └── settings_screen.dart     # Paramètres parcelle/capteur/IA
└── pubspec.yaml
```

---

## Palette de couleurs — #a8e063 dominante

| Variable     | Hex       | Usage                        |
|--------------|-----------|------------------------------|
| g300         | #a8e063   | Couleur principale vert clair|
| g700         | #00a05a   | Vert moyen (gradients)       |
| g800         | #007542   | Vert foncé                   |
| g900         | #002a14   | Vert très foncé (texte/fond) |
| bg           | #0a1a10   | Fond écran                   |
| accent       | #c8a96e   | Or (SMS, profil)             |
| warn         | #e05a3a   | Alerte/NON irriguier         |
| sky          | #6ec4d4   | Info (pluie, prévisions)     |

---

## Installation

```bash
# 1. Installer Flutter SDK 3.x+
# https://flutter.dev/docs/get-started/install

# 2. Cloner / ouvrir le dossier
cd water5

# 3. Installer les dépendances
flutter pub get

# 4. Lancer sur émulateur ou appareil
flutter run

# 5. Build release Android
flutter build apk --release

# 6. Build release iOS (macOS requis)
flutter build ios --release

# 7. Build Web (PC/navigateur)
flutter build web --release
```

---

## Packages utilisés

| Package              | Usage                              |
|---------------------|------------------------------------|
| google_fonts         | DM Sans + Cormorant Garamond       |
| fl_chart             | Graphiques (à étendre)             |
| provider             | State management                   |
| shared_preferences   | Préférences locales                |
| intl                 | Formatage dates                    |
| animate_do           | Animations supplémentaires         |

---

## Écrans

1. **Splash** — Logo animé W, cartes stats, bouton Demarrer
2. **Loading** — 5 étapes séquentielles (API, Arduino, ETc, ML, SMS)
3. **Dashboard** — Décision hero, 5 métriques, jauge sol, prévisions 4 jours
4. **Historique** — Stats mensuelles, liste chronologique, bande couleur
5. **Alertes** — Toggle SMS, aperçu SMS M. Koffi, 4 notifications
6. **Réglages** — Profil, parcelle, Arduino, notifications, modèle IA

---

## Intégration future

- **Backend** : FastAPI Python → endpoints REST `/decision`, `/history`, `/sensors`
- **Serial** : `flutter_libserialport` pour lecture Arduino USB
- **SMS** : `Africa's Talking` API ou `flutter_sms`
- **Notifications** : `flutter_local_notifications`
- **Charts** : `fl_chart` pour historique volumétrique

---

## Accessibilité

- `Semantics()` sur tous les éléments interactifs
- Contraste WCAG AA sur fond sombre (#0a1a10)
- Textes non scalables (textScaler: noScaling) — lisibilité fixe
- Support mode paysage (landscape) activé

---

**M. Koffi — Yamoussoukro, Côte d'Ivoire**
**Water 5 v2.0 — 2026**
