// lib/models/models.dart

// Décision d'irrigation du jour
class IrrigationDecision {
  final bool shouldIrrigate;
  final double volumeLitres;
  final double confidence;
  final double soilMoisture;
  final double tempMax;
  final double windSpeed;
  final double rainfall;
  final double et0;
  final double deficit;
  final String stage;
  final double kc;
  final DateTime date;

  const IrrigationDecision({
    required this.shouldIrrigate,
    required this.volumeLitres,
    required this.confidence,
    required this.soilMoisture,
    required this.tempMax,
    required this.windSpeed,
    required this.rainfall,
    required this.et0,
    required this.deficit,
    required this.stage,
    required this.kc,
    required this.date,
  });

  // Données de démonstration
  static IrrigationDecision get demo => IrrigationDecision(
    shouldIrrigate: true,
    volumeLitres: 820,
    confidence: 97.3,
    soilMoisture: 38,
    tempMax: 34,
    windSpeed: 2.0,
    rainfall: 0,
    et0: 5.0,
    deficit: 3.2,
    stage: 'Mi-saison',
    kc: 1.05,
    date: DateTime(2026, 3, 9),
  );
}

// Prévision météo journalière
class DayForecast {
  final String label;
  final WeatherType weather;
  final bool irrigate;
  final double volume;

  const DayForecast({
    required this.label,
    required this.weather,
    required this.irrigate,
    required this.volume,
  });

  static List<DayForecast> get demoList => [
    const DayForecast(label: "Auj.", weather: WeatherType.sunny,   irrigate: true,  volume: 820),
    const DayForecast(label: "J+1",  weather: WeatherType.cloudy,  irrigate: true,  volume: 750),
    const DayForecast(label: "J+2",  weather: WeatherType.rainy,   irrigate: false, volume: 0),
    const DayForecast(label: "J+3",  weather: WeatherType.overcast,irrigate: true,  volume: 420),
  ];
}

enum WeatherType { sunny, cloudy, rainy, overcast, storm }

// Entrée historique
class HistoryEntry {
  final DateTime date;
  final bool irrigated;
  final double volume;
  final double mlConfidence;
  final String reason;
  final String stage;
  final double kc;
  final String source;

  const HistoryEntry({
    required this.date,
    required this.irrigated,
    required this.volume,
    required this.mlConfidence,
    required this.reason,
    required this.stage,
    required this.kc,
    required this.source,
  });

  static List<HistoryEntry> get demoList => [
    HistoryEntry(date: DateTime(2026,3,9),  irrigated: true,  volume: 820,  mlConfidence: 97.3, reason: 'Deficit 3.2 mm',  stage: 'Mi-saison', kc: 1.05, source: 'DHT22'),
    HistoryEntry(date: DateTime(2026,3,8),  irrigated: false, volume: 0,    mlConfidence: 0,    reason: 'Pluie 14 mm',    stage: 'Mi-saison', kc: 1.05, source: 'Agro'),
    HistoryEntry(date: DateTime(2026,3,8),  irrigated: true,  volume: 960,  mlConfidence: 94.1, reason: 'Deficit 4.1 mm',  stage: 'Mi-saison', kc: 1.05, source: 'ML'),
    HistoryEntry(date: DateTime(2026,3,7),  irrigated: true,  volume: 1120, mlConfidence: 99.1, reason: 'Sol sec 35%',     stage: 'Mi-saison', kc: 1.05, source: 'API'),
    HistoryEntry(date: DateTime(2026,3,6),  irrigated: true,  volume: 780,  mlConfidence: 96.2, reason: 'Deficit 2.9 mm',  stage: 'Mi-saison', kc: 1.05, source: 'DHT22'),
    HistoryEntry(date: DateTime(2026,3,5),  irrigated: false, volume: 0,    mlConfidence: 0,    reason: 'Pluie 22 mm',    stage: 'Mi-saison', kc: 1.05, source: 'Agro'),
  ];
}

// Alerte notification
class AppAlert {
  final String title;
  final String message;
  final AlertType type;
  final DateTime time;
  final bool isUnread;

  const AppAlert({
    required this.title,
    required this.message,
    required this.type,
    required this.time,
    this.isUnread = false,
  });

  static List<AppAlert> get demoList => [
    AppAlert(title: 'Decision disponible', message: 'Irrigation recommandee — 820 L — Confiance ML 97%', type: AlertType.success, time: DateTime(2026,3,9,6,0),  isUnread: true),
    AppAlert(title: 'Sol sec detecte',     message: 'Humidite sol a 38% — en dessous du seuil optimal (65%)', type: AlertType.warning, time: DateTime(2026,3,9,5,45), isUnread: true),
    AppAlert(title: 'Pluie prevue J+2',    message: '14 mm prevus apres-demain — Irrigation annulee automatiquement', type: AlertType.info, time: DateTime(2026,3,9,5,45), isUnread: true),
    AppAlert(title: 'Synchronisation reussie', message: 'API Open-Meteo — 4 jours de previsions recuperes', type: AlertType.success, time: DateTime(2026,3,8,6,0),  isUnread: false),
  ];
}

enum AlertType { success, warning, info, error }
