// lib/screens/dashboard_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';
import '../models/models.dart';
import '../widgets/common_widgets.dart';

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final decision = IrrigationDecision.demo;
    final forecasts = DayForecast.demoList;

    return CustomScrollView(
      physics: const BouncingScrollPhysics(),
      slivers: [
        // ── Hero Decision ──────────────────────────────
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'DECISION DU JOUR — 09 MARS 2026',
                  style: GoogleFonts.dmSans(
                    fontSize: 10, fontWeight: FontWeight.w600,
                    color: W5Colors.textMuted, letterSpacing: 2.5,
                  ),
                ),
                const SizedBox(height: 14),
                _HeroCard(decision: decision),
              ],
            ),
          ),
        ),

        // ── Métriques capteurs ─────────────────────────
        SliverToBoxAdapter(
          child: SectionLabel(
            'Capteurs et meteo',
            action: 'Actualiser',
          ),
        ),
        SliverToBoxAdapter(
          child: SizedBox(
            height: 120,
            child: ListView(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              scrollDirection: Axis.horizontal,
              physics: const BouncingScrollPhysics(),
              children: [
                MetricCard(
                  icon: Icons.water_drop_outlined,
                  value: '38', unit: '%', label: 'Sol DHT22',
                ),
                const SizedBox(width: 10),
                MetricCard(
                  icon: Icons.thermostat_outlined,
                  value: '34', unit: '°', label: 'Temp max',
                ),
                const SizedBox(width: 10),
                MetricCard(
                  icon: Icons.air_outlined,
                  value: '2.0', unit: ' m/s', label: 'Vent',
                ),
                const SizedBox(width: 10),
                MetricCard(
                  icon: Icons.grain,
                  value: '0', unit: ' mm', label: 'Pluie',
                ),
                const SizedBox(width: 10),
                MetricCard(
                  icon: Icons.wb_sunny_outlined,
                  value: '5.0', unit: ' mm', label: 'ET0',
                ),
              ],
            ),
          ),
        ),

        // ── Jauge sol ──────────────────────────────────
        const SliverToBoxAdapter(
          child: SectionLabel('Humidite du sol'),
        ),
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: SoilGauge(value: 38, threshold: 65),
          ),
        ),

        // ── Prévisions 4 jours ─────────────────────────
        const SliverToBoxAdapter(
          child: SectionLabel('Previsions 4 jours'),
        ),
        SliverToBoxAdapter(
          child: SizedBox(
            height: 140,
            child: ListView.separated(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              scrollDirection: Axis.horizontal,
              physics: const BouncingScrollPhysics(),
              itemCount: forecasts.length,
              separatorBuilder: (_, __) => const SizedBox(width: 10),
              itemBuilder: (_, i) => ForecastCard(
                forecast: forecasts[i],
                isToday: i == 0,
              ),
            ),
          ),
        ),

        const SliverToBoxAdapter(child: SizedBox(height: 24)),
      ],
    );
  }
}

// ─── Carte Hero Décision ─────────────────────────────────
class _HeroCard extends StatelessWidget {
  final IrrigationDecision decision;

  const _HeroCard({required this.decision});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(26),
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            W5Colors.g800.withOpacity(0.6),
            W5Colors.g700.withOpacity(0.3),
          ],
        ),
        border: Border.all(color: W5Colors.g300.withOpacity(0.25)),
        boxShadow: [
          BoxShadow(
            color: W5Colors.g700.withOpacity(0.15),
            blurRadius: 40, offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Titre + badge
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Recommandation IA',
                      style: GoogleFonts.dmSans(
                        fontSize: 10, color: W5Colors.textMuted,
                        letterSpacing: 1.5, fontWeight: FontWeight.w500,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      decision.shouldIrrigate
                          ? 'Arroser\naujourd\'hui'
                          : 'Ne pas\narroser',
                      style: GoogleFonts.cormorantGaramond(
                        fontSize: 32, fontWeight: FontWeight.w700,
                        color: decision.shouldIrrigate
                            ? W5Colors.g300 : W5Colors.warn,
                        height: 1.1,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'Volume recommande : ${decision.volumeLitres.toInt()} litres',
                      style: GoogleFonts.dmSans(
                        fontSize: 13, fontWeight: FontWeight.w300,
                        color: W5Colors.textSecond,
                      ),
                    ),
                  ],
                ),
              ),
              ConfidenceBadge(decision.confidence),
            ],
          ),
          const SizedBox(height: 18),
          // Chips
          Wrap(
            spacing: 8, runSpacing: 8,
            children: [
              InfoChip('Deficit ${decision.deficit} mm'),
              InfoChip('${decision.tempMax.toInt()} deg max'),
              InfoChip(decision.stage),
              InfoChip('Kc ${decision.kc}'),
            ],
          ),
        ],
      ),
    );
  }
}
