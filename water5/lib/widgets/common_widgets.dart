// lib/widgets/common_widgets.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';
import '../models/models.dart';

// ─── Carte de base avec fond verre ───────────────────────
class GlassCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? padding;
  final double borderRadius;
  final Color? borderColor;
  final Color? bgColor;
  final VoidCallback? onTap;

  const GlassCard({
    super.key,
    required this.child,
    this.padding,
    this.borderRadius = 20,
    this.borderColor,
    this.bgColor,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        decoration: BoxDecoration(
          color: bgColor ?? W5Colors.surface.withOpacity(0.7),
          borderRadius: BorderRadius.circular(borderRadius),
          border: Border.all(
            color: borderColor ?? W5Colors.border,
            width: 1,
          ),
        ),
        padding: padding ?? const EdgeInsets.all(18),
        child: child,
      ),
    );
  }
}

// ─── Label section ────────────────────────────────────────
class SectionLabel extends StatelessWidget {
  final String text;
  final String? action;
  final VoidCallback? onAction;

  const SectionLabel(this.text, {super.key, this.action, this.onAction});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 24, 20, 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            text.toUpperCase(),
            style: GoogleFonts.dmSans(
              fontSize: 10, fontWeight: FontWeight.w700,
              color: W5Colors.textMuted, letterSpacing: 2.5,
            ),
          ),
          if (action != null)
            GestureDetector(
              onTap: onAction,
              child: Text(
                action!,
                style: GoogleFonts.dmSans(
                  fontSize: 12, fontWeight: FontWeight.w500,
                  color: W5Colors.g300,
                ),
              ),
            ),
        ],
      ),
    );
  }
}

// ─── Chip info ────────────────────────────────────────────
class InfoChip extends StatelessWidget {
  final String label;
  final Color? color;

  const InfoChip(this.label, {super.key, this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: (color ?? W5Colors.g300).withOpacity(0.1),
        borderRadius: BorderRadius.circular(100),
        border: Border.all(color: (color ?? W5Colors.g300).withOpacity(0.2)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 5, height: 5,
            decoration: BoxDecoration(
              color: color ?? W5Colors.g300,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            label,
            style: GoogleFonts.dmSans(
              fontSize: 11, fontWeight: FontWeight.w400,
              color: W5Colors.textPrimary,
            ),
          ),
        ],
      ),
    );
  }
}

// ─── Badge confidence ML ─────────────────────────────────
class ConfidenceBadge extends StatelessWidget {
  final double value;

  const ConfidenceBadge(this.value, {super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
      decoration: BoxDecoration(
        color: W5Colors.g300.withOpacity(0.15),
        borderRadius: BorderRadius.circular(100),
        border: Border.all(color: W5Colors.g300.withOpacity(0.35)),
      ),
      child: Text(
        'ML ${value.toStringAsFixed(0)}%',
        style: GoogleFonts.dmSans(
          fontSize: 11, fontWeight: FontWeight.w600,
          color: W5Colors.g300,
        ),
      ),
    );
  }
}

// ─── Carte météo prévision ────────────────────────────────
class ForecastCard extends StatelessWidget {
  final DayForecast forecast;
  final bool isToday;

  const ForecastCard({super.key, required this.forecast, this.isToday = false});

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      width: 78,
      padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 10),
      decoration: BoxDecoration(
        color: isToday
            ? W5Colors.g800.withOpacity(0.5)
            : W5Colors.surface.withOpacity(0.7),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: isToday ? W5Colors.g300.withOpacity(0.4) : W5Colors.border,
        ),
      ),
      child: Column(
        children: [
          Text(
            forecast.label,
            style: GoogleFonts.dmSans(
              fontSize: 10, fontWeight: FontWeight.w600,
              color: isToday ? W5Colors.g300 : W5Colors.textMuted,
              letterSpacing: 1,
            ),
          ),
          const SizedBox(height: 10),
          _WeatherIcon(type: forecast.weather, size: 24),
          const SizedBox(height: 10),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
            decoration: BoxDecoration(
              color: forecast.irrigate
                  ? W5Colors.g300.withOpacity(0.2)
                  : W5Colors.warn.withOpacity(0.2),
              borderRadius: BorderRadius.circular(100),
            ),
            child: Text(
              forecast.irrigate ? 'OUI' : 'NON',
              style: GoogleFonts.dmSans(
                fontSize: 9, fontWeight: FontWeight.w700,
                color: forecast.irrigate ? W5Colors.g300 : W5Colors.warn,
                letterSpacing: 0.5,
              ),
            ),
          ),
          const SizedBox(height: 6),
          Text(
            forecast.irrigate
                ? '${forecast.volume.toInt()} L'
                : '0 L',
            style: GoogleFonts.dmSans(
              fontSize: 11, fontWeight: FontWeight.w500,
              color: W5Colors.textPrimary,
            ),
          ),
        ],
      ),
    );
  }
}

// ─── Icône météo SVG custom ───────────────────────────────
class _WeatherIcon extends StatelessWidget {
  final WeatherType type;
  final double size;

  const _WeatherIcon({required this.type, required this.size});

  @override
  Widget build(BuildContext context) {
    final icon = switch (type) {
      WeatherType.sunny   => Icons.wb_sunny_outlined,
      WeatherType.cloudy  => Icons.cloud_outlined,
      WeatherType.rainy   => Icons.grain,
      WeatherType.overcast=> Icons.cloud_queue,
      WeatherType.storm   => Icons.thunderstorm_outlined,
    };
    final color = switch (type) {
      WeatherType.sunny   => const Color(0xFFF4C842),
      WeatherType.cloudy  => const Color(0xFF90A4AE),
      WeatherType.rainy   => W5Colors.sky,
      WeatherType.overcast=> const Color(0xFFB0BEC5),
      WeatherType.storm   => W5Colors.warn,
    };
    return Icon(icon, size: size, color: color);
  }
}

// ─── Jauge humidité sol ───────────────────────────────────
class SoilGauge extends StatefulWidget {
  final double value; // 0.0 - 100.0
  final double threshold;

  const SoilGauge({super.key, required this.value, this.threshold = 65});

  @override
  State<SoilGauge> createState() => _SoilGaugeState();
}

class _SoilGaugeState extends State<SoilGauge>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<double> _anim;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this, duration: const Duration(milliseconds: 1200),
    );
    _anim = Tween<double>(begin: 0, end: widget.value / 100)
        .animate(CurvedAnimation(parent: _ctrl, curve: Curves.easeOutCubic));
    Future.delayed(const Duration(milliseconds: 400), () => _ctrl.forward());
  }

  @override
  void dispose() { _ctrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return GlassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Capteur sol', style: GoogleFonts.dmSans(
                    fontSize: 13, fontWeight: FontWeight.w600,
                    color: W5Colors.textPrimary,
                  )),
                  const SizedBox(height: 3),
                  Text('Mesure terrain reelle', style: GoogleFonts.dmSans(
                    fontSize: 11, fontWeight: FontWeight.w300,
                    color: W5Colors.textMuted,
                  )),
                ],
              ),
              AnimatedBuilder(
                animation: _anim,
                builder: (_, __) => RichText(
                  text: TextSpan(
                    children: [
                      TextSpan(
                        text: (widget.value * _anim.value / (widget.value / 100))
                            .toStringAsFixed(0),
                        style: GoogleFonts.cormorantGaramond(
                          fontSize: 40, fontWeight: FontWeight.w600,
                          color: W5Colors.g300, height: 1,
                        ),
                      ),
                      TextSpan(
                        text: '%',
                        style: GoogleFonts.dmSans(
                          fontSize: 16, color: W5Colors.textMuted,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          // Barre de progression
          Stack(
            children: [
              Container(
                height: 6,
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.06),
                  borderRadius: BorderRadius.circular(100),
                ),
              ),
              AnimatedBuilder(
                animation: _anim,
                builder: (_, __) => FractionallySizedBox(
                  widthFactor: _anim.value,
                  child: Container(
                    height: 6,
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [W5Colors.g700, W5Colors.g300],
                      ),
                      borderRadius: BorderRadius.circular(100),
                    ),
                  ),
                ),
              ),
              // Seuil
              Positioned(
                left: MediaQuery.of(context).size.width *
                    (widget.threshold / 100) * 0.82,
                top: -3,
                child: Container(
                  width: 2, height: 12,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.35),
                    borderRadius: BorderRadius.circular(1),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('Sec 10%', style: GoogleFonts.dmSans(
                fontSize: 10, color: W5Colors.textMuted,
              )),
              Text('Seuil ${widget.threshold.toInt()}%',
                style: GoogleFonts.dmSans(
                  fontSize: 10, color: W5Colors.textMuted,
                )),
              Text('Humide 90%', style: GoogleFonts.dmSans(
                fontSize: 10, color: W5Colors.textMuted,
              )),
            ],
          ),
        ],
      ),
    );
  }
}

// ─── Carte métrique capteur ───────────────────────────────
class MetricCard extends StatelessWidget {
  final IconData icon;
  final String value;
  final String unit;
  final String label;

  const MetricCard({
    super.key,
    required this.icon,
    required this.value,
    required this.unit,
    required this.label,
  });

  @override
  Widget build(BuildContext context) {
    return GlassCard(
      padding: const EdgeInsets.all(14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 28, height: 28,
            decoration: BoxDecoration(
              color: W5Colors.g300.withOpacity(0.1),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(icon, size: 14, color: W5Colors.g300),
          ),
          const SizedBox(height: 10),
          RichText(
            text: TextSpan(
              children: [
                TextSpan(
                  text: value,
                  style: GoogleFonts.cormorantGaramond(
                    fontSize: 22, fontWeight: FontWeight.w600,
                    color: W5Colors.textPrimary, height: 1,
                  ),
                ),
                TextSpan(
                  text: unit,
                  style: GoogleFonts.dmSans(
                    fontSize: 11, color: W5Colors.textMuted,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 4),
          Text(label, style: GoogleFonts.dmSans(
            fontSize: 10, color: W5Colors.textMuted,
            fontWeight: FontWeight.w500, letterSpacing: 0.3,
          )),
        ],
      ),
    );
  }
}

// ─── Élément historique ───────────────────────────────────
class HistoryItem extends StatelessWidget {
  final HistoryEntry entry;

  const HistoryItem({super.key, required this.entry});

  @override
  Widget build(BuildContext context) {
    return GlassCard(
      padding: const EdgeInsets.all(0),
      onTap: () {},
      child: ClipRRect(
        borderRadius: BorderRadius.circular(20),
        child: Row(
          children: [
            // Bande couleur gauche
            Container(
              width: 3,
              height: 80,
              color: entry.irrigated ? W5Colors.g300 : W5Colors.warn,
            ),
            const SizedBox(width: 14),
            // Icône
            Container(
              width: 42, height: 42,
              decoration: BoxDecoration(
                color: entry.irrigated
                    ? W5Colors.g300.withOpacity(0.12)
                    : W5Colors.warn.withOpacity(0.12),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(
                entry.irrigated ? Icons.check : Icons.close,
                size: 18,
                color: entry.irrigated ? W5Colors.g300 : W5Colors.warn,
              ),
            ),
            const SizedBox(width: 12),
            // Contenu
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    entry.irrigated ? 'Irrigation effectuee' : 'Pas d\'irrigation',
                    style: GoogleFonts.dmSans(
                      fontSize: 14, fontWeight: FontWeight.w600,
                      color: W5Colors.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 3),
                  Text(
                    entry.irrigated
                        ? 'ML ${entry.mlConfidence.toStringAsFixed(0)}% — ${entry.reason}'
                        : entry.reason,
                    style: GoogleFonts.dmSans(
                      fontSize: 12, fontWeight: FontWeight.w300,
                      color: W5Colors.textMuted,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      _HTag(entry.stage, color: W5Colors.g300),
                      const SizedBox(width: 6),
                      _HTag('Kc ${entry.kc}', color: W5Colors.accent2),
                      const SizedBox(width: 6),
                      _HTag(entry.source, color: W5Colors.sky),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(width: 12),
            // Volume
            Padding(
              padding: const EdgeInsets.only(right: 16),
              child: Text(
                entry.irrigated ? '${entry.volume.toInt()} L' : '0 L',
                style: GoogleFonts.cormorantGaramond(
                  fontSize: 20, fontWeight: FontWeight.w600,
                  color: entry.irrigated ? W5Colors.g300 : W5Colors.warn,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _HTag extends StatelessWidget {
  final String label;
  final Color color;

  const _HTag(this.label, {required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(100),
      ),
      child: Text(
        label,
        style: GoogleFonts.dmSans(
          fontSize: 10, fontWeight: FontWeight.w600,
          color: color, letterSpacing: 0.3,
        ),
      ),
    );
  }
}

// ─── Élément alerte ───────────────────────────────────────
class AlertItem extends StatelessWidget {
  final AppAlert alert;

  const AlertItem({super.key, required this.alert});

  Color get _color => switch (alert.type) {
    AlertType.success => W5Colors.g300,
    AlertType.warning => W5Colors.accent2,
    AlertType.info    => W5Colors.sky,
    AlertType.error   => W5Colors.warn,
  };

  IconData get _icon => switch (alert.type) {
    AlertType.success => Icons.check,
    AlertType.warning => Icons.warning_amber_outlined,
    AlertType.info    => Icons.grain,
    AlertType.error   => Icons.error_outline,
  };

  @override
  Widget build(BuildContext context) {
    return GlassCard(
      onTap: () {},
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 44, height: 44,
            decoration: BoxDecoration(
              color: _color.withOpacity(0.12),
              borderRadius: BorderRadius.circular(13),
            ),
            child: Icon(_icon, size: 20, color: _color),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(alert.title, style: GoogleFonts.dmSans(
                  fontSize: 13, fontWeight: FontWeight.w600,
                  color: W5Colors.textPrimary,
                )),
                const SizedBox(height: 4),
                Text(alert.message, style: GoogleFonts.dmSans(
                  fontSize: 12, fontWeight: FontWeight.w300,
                  color: W5Colors.textMuted, height: 1.5,
                )),
                const SizedBox(height: 6),
                Text(
                  '${alert.time.hour.toString().padLeft(2,'0')}:${alert.time.minute.toString().padLeft(2,'0')}',
                  style: GoogleFonts.dmSans(
                    fontSize: 11, color: W5Colors.g300.withOpacity(0.4),
                    fontWeight: FontWeight.w300,
                  ),
                ),
              ],
            ),
          ),
          if (alert.isUnread)
            Container(
              width: 8, height: 8, margin: const EdgeInsets.only(top: 3),
              decoration: const BoxDecoration(
                color: W5Colors.accent, shape: BoxShape.circle,
              ),
            ),
        ],
      ),
    );
  }
}
