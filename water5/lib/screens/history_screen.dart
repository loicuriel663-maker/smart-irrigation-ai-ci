// lib/screens/history_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';
import '../models/models.dart';
import '../widgets/common_widgets.dart';

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final entries = HistoryEntry.demoList;

    return CustomScrollView(
      physics: const BouncingScrollPhysics(),
      slivers: [
        // Stats du mois
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
            child: Row(
              children: [
                _StatMini('22',     'Irrigations', W5Colors.g300),
                const SizedBox(width: 10),
                _StatMini('18 460', 'Litres total', W5Colors.accent2),
                const SizedBox(width: 10),
                _StatMini('99.5%',  'Precision IA', W5Colors.sky),
              ],
            ),
          ),
        ),

        // Liste
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _DateSep("Aujourd'hui — 09 Mars"),
                HistoryItem(entry: entries[0]),
                _DateSep("Hier — 08 Mars"),
                HistoryItem(entry: entries[1]),
                const SizedBox(height: 10),
                HistoryItem(entry: entries[2]),
                _DateSep("07 Mars"),
                HistoryItem(entry: entries[3]),
                _DateSep("06 Mars"),
                HistoryItem(entry: entries[4]),
                const SizedBox(height: 10),
                HistoryItem(entry: entries[5]),
                const SizedBox(height: 30),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class _StatMini extends StatelessWidget {
  final String value;
  final String label;
  final Color color;
  const _StatMini(this.value, this.label, this.color);

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: GlassCard(
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 12),
        child: Column(
          children: [
            Text(value, style: GoogleFonts.cormorantGaramond(
              fontSize: 24, fontWeight: FontWeight.w600,
              color: color, height: 1,
            )),
            const SizedBox(height: 4),
            Text(label, textAlign: TextAlign.center,
              style: GoogleFonts.dmSans(
                fontSize: 10, color: W5Colors.textMuted,
                fontWeight: FontWeight.w500,
              )),
          ],
        ),
      ),
    );
  }
}

class _DateSep extends StatelessWidget {
  final String text;
  const _DateSep(this.text);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 18, bottom: 10),
      child: Text(text.toUpperCase(), style: GoogleFonts.dmSans(
        fontSize: 10, fontWeight: FontWeight.w700,
        color: W5Colors.textMuted, letterSpacing: 2,
      )),
    );
  }
}
