// lib/screens/alerts_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';
import '../models/models.dart';
import '../widgets/common_widgets.dart';

class AlertsScreen extends StatefulWidget {
  const AlertsScreen({super.key});

  @override
  State<AlertsScreen> createState() => _AlertsScreenState();
}

class _AlertsScreenState extends State<AlertsScreen> {
  bool _smsEnabled = true;

  @override
  Widget build(BuildContext context) {
    final alerts = AppAlert.demoList;

    return CustomScrollView(
      physics: const BouncingScrollPhysics(),
      slivers: [
        // Carte SMS toggle
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
            child: Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(20),
                gradient: LinearGradient(
                  colors: [
                    W5Colors.accent.withOpacity(0.12),
                    W5Colors.accent2.withOpacity(0.06),
                  ],
                ),
                border: Border.all(color: W5Colors.accent.withOpacity(0.25)),
              ),
              child: Row(
                children: [
                  Container(
                    width: 48, height: 48,
                    decoration: BoxDecoration(
                      color: W5Colors.accent.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: const Icon(Icons.sms_outlined,
                        size: 22, color: W5Colors.accent2),
                  ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('SMS quotidien', style: GoogleFonts.dmSans(
                          fontSize: 14, fontWeight: FontWeight.w600,
                          color: W5Colors.textPrimary,
                        )),
                        const SizedBox(height: 3),
                        Text('M. Koffi — +225 07 XX XX XX',
                          style: GoogleFonts.dmSans(
                            fontSize: 12, fontWeight: FontWeight.w300,
                            color: W5Colors.textMuted,
                          )),
                      ],
                    ),
                  ),
                  GestureDetector(
                    onTap: () => setState(() => _smsEnabled = !_smsEnabled),
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 250),
                      width: 48, height: 26,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(100),
                        color: _smsEnabled
                            ? W5Colors.g700
                            : W5Colors.surface3,
                        border: Border.all(
                          color: _smsEnabled
                              ? W5Colors.g300.withOpacity(0.3)
                              : W5Colors.border,
                        ),
                      ),
                      child: AnimatedAlign(
                        duration: const Duration(milliseconds: 250),
                        alignment: _smsEnabled
                            ? Alignment.centerRight
                            : Alignment.centerLeft,
                        child: Padding(
                          padding: const EdgeInsets.all(3),
                          child: Container(
                            width: 20, height: 20,
                            decoration: const BoxDecoration(
                              shape: BoxShape.circle, color: Colors.white,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),

        // Aperçu SMS
        const SliverToBoxAdapter(
          child: SectionLabel('Dernier SMS envoye'),
        ),
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: GlassCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        width: 36, height: 36,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(10),
                          gradient: const LinearGradient(
                            colors: [W5Colors.accent, W5Colors.warn],
                          ),
                        ),
                        child: Center(child: Text('MK',
                          style: GoogleFonts.dmSans(
                            fontSize: 13, fontWeight: FontWeight.w700,
                            color: Colors.white,
                          ))),
                      ),
                      const SizedBox(width: 10),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('M. Koffi', style: GoogleFonts.dmSans(
                            fontSize: 13, fontWeight: FontWeight.w600,
                          )),
                          Text("Envoye — 06h00 — aujourd'hui",
                            style: GoogleFonts.dmSans(
                              fontSize: 11, color: W5Colors.textMuted,
                              fontWeight: FontWeight.w300,
                            )),
                        ],
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  Container(
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: W5Colors.g300.withOpacity(0.06),
                      borderRadius: const BorderRadius.only(
                        topRight: Radius.circular(14),
                        bottomLeft: Radius.circular(14),
                        bottomRight: Radius.circular(14),
                      ),
                      border: Border.all(
                          color: W5Colors.g300.withOpacity(0.15)),
                    ),
                    child: Text(
                      'Water5 — 09/03/2026\n'
                      'Stade    : MI-SAISON\n'
                      'Kc       : 1.0500\n'
                      'Decision : ARROSER\n'
                      'Volume   : 820 L\n'
                      'Sol : 38% | Pluie : 0 mm\n'
                      'ET0 : 5.00 mm\n\n'
                      'Previsions :\n'
                      '  J+1 : OUI  750 L\n'
                      '  J+2 : NON  0 L\n'
                      '  J+3 : OUI  420 L',
                      style: GoogleFonts.dmMono(
                        fontSize: 12, color: W5Colors.textPrimary,
                        height: 1.7,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),

        // Liste alertes
        const SliverToBoxAdapter(
          child: SectionLabel('Notifications'),
        ),
        SliverPadding(
          padding: const EdgeInsets.fromLTRB(20, 0, 20, 30),
          sliver: SliverList.separated(
            itemCount: alerts.length,
            separatorBuilder: (_, __) => const SizedBox(height: 10),
            itemBuilder: (_, i) => AlertItem(alert: alerts[i]),
          ),
        ),
      ],
    );
  }
}
