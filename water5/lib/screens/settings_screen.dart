// lib/screens/settings_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';
import '../widgets/common_widgets.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool _smsEnabled = true;
  bool _notifEnabled = true;

  @override
  Widget build(BuildContext context) {
    return CustomScrollView(
      physics: const BouncingScrollPhysics(),
      slivers: [
        // Profil
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
            child: GlassCard(
              child: Row(
                children: [
                  Container(
                    width: 60, height: 60,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(18),
                      gradient: const LinearGradient(
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                        colors: [W5Colors.g700, W5Colors.accent],
                      ),
                    ),
                    child: Center(child: Text('K',
                      style: GoogleFonts.cormorantGaramond(
                        fontSize: 30, fontWeight: FontWeight.w700,
                        color: Colors.white, height: 1,
                      ))),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('M. Koffi', style: GoogleFonts.dmSans(
                          fontSize: 18, fontWeight: FontWeight.w600,
                          color: W5Colors.textPrimary,
                        )),
                        const SizedBox(height: 3),
                        Text("Yamoussoukro, Cote d'Ivoire",
                          style: GoogleFonts.dmSans(
                            fontSize: 13, fontWeight: FontWeight.w300,
                            color: W5Colors.textMuted,
                          )),
                      ],
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: W5Colors.g300.withOpacity(0.12),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(
                          color: W5Colors.g300.withOpacity(0.25)),
                    ),
                    child: Text('Pro', style: GoogleFonts.dmSans(
                      fontSize: 11, fontWeight: FontWeight.w700,
                      color: W5Colors.g300,
                    )),
                  ),
                ],
              ),
            ),
          ),
        ),

        // Groupe Parcelle
        const SliverToBoxAdapter(
          child: SectionLabel('Parcelle'),
        ),
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: _SettingsGroup(items: [
              _SettingItem(
                icon: Icons.eco_outlined,
                name: 'Culture',
                value: 'Tomate — Kc 1.05',
              ),
              _SettingItem(
                icon: Icons.crop_square_outlined,
                name: 'Surface',
                value: '200 m²',
              ),
              _SettingItem(
                icon: Icons.location_on_outlined,
                name: 'Localisation',
                value: '6.8205 N — 5.2767 W',
              ),
            ]),
          ),
        ),

        // Groupe Capteur
        const SliverToBoxAdapter(
          child: SectionLabel('Capteur Arduino'),
        ),
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: _SettingsGroup(items: [
              _SettingItem(
                icon: Icons.memory_outlined,
                name: 'Port serie',
                value: 'COM3 — 9600 baud',
              ),
              _SettingItem(
                icon: Icons.water_drop_outlined,
                name: 'Seuil sol sec',
                value: '45% — alerte LED rouge',
              ),
            ]),
          ),
        ),

        // Groupe Notifications
        const SliverToBoxAdapter(
          child: SectionLabel('Notifications'),
        ),
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: _SettingsGroup(items: [
              _SettingToggleItem(
                icon: Icons.sms_outlined,
                name: 'SMS quotidien',
                value: 'Active — 06h00',
                enabled: _smsEnabled,
                onToggle: (v) => setState(() => _smsEnabled = v),
              ),
              _SettingToggleItem(
                icon: Icons.notifications_outlined,
                name: 'Notifications push',
                value: 'Activees',
                enabled: _notifEnabled,
                onToggle: (v) => setState(() => _notifEnabled = v),
              ),
              _SettingItem(
                icon: Icons.phone_outlined,
                name: 'Numero SMS',
                value: '+225 07 XX XX XX',
              ),
            ]),
          ),
        ),

        // Groupe Modèle IA
        const SliverToBoxAdapter(
          child: SectionLabel('Modele IA'),
        ),
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: _SettingsGroup(items: [
              _SettingBadgeItem(
                icon: Icons.psychology_outlined,
                name: 'Modele actif',
                value: 'Random Forest — 2022-2024',
                badge: 'Actif',
              ),
              _SettingItem(
                icon: Icons.bar_chart_outlined,
                name: 'Performance',
                value: 'Precision 100% — R2 0.9954',
              ),
            ]),
          ),
        ),

        // Version
        SliverToBoxAdapter(
          child: Padding(
            padding: const EdgeInsets.all(32),
            child: Column(
              children: [
                Text('Water 5 — v2.0.0', style: GoogleFonts.dmSans(
                  fontSize: 12, color: W5Colors.textMuted,
                )),
                const SizedBox(height: 4),
                Text('AgroIrri CI — Yamoussoukro',
                  style: GoogleFonts.dmSans(
                    fontSize: 11,
                    color: W5Colors.g300.withOpacity(0.3),
                  )),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

// ─── Groupe settings ─────────────────────────────────────
class _SettingsGroup extends StatelessWidget {
  final List<Widget> items;
  const _SettingsGroup({required this.items});

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(18),
      child: Container(
        decoration: BoxDecoration(
          color: W5Colors.surface.withOpacity(0.7),
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: W5Colors.border),
        ),
        child: Column(
          children: items.asMap().entries.map((e) {
            final isLast = e.key == items.length - 1;
            return Column(
              children: [
                e.value,
                if (!isLast)
                  Divider(
                    height: 1, thickness: 1,
                    color: W5Colors.border,
                    indent: 60,
                  ),
              ],
            );
          }).toList(),
        ),
      ),
    );
  }
}

// ─── Item standard ───────────────────────────────────────
class _SettingItem extends StatelessWidget {
  final IconData icon;
  final String name;
  final String value;

  const _SettingItem({required this.icon, required this.name, required this.value});

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: () {},
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          child: Row(
            children: [
              Container(
                width: 34, height: 34,
                decoration: BoxDecoration(
                  color: W5Colors.g300.withOpacity(0.08),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(icon, size: 16, color: W5Colors.g300),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(name, style: GoogleFonts.dmSans(
                      fontSize: 14, fontWeight: FontWeight.w500,
                      color: W5Colors.textPrimary,
                    )),
                    Text(value, style: GoogleFonts.dmSans(
                      fontSize: 12, fontWeight: FontWeight.w300,
                      color: W5Colors.textMuted,
                    )),
                  ],
                ),
              ),
              Icon(Icons.chevron_right, size: 18, color: W5Colors.textDim),
            ],
          ),
        ),
      ),
    );
  }
}

// ─── Item avec toggle ─────────────────────────────────────
class _SettingToggleItem extends StatelessWidget {
  final IconData icon;
  final String name;
  final String value;
  final bool enabled;
  final ValueChanged<bool> onToggle;

  const _SettingToggleItem({
    required this.icon, required this.name, required this.value,
    required this.enabled, required this.onToggle,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      child: Row(
        children: [
          Container(
            width: 34, height: 34,
            decoration: BoxDecoration(
              color: W5Colors.g300.withOpacity(0.08),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, size: 16, color: W5Colors.g300),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(name, style: GoogleFonts.dmSans(
                  fontSize: 14, fontWeight: FontWeight.w500,
                  color: W5Colors.textPrimary,
                )),
                Text(value, style: GoogleFonts.dmSans(
                  fontSize: 12, fontWeight: FontWeight.w300,
                  color: W5Colors.textMuted,
                )),
              ],
            ),
          ),
          GestureDetector(
            onTap: () => onToggle(!enabled),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 250),
              width: 44, height: 24,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(100),
                color: enabled ? W5Colors.g700 : W5Colors.surface3,
                border: Border.all(
                  color: enabled
                      ? W5Colors.g300.withOpacity(0.3)
                      : W5Colors.border,
                ),
              ),
              child: AnimatedAlign(
                duration: const Duration(milliseconds: 250),
                alignment: enabled ? Alignment.centerRight : Alignment.centerLeft,
                child: Padding(
                  padding: const EdgeInsets.all(2.5),
                  child: Container(
                    width: 19, height: 19,
                    decoration: const BoxDecoration(
                      shape: BoxShape.circle, color: Colors.white,
                      boxShadow: [BoxShadow(color: Colors.black26, blurRadius: 4)],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ─── Item avec badge ─────────────────────────────────────
class _SettingBadgeItem extends StatelessWidget {
  final IconData icon;
  final String name;
  final String value;
  final String badge;

  const _SettingBadgeItem({
    required this.icon, required this.name,
    required this.value, required this.badge,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      child: Row(
        children: [
          Container(
            width: 34, height: 34,
            decoration: BoxDecoration(
              color: W5Colors.g300.withOpacity(0.08),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, size: 16, color: W5Colors.g300),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(name, style: GoogleFonts.dmSans(
                  fontSize: 14, fontWeight: FontWeight.w500,
                  color: W5Colors.textPrimary,
                )),
                Text(value, style: GoogleFonts.dmSans(
                  fontSize: 12, fontWeight: FontWeight.w300,
                  color: W5Colors.textMuted,
                )),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: W5Colors.g300.withOpacity(0.15),
              borderRadius: BorderRadius.circular(100),
              border: Border.all(color: W5Colors.g300.withOpacity(0.3)),
            ),
            child: Text(badge, style: GoogleFonts.dmSans(
              fontSize: 10, fontWeight: FontWeight.w700,
              color: W5Colors.g300,
            )),
          ),
        ],
      ),
    );
  }
}
