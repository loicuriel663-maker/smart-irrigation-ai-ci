// lib/screens/main_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';
import 'dashboard_screen.dart';
import 'history_screen.dart';
import 'alerts_screen.dart';
import 'settings_screen.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _currentIndex = 0;

  final List<Widget> _screens = const [
    DashboardScreen(),
    HistoryScreen(),
    AlertsScreen(),
    SettingsScreen(),
  ];

  final List<String> _titles = [
    'Water 5',
    'Historique',
    'Alertes',
    'Reglages',
  ];

  @override
  Widget build(BuildContext context) {
    // Style barre système
    SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      systemNavigationBarColor: W5Colors.bg,
      systemNavigationBarIconBrightness: Brightness.light,
    ));

    return Scaffold(
      backgroundColor: W5Colors.bg,
      extendBodyBehindAppBar: false,
      appBar: _buildTopBar(),
      body: Stack(
        children: [
          // Fond orbs permanents
          _buildOrbBackground(),
          // Contenu écran
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 280),
            transitionBuilder: (child, anim) => FadeTransition(
              opacity: anim,
              child: SlideTransition(
                position: Tween<Offset>(
                  begin: const Offset(0, 0.02),
                  end: Offset.zero,
                ).animate(CurvedAnimation(parent: anim, curve: Curves.easeOut)),
                child: child,
              ),
            ),
            child: KeyedSubtree(
              key: ValueKey(_currentIndex),
              child: _screens[_currentIndex],
            ),
          ),
        ],
      ),
      bottomNavigationBar: _buildBottomNav(),
    );
  }

  PreferredSizeWidget _buildTopBar() {
    return PreferredSize(
      preferredSize: const Size.fromHeight(64),
      child: ClipRect(
        child: Container(
          decoration: BoxDecoration(
            color: W5Colors.bg.withOpacity(0.85),
            border: const Border(
              bottom: BorderSide(color: W5Colors.border, width: 1),
            ),
          ),
          child: SafeArea(
            bottom: false,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
              child: Row(
                children: [
                  // Logo
                  Container(
                    width: 32, height: 32,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(10),
                      gradient: const LinearGradient(
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                        colors: [W5Colors.g700, W5Colors.g300],
                      ),
                    ),
                    child: Center(child: Text('W',
                      style: GoogleFonts.cormorantGaramond(
                        fontSize: 18, fontWeight: FontWeight.w700,
                        color: W5Colors.g900, height: 1,
                      ))),
                  ),
                  const SizedBox(width: 10),
                  Text(
                    _titles[_currentIndex],
                    style: GoogleFonts.cormorantGaramond(
                      fontSize: 22, fontWeight: FontWeight.w600,
                      color: W5Colors.textPrimary,
                    ),
                  ),
                  const Spacer(),
                  // Date
                  if (_currentIndex == 0)
                    Text(
                      'Lun 09 Mars 2026',
                      style: GoogleFonts.dmSans(
                        fontSize: 11, color: W5Colors.textMuted,
                        fontWeight: FontWeight.w300, letterSpacing: 0.3,
                      ),
                    ),
                  // Filtre mois (historique)
                  if (_currentIndex == 1)
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: W5Colors.surface,
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: W5Colors.border),
                      ),
                      child: Text('Mars 2026',
                        style: GoogleFonts.dmSans(
                          fontSize: 12, color: W5Colors.textMuted,
                        )),
                    ),
                  // Badge alertes
                  if (_currentIndex == 2)
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 4),
                      decoration: BoxDecoration(
                        color: W5Colors.warn.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(100),
                        border: Border.all(
                            color: W5Colors.warn.withOpacity(0.35)),
                      ),
                      child: Text('3 nouvelles',
                        style: GoogleFonts.dmSans(
                          fontSize: 11, fontWeight: FontWeight.w600,
                          color: W5Colors.warn,
                        )),
                    ),
                  const SizedBox(width: 8),
                  // Bouton notif
                  GestureDetector(
                    onTap: () => setState(() => _currentIndex = 2),
                    child: Container(
                      width: 36, height: 36,
                      decoration: BoxDecoration(
                        color: W5Colors.surface,
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: W5Colors.border),
                      ),
                      child: Stack(
                        alignment: Alignment.center,
                        children: [
                          const Icon(Icons.notifications_outlined,
                              size: 18, color: W5Colors.g300),
                          Positioned(
                            top: 7, right: 7,
                            child: Container(
                              width: 7, height: 7,
                              decoration: BoxDecoration(
                                color: W5Colors.warn,
                                shape: BoxShape.circle,
                                border: Border.all(
                                    color: W5Colors.surface, width: 1.5),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  // Avatar
                  Container(
                    width: 36, height: 36,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(10),
                      gradient: const LinearGradient(
                        colors: [W5Colors.g800, W5Colors.accent],
                      ),
                    ),
                    child: Center(child: Text('MK',
                      style: GoogleFonts.dmSans(
                        fontSize: 12, fontWeight: FontWeight.w700,
                        color: Colors.white,
                      ))),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildBottomNav() {
    final items = [
      (Icons.home_outlined,        Icons.home,                 'Accueil'),
      (Icons.show_chart_outlined,  Icons.show_chart,           'Historique'),
      (Icons.notifications_outlined, Icons.notifications,      'Alertes'),
      (Icons.settings_outlined,    Icons.settings,             'Reglages'),
    ];

    return Container(
      decoration: const BoxDecoration(
        border: Border(top: BorderSide(color: W5Colors.border, width: 1)),
      ),
      child: NavigationBar(
        backgroundColor: W5Colors.bg.withOpacity(0.95),
        selectedIndex: _currentIndex,
        onDestinationSelected: (i) => setState(() => _currentIndex = i),
        height: 68,
        labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        destinations: items.map((item) {
          return NavigationDestination(
            icon: Icon(item.$1),
            selectedIcon: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              decoration: BoxDecoration(
                color: W5Colors.g300.withOpacity(0.12),
                borderRadius: BorderRadius.circular(100),
              ),
              child: Icon(item.$2, color: W5Colors.g300),
            ),
            label: item.$3,
          );
        }).toList(),
      ),
    );
  }

  Widget _buildOrbBackground() {
    return Stack(
      children: [
        Positioned(
          top: -80, left: -60,
          child: Container(
            width: 350, height: 350,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: RadialGradient(
                colors: [W5Colors.g800.withOpacity(0.2), Colors.transparent],
              ),
            ),
          ),
        ),
        Positioned(
          bottom: 100, right: -50,
          child: Container(
            width: 280, height: 280,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: RadialGradient(
                colors: [W5Colors.g700.withOpacity(0.15), Colors.transparent],
              ),
            ),
          ),
        ),
      ],
    );
  }
}
