// lib/screens/loading_screen.dart
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';

class LoadingScreen extends StatefulWidget {
  const LoadingScreen({super.key});

  @override
  State<LoadingScreen> createState() => _LoadingScreenState();
}

class _LoadingScreenState extends State<LoadingScreen>
    with TickerProviderStateMixin {
  late AnimationController _spinCtrl;
  int _currentStep = -1;
  final List<String> _steps = [
    'Connexion API Open-Meteo',
    'Lecture capteur Arduino DHT22',
    'Calcul ETc et deficit hydrique',
    'Modele Random Forest — prediction',
    'Generation decision et SMS M. Koffi',
  ];

  @override
  void initState() {
    super.initState();
    _spinCtrl = AnimationController(
      vsync: this, duration: const Duration(seconds: 1),
    )..repeat();
    _runSequence();
  }

  void _runSequence() async {
    for (int i = 0; i < _steps.length; i++) {
      await Future.delayed(Duration(milliseconds: 600 + (i * 120)));
      if (!mounted) return;
      setState(() => _currentStep = i);
    }
    await Future.delayed(const Duration(milliseconds: 700));
    if (!mounted) return;
    Navigator.pushReplacementNamed(context, '/main');
  }

  @override
  void dispose() {
    _spinCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: W5Colors.bg,
      body: Stack(
        children: [
          // Background
          Positioned(
            top: -100, left: -80,
            child: Container(
              width: 400, height: 400,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(
                  colors: [W5Colors.g800.withOpacity(0.25), Colors.transparent],
                ),
              ),
            ),
          ),
          SafeArea(
            child: Center(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    // Logo spinner
                    SizedBox(
                      width: 96, height: 96,
                      child: Stack(
                        alignment: Alignment.center,
                        children: [
                          AnimatedBuilder(
                            animation: _spinCtrl,
                            builder: (_, __) => Transform.rotate(
                              angle: _spinCtrl.value * 6.28318,
                              child: Container(
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  border: Border.all(
                                    color: Colors.transparent,
                                    width: 2,
                                  ),
                                  gradient: const SweepGradient(
                                    colors: [
                                      W5Colors.g300,
                                      Colors.transparent,
                                      Colors.transparent,
                                    ],
                                    stops: [0.0, 0.3, 1.0],
                                  ),
                                ),
                              ),
                            ),
                          ),
                          Container(
                            width: 80, height: 80,
                            decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(22),
                              gradient: const LinearGradient(
                                begin: Alignment.topLeft,
                                end: Alignment.bottomRight,
                                colors: [W5Colors.g700, W5Colors.g300],
                              ),
                              boxShadow: [
                                BoxShadow(
                                  color: W5Colors.g300.withOpacity(0.25),
                                  blurRadius: 30,
                                ),
                              ],
                            ),
                            child: Center(
                              child: Text('W',
                                style: GoogleFonts.cormorantGaramond(
                                  fontSize: 40, fontWeight: FontWeight.w700,
                                  color: W5Colors.g900, height: 1,
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 40),
                    Text(
                      'Analyse en cours...',
                      style: GoogleFonts.cormorantGaramond(
                        fontSize: 28, fontWeight: FontWeight.w300,
                        color: W5Colors.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 40),
                    // Étapes
                    ..._steps.asMap().entries.map((e) {
                      final isDone   = e.key < _currentStep;
                      final isActive = e.key == _currentStep;
                      return AnimatedContainer(
                        duration: const Duration(milliseconds: 300),
                        margin: const EdgeInsets.only(bottom: 12),
                        padding: const EdgeInsets.symmetric(
                          horizontal: 18, vertical: 14,
                        ),
                        decoration: BoxDecoration(
                          color: isActive
                              ? W5Colors.g300.withOpacity(0.08)
                              : W5Colors.surface.withOpacity(
                                  isDone ? 0.5 : 0.4),
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(
                            color: isActive
                                ? W5Colors.g300.withOpacity(0.4)
                                : isDone
                                    ? W5Colors.g300.withOpacity(0.2)
                                    : W5Colors.border2,
                          ),
                        ),
                        child: Row(
                          children: [
                            AnimatedContainer(
                              duration: const Duration(milliseconds: 300),
                              width: 8, height: 8,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: isActive || isDone
                                    ? W5Colors.g300
                                    : W5Colors.textMuted.withOpacity(0.4),
                                boxShadow: isActive
                                    ? [BoxShadow(
                                        color: W5Colors.g300.withOpacity(0.5),
                                        blurRadius: 8,
                                      )]
                                    : null,
                              ),
                            ),
                            const SizedBox(width: 14),
                            Expanded(
                              child: Text(
                                e.value,
                                style: GoogleFonts.dmSans(
                                  fontSize: 13, fontWeight: FontWeight.w500,
                                  color: isActive || isDone
                                      ? W5Colors.textPrimary
                                      : W5Colors.textMuted.withOpacity(0.5),
                                ),
                              ),
                            ),
                            if (isDone)
                              Text('✓', style: TextStyle(
                                color: W5Colors.g300, fontSize: 13,
                              )),
                          ],
                        ),
                      );
                    }),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
