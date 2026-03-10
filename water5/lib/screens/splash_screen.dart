// lib/screens/splash_screen.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with TickerProviderStateMixin {
  late AnimationController _logoCtrl;
  late AnimationController _ringCtrl;
  late AnimationController _contentCtrl;
  late Animation<double> _logoScale;
  late Animation<double> _logoFade;
  late Animation<double> _ringExpand;
  late Animation<double> _contentSlide;

  @override
  void initState() {
    super.initState();
    _logoCtrl = AnimationController(
      vsync: this, duration: const Duration(milliseconds: 900),
    );
    _ringCtrl = AnimationController(
      vsync: this, duration: const Duration(seconds: 4),
    )..repeat(reverse: true);
    _contentCtrl = AnimationController(
      vsync: this, duration: const Duration(milliseconds: 800),
    );

    _logoScale = Tween<double>(begin: 0.6, end: 1.0).animate(
      CurvedAnimation(parent: _logoCtrl, curve: Curves.elasticOut),
    );
    _logoFade = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _logoCtrl,
          curve: const Interval(0.0, 0.5, curve: Curves.easeOut)),
    );
    _ringExpand = Tween<double>(begin: 0.85, end: 1.0).animate(
      CurvedAnimation(parent: _ringCtrl, curve: Curves.easeInOut),
    );
    _contentSlide = Tween<double>(begin: 30, end: 0).animate(
      CurvedAnimation(parent: _contentCtrl, curve: Curves.easeOutCubic),
    );

    Future.delayed(const Duration(milliseconds: 200), () {
      _logoCtrl.forward();
      Future.delayed(const Duration(milliseconds: 400), () {
        _contentCtrl.forward();
      });
    });
  }

  @override
  void dispose() {
    _logoCtrl.dispose();
    _ringCtrl.dispose();
    _contentCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: W5Colors.bg,
      body: Stack(
        children: [
          // Fond orbs
          _buildOrbBackground(),
          // Anneaux pulsants
          _buildRings(),
          // Contenu centré
          SafeArea(
            child: Column(
              children: [
                const Spacer(flex: 2),
                // Logo animé
                AnimatedBuilder(
                  animation: _logoCtrl,
                  builder: (_, __) => Transform.scale(
                    scale: _logoScale.value,
                    child: Opacity(
                      opacity: _logoFade.value,
                      child: _buildLogoMark(),
                    ),
                  ),
                ),
                const SizedBox(height: 28),
                // Nom + tagline
                AnimatedBuilder(
                  animation: _contentCtrl,
                  builder: (_, child) => Transform.translate(
                    offset: Offset(0, _contentSlide.value),
                    child: Opacity(opacity: _contentCtrl.value, child: child),
                  ),
                  child: _buildTitles(),
                ),
                const SizedBox(height: 44),
                // Cartes stats
                AnimatedBuilder(
                  animation: _contentCtrl,
                  builder: (_, child) => Transform.translate(
                    offset: Offset(0, _contentSlide.value * 1.5),
                    child: Opacity(
                      opacity: (_contentCtrl.value - 0.2).clamp(0.0, 1.0),
                      child: child,
                    ),
                  ),
                  child: _buildStatCards(),
                ),
                const Spacer(flex: 2),
                // Bouton
                AnimatedBuilder(
                  animation: _contentCtrl,
                  builder: (_, child) => Opacity(
                    opacity: (_contentCtrl.value - 0.4).clamp(0.0, 1.0),
                    child: child,
                  ),
                  child: _buildStartButton(),
                ),
                const SizedBox(height: 24),
                // Version
                Text(
                  'v2.0 — M. Koffi — Yamoussoukro, CI',
                  style: GoogleFonts.dmSans(
                    fontSize: 11, color: W5Colors.textMuted.withOpacity(0.6),
                    fontWeight: FontWeight.w300, letterSpacing: 0.5,
                  ),
                ),
                const SizedBox(height: 40),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildOrbBackground() {
    return Stack(
      children: [
        Positioned(
          top: -100, left: -80,
          child: Container(
            width: 400, height: 400,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: RadialGradient(
                colors: [W5Colors.g800.withOpacity(0.3), Colors.transparent],
              ),
            ),
          ),
        ),
        Positioned(
          bottom: -80, right: -60,
          child: Container(
            width: 320, height: 320,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: RadialGradient(
                colors: [W5Colors.g700.withOpacity(0.2), Colors.transparent],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildRings() {
    return AnimatedBuilder(
      animation: _ringCtrl,
      builder: (_, __) {
        return Center(
          child: Stack(
            alignment: Alignment.center,
            children: [
              for (int i = 0; i < 3; i++)
                Opacity(
                  opacity: (0.3 - i * 0.08) * _ringExpand.value,
                  child: Container(
                    width: (280 + i * 130).toDouble(),
                    height: (280 + i * 130).toDouble(),
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: W5Colors.g300.withOpacity(0.15),
                      ),
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildLogoMark() {
    return Container(
      width: 76, height: 76,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [W5Colors.g700, W5Colors.g300],
        ),
        boxShadow: [
          BoxShadow(
            color: W5Colors.g300.withOpacity(0.35),
            blurRadius: 40,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Center(
        child: Text(
          'W',
          style: GoogleFonts.cormorantGaramond(
            fontSize: 38, fontWeight: FontWeight.w700,
            color: W5Colors.g900, height: 1,
          ),
        ),
      ),
    );
  }

  Widget _buildTitles() {
    return Column(
      children: [
        RichText(
          textAlign: TextAlign.center,
          text: TextSpan(
            children: [
              TextSpan(
                text: 'Water',
                style: GoogleFonts.cormorantGaramond(
                  fontSize: 58, fontWeight: FontWeight.w300,
                  color: W5Colors.textPrimary, letterSpacing: -2, height: 1,
                ),
              ),
              TextSpan(
                text: '5',
                style: GoogleFonts.cormorantGaramond(
                  fontSize: 58, fontWeight: FontWeight.w700,
                  foreground: Paint()
                    ..shader = const LinearGradient(
                      colors: [W5Colors.g300, W5Colors.accent2],
                    ).createShader(const Rect.fromLTWH(0, 0, 60, 60)),
                  letterSpacing: -2, height: 1,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 10),
        Text(
          'IRRIGATION INTELLIGENTE PAR IA',
          style: GoogleFonts.dmSans(
            fontSize: 11, fontWeight: FontWeight.w400,
            color: W5Colors.textMuted, letterSpacing: 3,
          ),
        ),
      ],
    );
  }

  Widget _buildStatCards() {
    final stats = [
      ('200', 'm²\nTomate'),
      ('100%', 'Precision\nIA'),
      ('1096', 'Jours de\ndonnees'),
    ];
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: stats.map((s) {
        return Container(
          margin: const EdgeInsets.symmetric(horizontal: 6),
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          decoration: BoxDecoration(
            color: W5Colors.surface.withOpacity(0.6),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: W5Colors.border),
          ),
          child: Column(
            children: [
              Text(s.$1, style: GoogleFonts.cormorantGaramond(
                fontSize: 26, fontWeight: FontWeight.w600,
                color: W5Colors.g300, height: 1,
              )),
              const SizedBox(height: 4),
              Text(s.$2, textAlign: TextAlign.center,
                style: GoogleFonts.dmSans(
                  fontSize: 10, fontWeight: FontWeight.w400,
                  color: W5Colors.textMuted, height: 1.4,
                )),
            ],
          ),
        );
      }).toList(),
    );
  }

  Widget _buildStartButton() {
    return GestureDetector(
      onTap: () => Navigator.pushReplacementNamed(context, '/loading'),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 48, vertical: 18),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(100),
          gradient: const LinearGradient(
            colors: [W5Colors.g700, W5Colors.g500],
          ),
          boxShadow: [
            BoxShadow(
              color: W5Colors.g700.withOpacity(0.45),
              blurRadius: 30,
              offset: const Offset(0, 10),
            ),
          ],
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'Demarrer',
              style: GoogleFonts.dmSans(
                fontSize: 16, fontWeight: FontWeight.w600,
                color: W5Colors.g900, letterSpacing: 0.5,
              ),
            ),
            const SizedBox(width: 12),
            Container(
              width: 24, height: 24,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: W5Colors.g900.withOpacity(0.25),
              ),
              child: const Icon(Icons.arrow_forward,
                  size: 14, color: W5Colors.g900),
            ),
          ],
        ),
      ),
    );
  }
}
