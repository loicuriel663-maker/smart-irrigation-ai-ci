// lib/theme/app_theme.dart
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

// ═══════════════════════════════════════════════
//  PALETTE WATER 5 — dominante #a8e063 vert clair
// ═══════════════════════════════════════════════
class W5Colors {
  // Verts
  static const Color g900 = Color(0xFF002A14);
  static const Color g800 = Color(0xFF007542);
  static const Color g700 = Color(0xFF00A05A);
  static const Color g500 = Color(0xFF3ABF7E);
  static const Color g300 = Color(0xFFa8e063); // couleur principale
  static const Color g200 = Color(0xFFCAED94);
  static const Color g100 = Color(0xFFD4F0B0);
  static const Color g050 = Color(0xFFF0FAF0);

  // Accents
  static const Color accent  = Color(0xFFC8A96E);
  static const Color accent2 = Color(0xFFE8C98E);
  static const Color warn    = Color(0xFFE05A3A);
  static const Color sky     = Color(0xFF6EC4D4);

  // Surfaces
  static const Color bg       = Color(0xFF0A1A10);
  static const Color surface  = Color(0xFF132A1C);
  static const Color surface2 = Color(0xFF1A3524);
  static const Color surface3 = Color(0xFF203D2A);

  // Texte
  static const Color textPrimary = Color(0xFFF0FAF0);
  static const Color textSecond  = Color(0xFFa8e063);
  static const Color textMuted   = Color(0xFF6A9A6A);
  static const Color textDim     = Color(0xFF3D6B3D);

  // Borders
  static const Color border  = Color(0x2Ea8e063);
  static const Color border2 = Color(0x18FFFFFF);
}

class AppTheme {
  static ThemeData get theme => ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    scaffoldBackgroundColor: W5Colors.bg,
    colorScheme: const ColorScheme.dark(
      primary: W5Colors.g300,
      secondary: W5Colors.g500,
      surface: W5Colors.surface,
      error: W5Colors.warn,
      onPrimary: W5Colors.g900,
      onSecondary: W5Colors.textPrimary,
      onSurface: W5Colors.textPrimary,
    ),
    textTheme: GoogleFonts.dmSansTextTheme(ThemeData.dark().textTheme).copyWith(
      displayLarge: GoogleFonts.cormorantGaramond(
        fontSize: 56, fontWeight: FontWeight.w300,
        color: W5Colors.textPrimary, letterSpacing: -2,
      ),
      displayMedium: GoogleFonts.cormorantGaramond(
        fontSize: 40, fontWeight: FontWeight.w600,
        color: W5Colors.textPrimary, letterSpacing: -1,
      ),
      displaySmall: GoogleFonts.cormorantGaramond(
        fontSize: 32, fontWeight: FontWeight.w600,
        color: W5Colors.textPrimary,
      ),
      headlineLarge: GoogleFonts.cormorantGaramond(
        fontSize: 26, fontWeight: FontWeight.w600,
        color: W5Colors.textPrimary,
      ),
      headlineMedium: GoogleFonts.dmSans(
        fontSize: 18, fontWeight: FontWeight.w600,
        color: W5Colors.textPrimary,
      ),
      bodyLarge: GoogleFonts.dmSans(
        fontSize: 15, fontWeight: FontWeight.w400,
        color: W5Colors.textPrimary,
      ),
      bodyMedium: GoogleFonts.dmSans(
        fontSize: 13, fontWeight: FontWeight.w400,
        color: W5Colors.textPrimary,
      ),
      bodySmall: GoogleFonts.dmSans(
        fontSize: 11, fontWeight: FontWeight.w400,
        color: W5Colors.textMuted,
      ),
      labelSmall: GoogleFonts.dmSans(
        fontSize: 10, fontWeight: FontWeight.w600,
        color: W5Colors.textMuted, letterSpacing: 1.5,
      ),
    ),
    appBarTheme: AppBarTheme(
      backgroundColor: W5Colors.bg.withOpacity(0.85),
      elevation: 0,
      centerTitle: false,
      iconTheme: const IconThemeData(color: W5Colors.g300),
      titleTextStyle: GoogleFonts.cormorantGaramond(
        fontSize: 22, fontWeight: FontWeight.w600,
        color: W5Colors.textPrimary,
      ),
    ),
    navigationBarTheme: NavigationBarThemeData(
      backgroundColor: W5Colors.bg.withOpacity(0.92),
      indicatorColor: W5Colors.g300.withOpacity(0.15),
      labelTextStyle: WidgetStateProperty.resolveWith((states) {
        final active = states.contains(WidgetState.selected);
        return GoogleFonts.dmSans(
          fontSize: 10, fontWeight: FontWeight.w600,
          color: active ? W5Colors.g300 : W5Colors.textMuted,
        );
      }),
      iconTheme: WidgetStateProperty.resolveWith((states) {
        final active = states.contains(WidgetState.selected);
        return IconThemeData(
          color: active ? W5Colors.g300 : W5Colors.textMuted,
          size: 22,
        );
      }),
    ),
    cardTheme: CardThemeData(
      color: W5Colors.surface,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(20),
        side: const BorderSide(color: W5Colors.border, width: 1),
      ),
    ),
  );
}
