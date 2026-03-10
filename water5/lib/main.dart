// lib/main.dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'theme/app_theme.dart';
import 'screens/splash_screen.dart';
import 'screens/loading_screen.dart';
import 'screens/main_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  // Mode immersif — plein écran
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);

  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.light,
    systemNavigationBarColor: Color(0xFF0A1A10),
    systemNavigationBarIconBrightness: Brightness.light,
  ));

  runApp(const Water5App());
}

class Water5App extends StatelessWidget {
  const Water5App({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Water 5',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.theme,

      // Routes nommées
      initialRoute: '/',
      routes: {
        '/':        (_) => const SplashScreen(),
        '/loading': (_) => const LoadingScreen(),
        '/main':    (_) => const MainScreen(),
      },

      // Transition entre routes
      onGenerateRoute: (settings) {
        final routes = {
          '/':        const SplashScreen(),
          '/loading': const LoadingScreen(),
          '/main':    const MainScreen(),
        };
        final page = routes[settings.name];
        if (page == null) return null;
        return PageRouteBuilder(
          settings: settings,
          pageBuilder: (_, __, ___) => page,
          transitionsBuilder: (_, anim, __, child) => FadeTransition(
            opacity: CurvedAnimation(parent: anim, curve: Curves.easeOut),
            child: child,
          ),
          transitionDuration: const Duration(milliseconds: 350),
        );
      },

      // Builder global pour MediaQuery
      builder: (context, child) {
        return MediaQuery(
          data: MediaQuery.of(context).copyWith(
            textScaler: TextScaler.noScaling,
          ),
          child: child!,
        );
      },
    );
  }
}
