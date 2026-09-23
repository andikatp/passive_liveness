import 'dart:async';

import 'package:flutter/material.dart';

/// Controller for orchestrating momentary high-brightness screen flashes
/// for active photometric stereo assist.
class LivenessFlashController extends ChangeNotifier {
  /// Creates a [LivenessFlashController].
  LivenessFlashController({
    this.flashColor = const Color(0xFFFFFFFF),
    this.flashDuration = const Duration(milliseconds: 150),
  });
  bool _isFlashing = false;

  /// The active screen flash background overlay color.
  Color flashColor;

  /// Duration of the screen flash burst.
  Duration flashDuration;

  /// Whether the screen flash overlay is currently active.
  bool get isFlashing => _isFlashing;

  /// Triggers a brief screen flash overlay burst.
  Future<void> triggerFlash({
    Color? color,
    Duration? duration,
  }) async {
    if (_isFlashing) return;

    if (color != null) flashColor = color;
    if (duration != null) flashDuration = duration;

    _isFlashing = true;
    notifyListeners();

    await Future<void>.delayed(flashDuration);

    _isFlashing = false;
    notifyListeners();
  }
}

/// Overlay widget that renders a full-screen solid flash overlay when
/// triggered by [controller].
class AdaptiveScreenFlashOverlay extends StatelessWidget {
  const AdaptiveScreenFlashOverlay({
    required this.controller,
    required this.child,
    super.key,
  });

  /// Controller driving screen flash state.
  final LivenessFlashController controller;

  /// Child widget (typically camera stream preview).
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        child,
        ListenableBuilder(
          listenable: controller,
          builder: (context, _) {
            if (!controller.isFlashing) {
              return const SizedBox.shrink();
            }

            return Positioned.fill(
              child: Container(
                color: controller.flashColor,
              ),
            );
          },
        ),
      ],
    );
  }
}
