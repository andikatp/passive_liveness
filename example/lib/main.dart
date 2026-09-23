import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:passive_liveness/passive_liveness.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Passive Liveness Example',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const LivenessHomePage(),
    );
  }
}

class LivenessHomePage extends StatefulWidget {
  const LivenessHomePage({super.key});

  @override
  State<LivenessHomePage> createState() => _LivenessHomePageState();
}

class _LivenessHomePageState extends State<LivenessHomePage> {
  final _detector = PassiveLivenessDetector();
  bool _initialized = false;
  String _status = 'Initializing engine...';

  @override
  void initState() {
    super.initState();
    unawaited(_initDetector());
  }

  Future<void> _initDetector() async {
    try {
      await _detector.initialize();
      setState(() {
        _initialized = true;
        _status = 'Detector engine ready!';
      });
    } on Exception catch (e) {
      setState(() {
        _status = 'Init error: $e';
      });
    }
  }

  /// Generates a valid in-memory PNG image byte array for testing.
  Future<Uint8List> _createSampleImageBytes() async {
    final recorder = ui.PictureRecorder();
    Canvas(recorder, const Rect.fromLTWH(0, 0, 200, 200))
      ..drawRect(
        const Rect.fromLTWH(0, 0, 200, 200),
        Paint()..color = const Color(0xFFF5D0A9),
      )
      ..drawOval(
        const Rect.fromLTWH(40, 30, 120, 140),
        Paint()..color = const Color(0xFFE5B089),
      );

    final picture = recorder.endRecording();
    final image = await picture.toImage(200, 200);
    final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    image.dispose();
    picture.dispose();
    return byteData!.buffer.asUint8List();
  }

  Future<void> _testLiveness() async {
    if (!_initialized) return;

    try {
      final imageBytes = await _createSampleImageBytes();

      final result = await _detector.detectLivenessFromImageBytes(
        imageBytes,
        boundingBox:
            const FaceBoundingBox(x: 40, y: 30, width: 120, height: 140),
      );

      setState(() {
        _status = 'Liveness Result: ${result.status.name.toUpperCase()}\n'
            'Real Score: ${(result.realScore * 100).toStringAsFixed(1)}%\n'
            'Spoof Score: ${(result.spoofScore * 100).toStringAsFixed(1)}%\n'
            'Logit Diff: ${result.logitDiff.toStringAsFixed(2)}\n'
            'Inference Time: ${result.inferenceTime.inMilliseconds}ms';
      });
    } on Exception catch (e) {
      setState(() {
        _status = 'Detection error: $e';
      });
    }
  }

  @override
  void dispose() {
    unawaited(_detector.dispose());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Passive Liveness Demo'),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(
                Icons.face,
                size: 80,
                color: Colors.blue,
              ),
              const SizedBox(height: 24),
              Text(
                _status,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 32),
              ElevatedButton.icon(
                onPressed: _initialized ? _testLiveness : null,
                icon: const Icon(Icons.play_arrow),
                label: const Text('Run Test Liveness'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
