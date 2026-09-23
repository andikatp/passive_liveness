import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:passive_liveness/passive_liveness.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('FaceProximityGate tests', () {
    const gate = FaceProximityGate();

    test(
      'Rejects face bounding box when face is too far (area ratio < 5%)',
      () {
        // 1000x1000 frame (1M px). Face 50x50 = 2500 px (0.25% ratio)
        const bbox = FaceBoundingBox(x: 100, y: 100, width: 50, height: 50);
        final result = gate.evaluate(
          boundingBox: bbox,
          frameWidth: 1000,
          frameHeight: 1000,
        );

        expect(result.isValid, isFalse);
        expect(result.status, equals(LivenessStatus.tooFar));
        expect(result.faceAreaRatio, equals(0.0025));
      },
    );

    test(
      'Rejects face bounding box when face is too close (area ratio > 85%)',
      () {
        // 1000x1000 frame (1M px). Face 950x950 = 902,500 px (90.25% ratio)
        const bbox = FaceBoundingBox(x: 25, y: 25, width: 950, height: 950);
        final result = gate.evaluate(
          boundingBox: bbox,
          frameWidth: 1000,
          frameHeight: 1000,
        );

        expect(result.isValid, isFalse);
        expect(result.status, equals(LivenessStatus.tooClose));
      },
    );

    test('Rejects face bounding box with unnatural aspect ratio', () {
      // 1000x1000 frame. Face 600x200 = area ratio 12%, aspect ratio 3.0
      const bbox = FaceBoundingBox(x: 200, y: 300, width: 600, height: 200);
      final result = gate.evaluate(
        boundingBox: bbox,
        frameWidth: 1000,
        frameHeight: 1000,
      );

      expect(result.isValid, isFalse);
      expect(result.status, equals(LivenessStatus.invalidAspectRatio));
    });

    test(
      'Accepts valid face bounding box within standard proximity and bounds',
      () {
        // 1000x1000 frame. Face 400x500 = area ratio 20%, aspect ratio 0.8
        const bbox = FaceBoundingBox(x: 300, y: 250, width: 400, height: 500);
        final result = gate.evaluate(
          boundingBox: bbox,
          frameWidth: 1000,
          frameHeight: 1000,
        );

        expect(result.isValid, isTrue);
        expect(result.status, equals(LivenessStatus.real));
        expect(result.faceAreaRatio, equals(0.20));
        expect(result.aspectRatio, equals(0.8));
      },
    );
  });

  group('LbpHogAnalyzer micro-texture tests', () {
    const analyzer = LbpHogAnalyzer();

    test('Analyzes smooth grayscale crop with low non-uniform LBP ratio', () {
      // 32x32 uniform smooth gradient
      final bytes = Uint8List(32 * 32);
      for (var y = 0; y < 32; y++) {
        for (var x = 0; x < 32; x++) {
          bytes[y * 32 + x] = ((x + y) * 2).clamp(0, 255);
        }
      }

      final result = analyzer.analyzeGrayscaleCrop(bytes, 32, 32);
      expect(result.lbpNonUniformRatio, lessThan(0.35));
      expect(result.isPrintSpoof, isFalse);
    });

    test(
      'Detects directional screen grid alignment via HOG orientation',
      () {
        // 32x32 image with vertical stripe pattern (simulating LCD sub-pixel)
        final bytes = Uint8List(32 * 32);
        for (var y = 0; y < 32; y++) {
          for (var x = 0; x < 32; x++) {
            bytes[y * 32 + x] = (x ~/ 4).isEven ? 255 : 0;
          }
        }

        final result = analyzer.analyzeGrayscaleCrop(bytes, 32, 32);
        expect(result.hogPeakDominance, greaterThanOrEqualTo(0.40));
      },
    );
  });

  group('ColorSpaceAnalyzer tests', () {
    const analyzer = ColorSpaceAnalyzer(
      maxVarianceThreshold: 160,
      minVarianceThreshold: 1.5,
    );

    test('Calculates chrominance variance for BGRA image buffer', () {
      final bytes = Uint8List(20 * 20 * 4);
      for (var i = 0; i < bytes.length; i += 4) {
        bytes[i] = 180; // B
        bytes[i + 1] = 120; // G
        bytes[i + 2] = 200; // R
        bytes[i + 3] = 255;
      }

      final buffer = LivenessImageBuffer(
        width: 20,
        height: 20,
        format: LivenessImageFormat.bgra8888,
        planes: [
          LivenessImagePlane(
            bytes: bytes,
            bytesPerRow: 20 * 4,
            bytesPerPixel: 4,
          ),
        ],
      );

      final result = analyzer.analyzeBuffer(buffer);
      expect(result.meanCb, greaterThan(0.0));
      expect(result.meanCr, greaterThan(0.0));
      // Flat solid color has 0 variance (< minVarianceThreshold 1.5)
      expect(result.chrominanceVariance, lessThan(1.5));
      expect(result.isScreenReplaySpoof, isTrue);
    });

    test('Reads RGBA image buffers without swapping red and blue channels', () {
      final bytes = Uint8List(20 * 20 * 4);
      for (var i = 0; i < bytes.length; i += 4) {
        bytes[i] = 200; // R
        bytes[i + 1] = 120; // G
        bytes[i + 2] = 180; // B
        bytes[i + 3] = 255;
      }

      final buffer = LivenessImageBuffer(
        width: 20,
        height: 20,
        format: LivenessImageFormat.rgba8888,
        planes: [
          LivenessImagePlane(
            bytes: bytes,
            bytesPerRow: 20 * 4,
            bytesPerPixel: 4,
          ),
        ],
      );

      final result = analyzer.analyzeBuffer(buffer);
      expect(result.meanCb, lessThan(result.meanCr));
      expect(result.chrominanceVariance, lessThan(1.5));
      expect(result.isScreenReplaySpoof, isTrue);
    });
  });

  group('HighResScreenAnalyzer tests', () {
    const analyzer = HighResScreenAnalyzer(
      minPatchDispersalThreshold: 4,
      maxSpecularRatioThreshold: 0.08,
    );

    test(
      'Detects flat 2D screen focal plane via low patch Laplacian dispersal',
      () {
        // 64x64 flat texture (uniform high-frequency grid across all patches)
        final bytes = Uint8List(64 * 64);
        for (var y = 0; y < 64; y++) {
          for (var x = 0; x < 64; x++) {
            bytes[y * 64 + x] = (x + y).isEven ? 200 : 50;
          }
        }

        final result = analyzer.analyzeGrayscaleCrop(bytes, 64, 64);
        expect(result.patchLaplacianDispersal, lessThan(4.0));
        expect(result.isHighResScreenSpoof, isTrue);
      },
    );

    test('Detects excessive glass screen specular glare highlights', () {
      // 64x64 image with 20% pure white (255) specular glare hotspots
      final bytes = Uint8List(64 * 64);
      for (var i = 0; i < bytes.length; i++) {
        bytes[i] = (i < bytes.length * 0.20) ? 255 : 120;
      }

      final result = analyzer.analyzeGrayscaleCrop(bytes, 64, 64);
      expect(result.specularHighlightRatio, greaterThan(0.08));
      expect(result.isHighResScreenSpoof, isTrue);
    });
  });

  group('LivenessFlashController tests', () {
    testWidgets('Toggles flash state during burst duration', (tester) async {
      final controller = LivenessFlashController(
        flashDuration: const Duration(milliseconds: 50),
      );

      expect(controller.isFlashing, isFalse);

      final flashFuture = controller.triggerFlash();
      expect(controller.isFlashing, isTrue);

      await tester.pump(const Duration(milliseconds: 60));
      await flashFuture;

      expect(controller.isFlashing, isFalse);
    });
  });

  group('Low-light detection & acceptance tests', () {
    test('LivenessResult correctly reflects meanLuminance and isLowLight', () {
      const darkResult = LivenessResult(
        isReal: true,
        status: LivenessStatus.real,
        realScore: 0.9,
        spoofScore: 0.1,
        realLogit: 2,
        spoofLogit: -2,
        logitDiff: 4,
        confidence: 4,
        threshold: 0,
        inferenceTime: Duration.zero,
        meanLuminance: 28.5,
        isLowLight: true,
      );

      expect(darkResult.meanLuminance, equals(28.5));
      expect(darkResult.isLowLight, isTrue);
      expect(darkResult.toJson()['meanLuminance'], equals(28.5));
      expect(darkResult.toJson()['isLowLight'], isTrue);
      expect(darkResult.toString(), contains('luminance: 28.5'));
      expect(darkResult.toString(), contains('isLowLight: true'));
    });
  });
}
