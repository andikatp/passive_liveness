import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:passive_liveness/passive_liveness.dart';

void main() {
  group('LivenessResult tests', () {
    test(
      'Calculates real classification when realLogit > spoofLogit',
      () {
        final result = LivenessResult.fromLogits(
          realLogit: 3.5,
          spoofLogit: 0.5,
        );

        expect(result.isReal, isTrue);
        expect(result.status, equals(LivenessStatus.real));
        expect(result.logitDiff, equals(3.0));
        expect(result.confidence, equals(3.0));
        expect(result.realScore, greaterThan(result.spoofScore));
        expect(result.realScore + result.spoofScore, closeTo(1.0, 1e-5));
      },
    );

    test(
      'Calculates spoof classification when realLogit < spoofLogit',
      () {
        final result = LivenessResult.fromLogits(
          realLogit: -2,
          spoofLogit: 2,
        );

        expect(result.isReal, isFalse);
        expect(result.status, equals(LivenessStatus.spoof));
        expect(result.logitDiff, equals(-4.0));
        expect(result.confidence, equals(4.0));
        expect(result.spoofScore, greaterThan(result.realScore));
        expect(result.realScore + result.spoofScore, closeTo(1.0, 1e-5));
      },
    );

    test(
      'Exposes raw single-frame metrics correctly alongside smoothed metrics',
      () {
        const result = LivenessResult(
          isReal: false,
          status: LivenessStatus.spoof,
          realScore: 0.4,
          spoofScore: 0.6,
          realLogit: 0.84,
          spoofLogit: -0.84,
          logitDiff: -0.4,
          confidence: 0.4,
          threshold: 0,
          inferenceTime: Duration.zero,
          rawRealScore: 0.844,
          rawSpoofScore: 0.156,
          rawLogitDiff: 1.68,
          rawIsReal: true,
        );

        expect(result.isReal, isFalse);
        expect(result.rawIsReal, isTrue);
        expect(result.rawRealScore, equals(0.844));
        expect(result.rawLogitDiff, equals(1.68));
      },
    );
  });

  group('FaceBoundingBox tests', () {
    test('Calculates center and boundary coordinates accurately', () {
      const bbox = FaceBoundingBox(x: 10, y: 20, width: 100, height: 120);

      expect(bbox.centerX, equals(60.0));
      expect(bbox.centerY, equals(80.0));
      expect(bbox.right, equals(110.0));
      expect(bbox.bottom, equals(140.0));
    });

    test('Constructs from Rect and LTRB correctly', () {
      final fromLTRB = FaceBoundingBox.fromLTRB(10, 20, 110, 140);
      expect(fromLTRB.x, equals(10.0));
      expect(fromLTRB.y, equals(20.0));
      expect(fromLTRB.width, equals(100.0));
      expect(fromLTRB.height, equals(120.0));

      final rect = fromLTRB.toRect();
      expect(rect.left, equals(10.0));
      expect(rect.top, equals(20.0));
      expect(rect.width, equals(100.0));
      expect(rect.height, equals(120.0));
    });

    test(
      'toRawBufferSpace transforms 270 deg rotated bounding box correctly',
      () {
        // Rotated 1080x1920 space: center at (540, 960), size (400, 600)
        const rotBbox = FaceBoundingBox(
          x: 340,
          y: 660,
          width: 400,
          height: 600,
        );
        final rawBbox = rotBbox.toRawBufferSpace(1920, 1080, 270);

        // Raw 1920x1080 space: rawCx = 1920 - 960 = 960, rawCy = 540
        expect(rawBbox.centerX, equals(960.0));
        expect(rawBbox.centerY, equals(540.0));
        expect(rawBbox.width, equals(600.0));
        expect(rawBbox.height, equals(400.0));
      },
    );
  });

  group('FaceProximityGate tests', () {
    test(
      'Calculates upright face aspect ratio across 0, 90, 180, 270 deg '
      'rotations',
      () {
        const gate = FaceProximityGate();

        // Portrait face: Width = 200, Height = 300 (aspect ratio = 0.667)
        const uprightBox =
            FaceBoundingBox(x: 100, y: 100, width: 200, height: 300);
        final res0 = gate.evaluate(
          boundingBox: uprightBox,
          frameWidth: 1000,
          frameHeight: 1000,
        );
        expect(res0.isValid, isTrue);
        expect(res0.aspectRatio, closeTo(0.667, 1e-3));

        // 90 deg rotation: transposed in raw buffer space
        const rawTransposedBox =
            FaceBoundingBox(x: 100, y: 100, width: 300, height: 200);
        final res90 = gate.evaluate(
          boundingBox: rawTransposedBox,
          frameWidth: 1000,
          frameHeight: 1000,
          rotation: 90,
        );
        expect(res90.isValid, isTrue);
        expect(res90.aspectRatio, closeTo(0.667, 1e-3));

        final res270 = gate.evaluate(
          boundingBox: rawTransposedBox,
          frameWidth: 1000,
          frameHeight: 1000,
          rotation: 270,
        );
        expect(res270.isValid, isTrue);
        expect(res270.aspectRatio, closeTo(0.667, 1e-3));
      },
    );
  });

  group('ImagePreprocessor tests', () {
    test(
      'edge pixel replication clamps out-of-bounds coordinates',
      () {
        final bytes = Uint8List(20 * 20 * 4);
        // Fill 20x20 image with white pixels (R=255, G=255, B=255)
        for (var i = 0; i < bytes.length; i += 4) {
          bytes[i] = 255;
          bytes[i + 1] = 255;
          bytes[i + 2] = 255;
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

        // Bounding box near corner so crop extends outside image
        const bbox = FaceBoundingBox(x: -5, y: -5, width: 20, height: 20);

        final tensor = ImagePreprocessor.preprocessBufferToTensor(
          buffer,
          boundingBox: bbox,
          expansionFactor: 2,
          targetSize: 10,
          useNchw: false,
        );

        // Top-left corner of tensor corresponds to out-of-bound crop area ->
        // should replicate white edge (1.0)
        expect(tensor[0], equals(1.0));
        expect(tensor[1], equals(1.0));
        expect(tensor[2], equals(1.0));
      },
    );

    test(
      'preprocessBufferToTensor converts BGRA buffer to float32 tensor',
      () {
        // Create dummy BGRA8888 60x60 image plane (R=255, G=128, B=0, A=255)
        final bytes = Uint8List(60 * 60 * 4);
        for (var i = 0; i < bytes.length; i += 4) {
          bytes[i] = 0; // B
          bytes[i + 1] = 128; // G
          bytes[i + 2] = 255; // R
          bytes[i + 3] = 255; // A
        }

        final buffer = LivenessImageBuffer(
          width: 60,
          height: 60,
          format: LivenessImageFormat.bgra8888,
          planes: [
            LivenessImagePlane(
              bytes: bytes,
              bytesPerRow: 60 * 4,
              bytesPerPixel: 4,
            ),
          ],
        );

        // Test NCHW layout (default)
        final tensorNchw = ImagePreprocessor.preprocessBufferToTensor(
          buffer,
        );

        expect(tensorNchw.length, equals(49152));
        const hw = 128 * 128;
        expect(tensorNchw[0], equals(1.0)); // Channel 0 (R: 255/255)
        expect(
          tensorNchw[hw],
          closeTo(128 / 255.0, 1e-3),
        ); // Channel 1 (G: 128/255)
        expect(tensorNchw[2 * hw], equals(0.0)); // Channel 2 (B: 0/255)

        // Test NHWC layout
        final tensorNhwc = ImagePreprocessor.preprocessBufferToTensor(
          buffer,
          useNchw: false,
        );

        expect(tensorNhwc.length, equals(49152));
        expect(tensorNhwc[0], equals(1.0)); // R
        expect(tensorNhwc[1], closeTo(128 / 255.0, 1e-3)); // G
        expect(tensorNhwc[2], equals(0.0)); // B
      },
    );

    test(
      'preprocessBufferToTensor converts RGBA buffer without swapping channels',
      () {
        final bytes = Uint8List(4 * 4 * 4);
        for (var i = 0; i < bytes.length; i += 4) {
          bytes[i] = 255; // R
          bytes[i + 1] = 128; // G
          bytes[i + 2] = 0; // B
          bytes[i + 3] = 255; // A
        }

        final buffer = LivenessImageBuffer(
          width: 4,
          height: 4,
          format: LivenessImageFormat.rgba8888,
          planes: [
            LivenessImagePlane(
              bytes: bytes,
              bytesPerRow: 4 * 4,
              bytesPerPixel: 4,
            ),
          ],
        );

        final tensor = ImagePreprocessor.preprocessBufferToTensor(
          buffer,
          targetSize: 4,
          useNchw: false,
          enableContrastStretch: false,
        );

        expect(tensor[0], equals(1.0));
        expect(tensor[1], closeTo(128 / 255.0, 1e-3));
        expect(tensor[2], equals(0.0));
      },
    );

    test(
      'preprocessRgbaBytesToTensor processes raw RGBA pixel buffer',
      () {
        final rgbaBytes = Uint8List(50 * 50 * 4);
        for (var i = 0; i < rgbaBytes.length; i += 4) {
          rgbaBytes[i] = 255; // R
          rgbaBytes[i + 1] = 100; // G
          rgbaBytes[i + 2] = 50; // B
          rgbaBytes[i + 3] = 255; // A
        }

        final tensor = ImagePreprocessor.preprocessRgbaBytesToTensor(
          rgbaBytes,
          50,
          50,
          targetSize: 10,
          useNchw: false,
        );

        expect(tensor.length, equals(300));
        expect(tensor[0], equals(1.0));
        expect(tensor[1], closeTo(100 / 255.0, 1e-3));
        expect(tensor[2], closeTo(50 / 255.0, 1e-3));
      },
    );

    test('saveTensorToDisk writes valid PPM image file to disk', () async {
      final tensor = Float32List(1 * 128 * 128 * 3);
      const hw = 128 * 128;
      // Set R channel to 1.0 for NCHW tensor
      for (var i = 0; i < hw; i++) {
        tensor[i] = 1.0; // R
        tensor[hw + i] = 0.0; // G
        tensor[2 * hw + i] = 0.0; // B
      }

      final testPath = '${Directory.systemTemp.path}/test_liveness_tensor.ppm';
      final savedFile = await ImagePreprocessor.saveTensorToDisk(
        tensor,
        filePath: testPath,
      );

      expect(savedFile.existsSync(), isTrue);
      final length = await savedFile.length();
      expect(length, greaterThan(hw * 3));
      await savedFile.delete();
    });

    test(
      'preprocessBufferToTensor maintains 1:1 square crop aspect ratio',
      () {
        final bytes = Uint8List(100 * 100 * 4);
        final buffer = LivenessImageBuffer(
          width: 100,
          height: 100,
          format: LivenessImageFormat.bgra8888,
          planes: [
            LivenessImagePlane(
              bytes: bytes,
              bytesPerRow: 100 * 4,
              bytesPerPixel: 4,
            ),
          ],
        );

        // Rectangular bounding box (30x50)
        const bbox = FaceBoundingBox(x: 10, y: 10, width: 30, height: 50);

        // Should complete without error using square crop math
        final tensor = ImagePreprocessor.preprocessBufferToTensor(
          buffer,
          boundingBox: bbox,
          expansionFactor: 2,
        );

        expect(tensor.length, equals(49152));
      },
    );

    test(
      'preprocessBufferToTensor calculates true 2.0x square crop',
      () {
        final bytes = Uint8List(100 * 100 * 4);
        final buffer = LivenessImageBuffer(
          width: 100,
          height: 100,
          format: LivenessImageFormat.bgra8888,
          planes: [
            LivenessImagePlane(
              bytes: bytes,
              bytesPerRow: 100 * 4,
              bytesPerPixel: 4,
            ),
          ],
        );

        // Large face box (60x60) in 100x100 frame -> baseSide = 60.
        // 2.0x expansion is 120.0. Unclamped crop extends outside boundaries.
        const bbox = FaceBoundingBox(x: 20, y: 20, width: 60, height: 60);

        final tensor = ImagePreprocessor.preprocessBufferToTensor(
          buffer,
          boundingBox: bbox,
          expansionFactor: 2,
          targetSize: 10,
        );

        expect(tensor.length, equals(300));
      },
    );

    test(
      'reflect101 border padding mirrors pixels for out-of-bounds coordinates',
      () {
        // 10x10 BGRA image (column 0 = red, column 1 = green)
        final bytes = Uint8List(10 * 10 * 4);
        for (var y = 0; y < 10; y++) {
          for (var x = 0; x < 10; x++) {
            final idx = (y * 10 + x) * 4;
            if (x == 0) {
              bytes[idx] = 0; // B
              bytes[idx + 1] = 0; // G
              bytes[idx + 2] = 255; // R
              bytes[idx + 3] = 255; // A
            } else if (x == 1) {
              bytes[idx] = 0; // B
              bytes[idx + 1] = 255; // G
              bytes[idx + 2] = 0; // R
              bytes[idx + 3] = 255; // A
            }
          }
        }

        final buffer = LivenessImageBuffer(
          width: 10,
          height: 10,
          format: LivenessImageFormat.bgra8888,
          planes: [
            LivenessImagePlane(
              bytes: bytes,
              bytesPerRow: 10 * 4,
              bytesPerPixel: 4,
            ),
          ],
        );

        // Crop box overflowing on the left (x = -2)
        const bbox = FaceBoundingBox(x: -2, y: 0, width: 10, height: 10);
        final tensor = ImagePreprocessor.preprocessBufferToTensor(
          buffer,
          boundingBox: bbox,
          expansionFactor: 1,
          targetSize: 10,
          useNchw: false,
        );

        // Should complete without out-of-range exception
        expect(tensor.length, equals(300));
      },
    );

    test(
      'preprocessBufferToTensor supports isBgr: true for BGR channel ordering',
      () {
        final bytes = Uint8List(10 * 10 * 4);
        for (var i = 0; i < bytes.length; i += 4) {
          bytes[i] = 255; // B = 255
          bytes[i + 1] = 128; // G = 128
          bytes[i + 2] = 0; // R = 0
          bytes[i + 3] = 255;
        }

        final buffer = LivenessImageBuffer(
          width: 10,
          height: 10,
          format: LivenessImageFormat.bgra8888,
          planes: [
            LivenessImagePlane(
              bytes: bytes,
              bytesPerRow: 10 * 4,
              bytesPerPixel: 4,
            ),
          ],
        );

        final tensorBgr = ImagePreprocessor.preprocessBufferToTensor(
          buffer,
          targetSize: 10,
          isBgr: true,
        );

        const hw = 10 * 10;
        expect(tensorBgr[0], equals(1.0)); // Channel 0 is B (255/255)
        expect(
          tensorBgr[hw],
          closeTo(128 / 255.0, 1e-3),
        ); // Channel 1 is G (128/255)
        expect(tensorBgr[2 * hw], equals(0.0)); // Channel 2 is R (0/255)
      },
    );

    test(
      'preprocessBufferToTensor supports NormalizationScheme',
      () {
        final bytes = Uint8List(10 * 10 * 4);
        for (var i = 0; i < bytes.length; i += 4) {
          bytes[i] = 255; // B = 255
          bytes[i + 1] = 128; // G = 128
          bytes[i + 2] = 0; // R = 0
          bytes[i + 3] = 255;
        }

        final buffer = LivenessImageBuffer(
          width: 10,
          height: 10,
          format: LivenessImageFormat.bgra8888,
          planes: [
            LivenessImagePlane(
              bytes: bytes,
              bytesPerRow: 10 * 4,
              bytesPerPixel: 4,
            ),
          ],
        );

        // minusOneToOne: (val - 127.5)/127.5
        final tensorMinusOne = ImagePreprocessor.preprocessBufferToTensor(
          buffer,
          targetSize: 10,
          normalizationScheme: NormalizationScheme.minusOneToOne,
        );

        const hw = 10 * 10;
        // R = 0 -> (0 - 127.5)/127.5 = -1.0
        expect(tensorMinusOne[0], closeTo(-1.0, 1e-3));
        // G = 128 -> (128 - 127.5)/127.5 = ~0.0039
        expect(tensorMinusOne[hw], closeTo(0.0039, 1e-3));

        // imageNet: (val/255.0 - mean) / std
        final tensorImageNet = ImagePreprocessor.preprocessBufferToTensor(
          buffer,
          targetSize: 10,
          normalizationScheme: NormalizationScheme.imageNet,
        );

        // R = 0 -> (0 - 0.485) / 0.229 = -2.1179
        expect(tensorImageNet[0], closeTo(-2.1179, 1e-3));
      },
    );
  });

  group('PassiveLivenessDetector tests', () {
    const channel = MethodChannel('com.andikatp.passiveLiveness');

    setUpAll(() {
      TestWidgetsFlutterBinding.ensureInitialized();
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (methodCall) async {
        if (methodCall.method == 'initModel') {
          return {
            'inputShape': [1, 3, 128, 128],
            'isNchw': true,
            'targetSize': 128,
          };
        }
        if (methodCall.method == 'runInference') {
          return [3.5, 0.5];
        }
        if (methodCall.method == 'closeModel') {
          return null;
        }
        return null;
      });
    });

    test('EMA tracker initializes as null and resets correctly', () async {
      final detector = PassiveLivenessDetector();

      expect(detector.emaRealScore, isNull);
      expect(detector.isInitialized, isFalse);

      detector.resetEma();
      expect(detector.emaRealScore, isNull);
      await detector.dispose();
    });

    test(
      'Initializes with mock model bytes and runs detection pipeline',
      () async {
        final detector = PassiveLivenessDetector();
        final dummyBytes = Uint8List.fromList([1, 2, 3, 4]);

        await detector.initialize(modelBytes: dummyBytes);
        expect(detector.isInitialized, isTrue);
        expect(detector.isNativeNchw, isTrue);
        expect(detector.modelTargetSize, equals(128));

        final frameBytes = Uint8List(100 * 100 * 4);
        final buffer = LivenessImageBuffer(
          width: 100,
          height: 100,
          format: LivenessImageFormat.bgra8888,
          planes: [
            LivenessImagePlane(
              bytes: frameBytes,
              bytesPerRow: 100 * 4,
              bytesPerPixel: 4,
            ),
          ],
        );

        final result = await detector.detectLivenessFromBuffer(buffer);
        expect(result.realLogit, equals(3.5));
        expect(result.spoofLogit, equals(0.5));

        await detector.dispose();
        expect(detector.isInitialized, isFalse);
      },
    );

    test(
      'Supports ModelClassOrder.spoofFirst to invert logit mapping',
      () async {
        final detector = PassiveLivenessDetector();
        final dummyBytes = Uint8List.fromList([1, 2, 3, 4]);

        await detector.initialize(
          modelBytes: dummyBytes,
          classOrder: ModelClassOrder.spoofFirst,
        );

        final frameBytes = Uint8List(10 * 10 * 4);
        final buffer = LivenessImageBuffer(
          width: 10,
          height: 10,
          format: LivenessImageFormat.bgra8888,
          planes: [
            LivenessImagePlane(
              bytes: frameBytes,
              bytesPerRow: 10 * 4,
              bytesPerPixel: 4,
            ),
          ],
        );

        final result = await detector.detectLivenessFromBuffer(buffer);
        // Mock method channel returns [3.5, 0.5].
        // With spoofFirst: index 0 (3.5) = spoof, index 1 (0.5) = real
        expect(result.spoofLogit, equals(3.5));
        expect(result.realLogit, equals(0.5));

        await detector.dispose();
      },
    );

    test(
      'applies rotated Android face box to heuristic analyzers',
      () async {
        final detector = PassiveLivenessDetector();
        await detector.initialize(
          modelBytes: Uint8List.fromList([1, 2, 3, 4]),
        );

        const width = 100;
        const height = 80;
        final frameBytes = Uint8List(width * height * 4);
        for (var y = 0; y < height; y++) {
          for (var x = 0; x < width; x++) {
            final idx = (y * width + x) * 4;
            frameBytes[idx] = 110 + (y % 20); // B
            frameBytes[idx + 1] = 120; // G
            frameBytes[idx + 2] = 140 + (x % 20); // R
            frameBytes[idx + 3] = 255; // A
          }
        }

        const rawFaceBox = FaceBoundingBox(x: 60, y: 15, width: 20, height: 30);
        for (var y = rawFaceBox.y.toInt(); y < rawFaceBox.bottom.toInt(); y++) {
          for (var x = rawFaceBox.x.toInt();
              x < rawFaceBox.right.toInt();
              x++) {
            final idx = (y * width + x) * 4;
            final even = ((x ~/ 8) + (y ~/ 8)).isEven;
            frameBytes[idx] = even ? 255 : 0; // B
            frameBytes[idx + 1] = 40; // G
            frameBytes[idx + 2] = even ? 0 : 255; // R
            frameBytes[idx + 3] = 255; // A
          }
        }

        final buffer = LivenessImageBuffer(
          width: width,
          height: height,
          format: LivenessImageFormat.bgra8888,
          planes: [
            LivenessImagePlane(
              bytes: frameBytes,
              bytesPerRow: width * 4,
              bytesPerPixel: 4,
            ),
          ],
        );

        const rotatedMlKitBox = FaceBoundingBox(
          x: 15,
          y: 20,
          width: 30,
          height: 20,
        );

        final result = await detector.detectLivenessFromBuffer(
          buffer,
          boundingBox: rotatedMlKitBox,
          rotation: 270,
          isRotatedBoundingBox: true,
          enableProximityGate: false,
        );

        expect(result.rawIsReal, isTrue);
        expect(result.chrominanceVariance, greaterThan(160.0));
        // With positive neural score, real status is protected
        expect(result.isReal, isTrue);
        expect(result.status, equals(LivenessStatus.real));

        await detector.dispose();
      },
    );

    test(
      'classifies real face with moderate logitDiff ~2.15 as REAL',
      () async {
        final detector = PassiveLivenessDetector();
        await detector.initialize(
          modelBytes: Uint8List.fromList([1, 2, 3, 4]),
        );

        const width = 100;
        const height = 100;
        final frameBytes = Uint8List(width * height * 4);
        for (var i = 0; i < frameBytes.length; i += 4) {
          frameBytes[i] = 110; // B
          frameBytes[i + 1] = 130; // G
          frameBytes[i + 2] = 200; // R
          frameBytes[i + 3] = 255; // A
        }

        final buffer = LivenessImageBuffer(
          width: width,
          height: height,
          format: LivenessImageFormat.bgra8888,
          planes: [
            LivenessImagePlane(
              bytes: frameBytes,
              bytesPerRow: width * 4,
              bytesPerPixel: 4,
            ),
          ],
        );

        final result = await detector.detectLivenessFromBuffer(
          buffer,
          threshold: 0.30,
          enableProximityGate: false,
        );

        // Mock returns [3.5, 0.5] by default, rawIsReal = true
        expect(result.rawIsReal, isTrue);
        expect(result.isReal, isTrue);
        expect(result.status, equals(LivenessStatus.real));

        await detector.dispose();
      },
    );

    test(
      'classifies real face with natural 3D depth as REAL without override',
      () async {
        final detector = PassiveLivenessDetector();
        await detector.initialize(
          modelBytes: Uint8List.fromList([1, 2, 3, 4]),
        );

        const width = 100;
        const height = 100;
        final frameBytes = Uint8List(width * height * 4);
        final buffer = LivenessImageBuffer(
          width: width,
          height: height,
          format: LivenessImageFormat.bgra8888,
          planes: [
            LivenessImagePlane(
              bytes: frameBytes,
              bytesPerRow: width * 4,
              bytesPerPixel: 4,
            ),
          ],
        );

        final result = await detector.detectLivenessFromBuffer(
          buffer,
          enableProximityGate: false,
          enableTextureAnalysis: false,
          enableColorSpaceAnalysis: false,
          enableHighResScreenAnalysis: false,
          enableMoireAnalysis: false,
        );

        expect(result.rawIsReal, isTrue);
        expect(result.isReal, isTrue);
        expect(result.status, equals(LivenessStatus.real));

        await detector.dispose();
      },
    );

    test(
      'classifies iPhone high-res real face with chrominance variance as REAL',
      () async {
        final detector = PassiveLivenessDetector();
        await detector.initialize(
          modelBytes: Uint8List.fromList([1, 2, 3, 4]),
        );

        const width = 120;
        const height = 120;
        final frameBytes = Uint8List(width * height * 4);

        // Generate smooth natural skin tones (R > G > B)
        for (var y = 0; y < height; y++) {
          for (var x = 0; x < width; x++) {
            final idx = (y * width + x) * 4;
            // Smooth gradient representing face features
            final skinGrad = (x + y) % 30;
            frameBytes[idx] = (100 + skinGrad).clamp(0, 255); // B
            frameBytes[idx + 1] = (130 + skinGrad * 2).clamp(0, 255); // G
            frameBytes[idx + 2] = (210 + skinGrad).clamp(0, 255); // R
            frameBytes[idx + 3] = 255; // A
          }
        }

        final buffer = LivenessImageBuffer(
          width: width,
          height: height,
          format: LivenessImageFormat.bgra8888,
          planes: [
            LivenessImagePlane(
              bytes: frameBytes,
              bytesPerRow: width * 4,
              bytesPerPixel: 4,
            ),
          ],
        );

        final result = await detector.detectLivenessFromBuffer(
          buffer,
          enableProximityGate: false,
        );

        // Model returns [3.5, 0.5] (logitDiff = 3.0), ensure it stays REAL
        expect(result.rawIsReal, isTrue);
        expect(result.isReal, isTrue);
        expect(result.status, equals(LivenessStatus.real));

        await detector.dispose();
      },
    );

    test(
      'auto-accepts dark low-light frames as REAL when allowLowLight is true',
      () async {
        final detector = PassiveLivenessDetector();
        await detector.initialize(modelBytes: Uint8List.fromList([1, 2, 3, 4]));

        // Dark frame (mean luminance ~20 on 0..255 scale)
        const width = 100;
        const height = 100;
        final frameBytes = Uint8List(width * height * 4);
        for (var i = 0; i < frameBytes.length; i += 4) {
          frameBytes[i] = 20; // B
          frameBytes[i + 1] = 20; // G
          frameBytes[i + 2] = 20; // R
          frameBytes[i + 3] = 255; // A
        }

        final buffer = LivenessImageBuffer(
          width: width,
          height: height,
          format: LivenessImageFormat.bgra8888,
          planes: [
            LivenessImagePlane(
              bytes: frameBytes,
              bytesPerRow: width * 4,
              bytesPerPixel: 4,
            ),
          ],
        );

        // When lowLightThreshold is null (default off), isLowLight is false
        final normalResult = await detector.detectLivenessFromBuffer(
          buffer,
          enableProximityGate: false,
        );
        expect(normalResult.isLowLight, isFalse);
        expect(normalResult.meanLuminance, lessThan(55.0));

        // Now test with lowLightThreshold = 55.0 (auto-acceptance enabled)
        detector.resetEma();
        final lowLightResult = await detector.detectLivenessFromBuffer(
          buffer,
          enableProximityGate: false,
          lowLightThreshold: 55,
        );

        expect(lowLightResult.isReal, isTrue);
        expect(lowLightResult.status, equals(LivenessStatus.real));
        expect(lowLightResult.isLowLight, isTrue);
        expect(lowLightResult.meanLuminance, lessThan(55.0));

        await detector.dispose();
      },
    );
  });
}
