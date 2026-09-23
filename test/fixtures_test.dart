// Diagnostic test for liveness fixtures.
// ignore_for_file: avoid_print

import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:passive_liveness/passive_liveness.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late PassiveLivenessDetector detector;
  final fixturesDir = Directory('test/fixtures');

  var mockLogits = <double>[3.5, 0.5];

  setUpAll(() async {
    const channel = MethodChannel('com.andikatp.passiveLiveness');
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
        return mockLogits;
      }
      if (methodCall.method == 'closeModel') {
        return null;
      }
      return null;
    });

    detector = PassiveLivenessDetector();
    await detector.initialize();
  });

  tearDownAll(() async {
    await detector.dispose();
  });

  test('Diagnose all fixture images in test/fixtures', () async {
    if (!fixturesDir.existsSync()) {
      print('test/fixtures directory does not exist.');
      return;
    }

    final files = fixturesDir.listSync().whereType<File>().toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    print('\n======================================================');
    print('DIAGNOSING FIXTURE IMAGES (${files.length} files found)');
    print('======================================================\n');

    final summaryLines = <String>[
      'FILENAME | EXPECTED | IS_REAL | STATUS | CHROM_VAR | LBP_RATIO',
      '------------------------------------------------------------',
    ];

    for (final file in files) {
      final lowerPath = file.path.toLowerCase();
      if (!lowerPath.endsWith('.jpg') &&
          !lowerPath.endsWith('.jpeg') &&
          !lowerPath.endsWith('.png')) {
        continue;
      }

      final fileName = file.path.split('/').last;
      detector.resetEma();
      mockLogits = [
        3.5,
        0.5,
      ]; // Test heuristic override when neural model predicts REAL
      final bytes = await file.readAsBytes();

      FaceBoundingBox? bbox;
      if (fileName.toLowerCase().contains('real5')) {
        bbox = const FaceBoundingBox(x: 520, y: 610, width: 700, height: 1200);
      }

      final result = await detector.detectLivenessFromImageBytes(
        bytes,
        boundingBox: bbox,
      );

      final expected = fileName.contains('real') ? 'REAL' : 'SPOOF';
      final isRealStr = result.isReal ? 'TRUE' : 'FALSE';

      final chromVar = result.chrominanceVariance?.toStringAsFixed(1) ?? 'N/A';
      final lbp = result.lbpUniformityScore?.toStringAsFixed(3) ?? 'N/A';
      final hog = result.hogGridDominance?.toStringAsFixed(3) ?? 'N/A';
      final lapDelta = result.laplacianDelta?.toStringAsFixed(3) ?? 'N/A';
      final satVar = result.saturationVariance?.toStringAsFixed(4) ?? 'N/A';
      final moireHf = result.moireHighFreqRatio?.toStringAsFixed(3) ?? 'N/A';

      summaryLines.add(
        '${fileName.padRight(15)} | $expected | $isRealStr | '
        '${result.status.name} | $chromVar | $lbp | $hog | '
        '$lapDelta | $satVar | $moireHf',
      );

      // Assert expected liveness result per fixture type
      if (fileName.contains('real')) {
        expect(
          result.isReal,
          isTrue,
          reason: 'Expected $fileName to be classified as REAL',
        );
      } else if (fileName.contains('spoof')) {
        expect(
          result.isReal,
          isFalse,
          reason: 'Expected screen replay $fileName to be caught as SPOOF',
        );
      }
    }

    print('\n${summaryLines.join('\n')}\n');
  });
}
