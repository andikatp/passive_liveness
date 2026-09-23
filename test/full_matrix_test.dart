// Diagnostic full matrix test.
// ignore_for_file: avoid_print

import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter_test/flutter_test.dart';
import 'package:passive_liveness/passive_liveness.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  final fixturesDir = Directory('test/fixtures');

  test('Print complete matrix of all heuristics for all 16 fixtures', () async {
    if (!fixturesDir.existsSync()) return;

    final files = fixturesDir.listSync().whereType<File>().toList()
      ..sort((a, b) => a.path.compareTo(b.path));

    print(
      '\n========================================================',
    );
    print(
      'FILENAME | EXPECTED | CHROM_VAR | MEAN_CB | MEAN_CR',
    );
    print(
      '--------------------------------------------------------',
    );

    for (final file in files) {
      final lowerPath = file.path.toLowerCase();
      if (!lowerPath.endsWith('.jpg') &&
          !lowerPath.endsWith('.jpeg') &&
          !lowerPath.endsWith('.png')) {
        continue;
      }

      final fileName = file.path.split('/').last;
      final bytes = await file.readAsBytes();

      final codec = await ui.instantiateImageCodec(bytes);
      final frameInfo = await codec.getNextFrame();
      final imgWidth = frameInfo.image.width;
      final imgHeight = frameInfo.image.height;
      final rawByteData = await frameInfo.image.toByteData();
      frameInfo.image.dispose();
      codec.dispose();

      final rgbaBytes = rawByteData!.buffer.asUint8List();
      final buffer = LivenessImageBuffer(
        width: imgWidth,
        height: imgHeight,
        format: LivenessImageFormat.rgba8888,
        planes: [
          LivenessImagePlane(
            bytes: rgbaBytes,
            bytesPerRow: imgWidth * 4,
            bytesPerPixel: 4,
          ),
        ],
      );

      const colorAnalyzer = ColorSpaceAnalyzer();
      final colorRes = colorAnalyzer.analyzeBuffer(buffer);

      const textureAnalyzer = LbpHogAnalyzer();
      final highResCrop = ImagePreprocessor.extractHighResCrop(buffer);
      final textureRes = textureAnalyzer.analyzeGrayscaleCrop(
        highResCrop,
        256,
        256,
      );

      const highResAnalyzer = HighResScreenAnalyzer();
      final highResRes = highResAnalyzer.analyzeGrayscaleCrop(
        highResCrop,
        256,
        256,
      );

      final expected = fileName.contains('real') ? 'REAL' : 'SPOOF';

      print(
        '${fileName.padRight(15)} | ${expected.padRight(6)} | '
        '${colorRes.chrominanceVariance.toStringAsFixed(1)} | '
        '${textureRes.lbpNonUniformRatio.toStringAsFixed(3)} | '
        '${textureRes.hogPeakDominance.toStringAsFixed(3)} | '
        '${highResRes.laplacianVariance.toStringAsFixed(1)} | '
        '${highResRes.patchLaplacianDispersal.toStringAsFixed(3)} | '
        '${colorRes.saturationVariance.toStringAsFixed(4)}',
      );
    }
    print(
      '========================================================\n',
    );
  });
}
