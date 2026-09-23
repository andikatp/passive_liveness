import 'dart:math' as math;
import 'dart:typed_data';

/// Result of evaluating Local Binary Pattern (LBP) and Histogram of Oriented
/// Gradients (HOG) micro-textures.
class TextureAnalysisResult {
  const TextureAnalysisResult({
    required this.lbpNonUniformRatio,
    required this.hogPeakDominance,
    required this.isPrintSpoof,
    required this.isScreenGridSpoof,
  });

  /// Ratio of non-uniform LBP patterns ($0.0 \dots 1.0$). High values ($>0.35$) indicate paper/inkjet print noise.
  final double lbpNonUniformRatio;

  /// Peak gradient orientation dominance ratio ($0.0 \dots 1.0$). High values ($>0.40$) indicate digital screen grid alignment.
  final double hogPeakDominance;

  /// Whether the micro-texture indicates an inkjet paper print spoof.
  final bool isPrintSpoof;

  /// Whether the micro-texture indicates a digital screen grid replay spoof.
  final bool isScreenGridSpoof;

  /// True if any micro-texture spoof pattern was detected.
  bool get isTextureSpoof => isPrintSpoof || isScreenGridSpoof;
}

/// Lightweight micro-texture analysis engine using Local Binary Pattern (LBP)
/// and Histogram of Oriented Gradients (HOG).
///
/// Features specular glare masking (preventing glasses glare false positives)
/// and multi-region HOG
/// frame de-biasing (preventing linear glasses rims from triggering screen
/// grid spoofs).
class LbpHogAnalyzer {
  const LbpHogAnalyzer({
    this.lbpPrintThreshold = 0.38,
    this.hogScreenThreshold = 0.42,
    this.specularThreshold = 245,
  });

  /// Threshold for non-uniform LBP pattern ratio indicating print
  /// spoofing (default: 0.38).
  final double lbpPrintThreshold;

  /// Threshold for HOG orientation peak dominance indicating screen
  /// grid replay (default: 0.42).
  final double hogScreenThreshold;

  /// Luminance threshold above which pixels are treated as specular
  /// glare (default: 245).
  final int specularThreshold;

  /// Pre-computed lookup table for 8-bit uniform LBP pattern check.
  static final Uint8List _uniformLbpLookup = _buildUniformLbpLookup();

  static Uint8List _buildUniformLbpLookup() {
    final table = Uint8List(256);
    for (var code = 0; code < 256; code++) {
      var transitions = 0;
      for (var i = 0; i < 8; i++) {
        final currentBit = (code >> i) & 1;
        final nextBit = (code >> ((i + 1) % 8)) & 1;
        if (currentBit != nextBit) {
          transitions++;
        }
      }
      // Uniform patterns have at most 2 bit transitions
      table[code] = transitions <= 2 ? 1 : 0;
    }
    return table;
  }

  /// Analyzes a grayscale pixel crop ([width] x [height] Uint8List).
  TextureAnalysisResult analyzeGrayscaleCrop(
    Uint8List grayBytes,
    int width,
    int height,
  ) {
    if (width < 8 || height < 8 || grayBytes.length < width * height) {
      return const TextureAnalysisResult(
        lbpNonUniformRatio: 0,
        hogPeakDominance: 0,
        isPrintSpoof: false,
        isScreenGridSpoof: false,
      );
    }

    var nonUniformLbpCount = 0;
    var totalLbpEvaluated = 0;

    // 8-neighbor offsets: (dx, dy)
    const dx = [-1, 0, 1, 1, 1, 0, -1, -1];
    const dy = [-1, -1, -1, 0, 1, 1, 1, 0];

    // 1. Calculate LBP histogram with Specular Glare Masking
    for (var y = 1; y < height - 1; y++) {
      final yOffset = y * width;
      for (var x = 1; x < width - 1; x++) {
        final centerPixel = grayBytes[yOffset + x];

        // Mask out specular glare pixels (bright highlights on glasses or
        //glossy surfaces)
        if (centerPixel >= specularThreshold) continue;

        var lbpCode = 0;

        for (var i = 0; i < 8; i++) {
          final neighborPixel = grayBytes[(y + dy[i]) * width + (x + dx[i])];
          if (neighborPixel >= centerPixel) {
            lbpCode |= 1 << i;
          }
        }

        if (_uniformLbpLookup[lbpCode] == 0) {
          nonUniformLbpCount++;
        }
        totalLbpEvaluated++;
      }
    }

    final lbpRatio =
        totalLbpEvaluated > 0 ? nonUniformLbpCount / totalLbpEvaluated : 0.0;

    // 2. Multi-Zone HOG calculation (Full Face vs Lower-Face De-Biasing for
    //Glasses Rims)
    final hogBinsGlobal = Float64List(9);
    final hogBinsLower = Float64List(9);
    var totalGlobalGradient = 0.0;
    var totalLowerGradient = 0.0;

    final upperBoundaryY = (height * 0.38).round();

    for (var y = 1; y < height - 1; y++) {
      final yOffset = y * width;
      final isLowerFace = y >= upperBoundaryY;

      for (var x = 1; x < width - 1; x++) {
        final centerPixel = grayBytes[yOffset + x];
        if (centerPixel >= specularThreshold) continue;

        final gx = (grayBytes[yOffset + x + 1] - grayBytes[yOffset + x - 1])
            .toDouble();
        final gy =
            (grayBytes[(y + 1) * width + x] - grayBytes[(y - 1) * width + x])
                .toDouble();

        final mag = math.sqrt(gx * gx + gy * gy);
        if (mag < 10.0) continue; // Ignore weak background gradients

        // Calculate angle in degrees [0, 180)
        var angleRad = math.atan2(gy, gx);
        if (angleRad < 0) angleRad += math.pi;
        var angleDeg = angleRad * (180.0 / math.pi);
        if (angleDeg >= 180.0) angleDeg = 179.9;

        final binIdx = (angleDeg / 20.0).floor().clamp(0, 8);
        hogBinsGlobal[binIdx] += mag;
        totalGlobalGradient += mag;

        if (isLowerFace) {
          hogBinsLower[binIdx] += mag;
          totalLowerGradient += mag;
        }
      }
    }

    var maxGlobalBin = 0.0;
    if (totalGlobalGradient > 0) {
      for (var b = 0; b < 9; b++) {
        if (hogBinsGlobal[b] > maxGlobalBin) {
          maxGlobalBin = hogBinsGlobal[b];
        }
      }
    }

    var maxLowerBin = 0.0;
    if (totalLowerGradient > 0) {
      for (var b = 0; b < 9; b++) {
        if (hogBinsLower[b] > maxLowerBin) {
          maxLowerBin = hogBinsLower[b];
        }
      }
    }

    final globalDominance =
        totalGlobalGradient > 0 ? maxGlobalBin / totalGlobalGradient : 0.0;
    final lowerDominance =
        totalLowerGradient > 0 ? maxLowerBin / totalLowerGradient : 0.0;

    // Use lower face dominance when glasses frames dominate the upper face,
    // or average them if lower face gradient is valid.
    final effectiveHogDominance =
        (totalLowerGradient > 0 && lowerDominance < globalDominance)
            ? (globalDominance * 0.4 + lowerDominance * 0.6)
            : globalDominance;

    final isPrintSpoof = lbpRatio >= lbpPrintThreshold;
    final isScreenGridSpoof = effectiveHogDominance >= hogScreenThreshold;

    return TextureAnalysisResult(
      lbpNonUniformRatio: lbpRatio,
      hogPeakDominance: effectiveHogDominance,
      isPrintSpoof: isPrintSpoof,
      isScreenGridSpoof: isScreenGridSpoof,
    );
  }
}
