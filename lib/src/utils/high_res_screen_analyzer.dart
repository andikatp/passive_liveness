import 'dart:math' as math;
import 'dart:typed_data';

import 'package:passive_liveness/passive_liveness.dart' show ImagePreprocessor;

import 'package:passive_liveness/src/utils/image_preprocessor.dart'
    show ImagePreprocessor;

/// Result of evaluating high-resolution screen replay indicators.
class HighResScreenAnalysisResult {
  const HighResScreenAnalysisResult({
    required this.laplacianVariance,
    required this.patchLaplacianDispersal,
    required this.specularHighlightRatio,
    required this.isHighResScreenSpoof,
    this.faceLaplacianVariance = 0.0,
    this.backgroundLaplacianVariance = 0.0,
    this.is2DFlatSpoof = false,
  });

  /// Global Laplacian variance ($\sigma^2_{\text{Lap}}$) measuring high-frequency image energy.
  final double laplacianVariance;

  /// Real 3D faces have natural depth-of-field variations across facial
  /// features ($\ge 4.0$).
  /// 2D screen replays have a uniform focal plane resulting in low
  /// dispersal ($< 4.0$).
  final double patchLaplacianDispersal;

  /// Ratio of high-brightness specular reflection pixels ($0.0 \dots 1.0$).
  final double specularHighlightRatio;

  /// Whether high-resolution screen replay characteristics were detected.
  final bool isHighResScreenSpoof;

  /// Laplacian variance of the center face region (inner 50% crop).
  final double faceLaplacianVariance;

  /// Laplacian variance of the outer background ring region.
  final double backgroundLaplacianVariance;

  /// Whether the image exhibits 2D flat focal plane characteristics.
  ///
  /// Real 3D faces have depth-of-field differences between face and
  /// background, producing a large Laplacian delta. Screens and prints are
  /// uniformly focused with near-zero delta.
  final bool is2DFlatSpoof;
}

/// Evaluates 2D Laplacian frequency variance, patch focus depth dispersal,
/// and specular screen reflections to detect high-resolution OLED/Retina
/// screen replay attacks.
class HighResScreenAnalyzer {
  const HighResScreenAnalyzer({
    this.minPatchDispersalThreshold = 0.35,
    this.maxSpecularRatioThreshold = 0.030,
  });

  /// Minimum relative patch Laplacian dispersal (CV = std / mean) for a
  /// genuine 3D face (default: 0.35). Values below this indicate extremely
  /// flat 2D screen focal planes.
  final double minPatchDispersalThreshold;

  /// Specular highlight ratio upper bound threshold (default: 0.030).
  /// High values indicate glass screen reflection hotspots.
  final double maxSpecularRatioThreshold;

  /// Analyzes a grayscale crop ([width] x [height] Uint8List) extracted from
  /// [ImagePreprocessor].
  HighResScreenAnalysisResult analyzeGrayscaleCrop(
    Uint8List grayBytes,
    int width,
    int height,
  ) {
    if (width < 32 || height < 32 || grayBytes.length < width * height) {
      return const HighResScreenAnalysisResult(
        laplacianVariance: 0,
        patchLaplacianDispersal: 10,
        specularHighlightRatio: 0,
        isHighResScreenSpoof: false,
      );
    }

    // 1. Calculate global 2D Laplacian response L(x, y) = 4*I(x,y) - I(x+1,y)
    // - I(x-1,y) - I(x,y+1) - I(x,y-1)
    final laplacianValues = Float64List((width - 2) * (height - 2));
    var sumLap = 0.0;
    var lapCount = 0;
    var specularCount = 0;

    for (var y = 1; y < height - 1; y++) {
      final yOffset = y * width;
      for (var x = 1; x < width - 1; x++) {
        final center = grayBytes[yOffset + x];

        // Specular highlight pixel check (high brightness >= 245)
        if (center >= 245) {
          specularCount++;
        }

        final left = grayBytes[yOffset + x - 1];
        final right = grayBytes[yOffset + x + 1];
        final top = grayBytes[(y - 1) * width + x];
        final bottom = grayBytes[(y + 1) * width + x];

        final lap = (4 * center - left - right - top - bottom).toDouble();
        laplacianValues[lapCount++] = lap;
        sumLap += lap;
      }
    }

    final totalPixels = width * height;
    final specularRatio = totalPixels > 0 ? specularCount / totalPixels : 0.0;

    if (lapCount == 0) {
      return const HighResScreenAnalysisResult(
        laplacianVariance: 0,
        patchLaplacianDispersal: 10,
        specularHighlightRatio: 0,
        isHighResScreenSpoof: false,
      );
    }

    final meanLap = sumLap / lapCount;
    var varLap = 0.0;
    for (var i = 0; i < lapCount; i++) {
      final diff = laplacianValues[i] - meanLap;
      varLap += diff * diff;
    }
    varLap /= lapCount;

    // 2. Patch-based Laplacian focus depth dispersal (4x4 patch grid)
    const gridCols = 4;
    const gridRows = 4;
    final patchWidth = width ~/ gridCols;
    final patchHeight = height ~/ gridRows;
    final patchVariances = Float64List(gridCols * gridRows);
    var validPatches = 0;

    for (var py = 0; py < gridRows; py++) {
      final startY = py * patchHeight;
      final endY = (py + 1) * patchHeight;

      for (var px = 0; px < gridCols; px++) {
        final startX = px * patchWidth;
        final endX = (px + 1) * patchWidth;

        var pSum = 0.0;
        var pCount = 0;

        for (var y = math.max(1, startY); y < math.min(height - 1, endY); y++) {
          final yOffset = y * width;
          for (var x = math.max(1, startX);
              x < math.min(width - 1, endX);
              x++) {
            final center = grayBytes[yOffset + x];
            final left = grayBytes[yOffset + x - 1];
            final right = grayBytes[yOffset + x + 1];
            final top = grayBytes[(y - 1) * width + x];
            final bottom = grayBytes[(y + 1) * width + x];

            final lap = (4 * center - left - right - top - bottom).toDouble();
            pSum += lap;
            pCount++;
          }
        }

        if (pCount > 0) {
          final pMean = pSum / pCount;
          var pVar = 0.0;

          for (var y = math.max(1, startY);
              y < math.min(height - 1, endY);
              y++) {
            final yOffset = y * width;
            for (var x = math.max(1, startX);
                x < math.min(width - 1, endX);
                x++) {
              final center = grayBytes[yOffset + x];
              final left = grayBytes[yOffset + x - 1];
              final right = grayBytes[yOffset + x + 1];
              final top = grayBytes[(y - 1) * width + x];
              final bottom = grayBytes[(y + 1) * width + x];

              final lap = (4 * center - left - right - top - bottom).toDouble();
              final diff = lap - pMean;
              pVar += diff * diff;
            }
          }
          pVar /= pCount;
          patchVariances[validPatches++] = pVar;
        }
      }
    }

    var patchDispersal = 0.0;
    if (validPatches > 1) {
      var sumPVar = 0.0;
      for (var i = 0; i < validPatches; i++) {
        sumPVar += patchVariances[i];
      }
      final meanPVar = sumPVar / validPatches;

      var varPVar = 0.0;
      for (var i = 0; i < validPatches; i++) {
        final diff = patchVariances[i] - meanPVar;
        varPVar += diff * diff;
      }
      varPVar /= validPatches;
      // Relative coefficient of variation (CV = std / mean)
      final stdPVar = math.sqrt(varPVar);
      patchDispersal = meanPVar > 0 ? (stdPVar / meanPVar) : 0.0;
    }

    // High-resolution screen replays feature flat/uniform focus dispersal
    // (low CV) or high specular glare
    final isHighResScreenSpoof = patchDispersal < minPatchDispersalThreshold ||
        specularRatio > maxSpecularRatioThreshold;

    // 2D Flatness Check: Face (center 50%) vs Background (outer ring)
    // Laplacian delta. Real 3D faces have depth-of-field differences;
    // screens/prints are uniformly focused.
    final centerStartX = width ~/ 4;
    final centerStartY = height ~/ 4;
    final centerEndX = width * 3 ~/ 4;
    final centerEndY = height * 3 ~/ 4;

    var faceLapSum = 0.0;
    var faceLapCount = 0;
    var bgLapSum = 0.0;
    var bgLapCount = 0;

    for (var y = 1; y < height - 1; y++) {
      final yOffset = y * width;
      for (var x = 1; x < width - 1; x++) {
        final center = grayBytes[yOffset + x];
        final left = grayBytes[yOffset + x - 1];
        final right = grayBytes[yOffset + x + 1];
        final top = grayBytes[(y - 1) * width + x];
        final bottom = grayBytes[(y + 1) * width + x];
        final lap = (4 * center - left - right - top - bottom).toDouble();

        final isCenter = x >= centerStartX &&
            x < centerEndX &&
            y >= centerStartY &&
            y < centerEndY;
        if (isCenter) {
          faceLapSum += lap;
          faceLapCount++;
        } else {
          bgLapSum += lap;
          bgLapCount++;
        }
      }
    }

    var faceLapVar = 0.0;
    var bgLapVar = 0.0;

    if (faceLapCount > 0) {
      final faceMean = faceLapSum / faceLapCount;
      var faceVarAccum = 0.0;
      for (var y = math.max(1, centerStartY);
          y < math.min(height - 1, centerEndY);
          y++) {
        final yOffset = y * width;
        for (var x = math.max(1, centerStartX);
            x < math.min(width - 1, centerEndX);
            x++) {
          final c = grayBytes[yOffset + x];
          final l = grayBytes[yOffset + x - 1];
          final r = grayBytes[yOffset + x + 1];
          final t = grayBytes[(y - 1) * width + x];
          final b = grayBytes[(y + 1) * width + x];
          final lap = (4 * c - l - r - t - b).toDouble();
          final diff = lap - faceMean;
          faceVarAccum += diff * diff;
        }
      }
      faceLapVar = faceVarAccum / faceLapCount;
    }

    if (bgLapCount > 0) {
      final bgMean = bgLapSum / bgLapCount;
      var bgVarAccum = 0.0;
      for (var y = 1; y < height - 1; y++) {
        final yOffset = y * width;
        for (var x = 1; x < width - 1; x++) {
          final isCenter = x >= centerStartX &&
              x < centerEndX &&
              y >= centerStartY &&
              y < centerEndY;
          if (isCenter) continue;
          final c = grayBytes[yOffset + x];
          final l = grayBytes[yOffset + x - 1];
          final r = grayBytes[yOffset + x + 1];
          final t = grayBytes[(y - 1) * width + x];
          final b = grayBytes[(y + 1) * width + x];
          final lap = (4 * c - l - r - t - b).toDouble();
          final diff = lap - bgMean;
          bgVarAccum += diff * diff;
        }
      }
      bgLapVar = bgVarAccum / bgLapCount;
    }

    // Compute relative delta between face and background Laplacian variances.
    // Use coefficient of variation (normalized difference) to be
    // scale-invariant.
    final avgLapVar = (faceLapVar + bgLapVar) / 2.0;
    final lapDelta = (faceLapVar - bgLapVar).abs();
    final normalizedDelta = avgLapVar > 0.0 ? lapDelta / avgLapVar : 0.0;

    // 2D flat planes (screens, prints) have extremely low normalizedDelta
    // (< 0.08) combined with uniform patch dispersal (< 0.30) or high laplacian
    // variance (> 2500).
    final is2DFlatSpoof =
        (normalizedDelta < 0.08 && patchDispersal < 0.30 && varLap > 500.0) ||
            (normalizedDelta < 0.04 && varLap > 2000.0);

    return HighResScreenAnalysisResult(
      laplacianVariance: varLap,
      patchLaplacianDispersal: patchDispersal,
      specularHighlightRatio: specularRatio,
      isHighResScreenSpoof: isHighResScreenSpoof,
      faceLaplacianVariance: faceLapVar,
      backgroundLaplacianVariance: bgLapVar,
      is2DFlatSpoof: is2DFlatSpoof,
    );
  }
}
