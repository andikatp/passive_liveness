import 'dart:math' as math;
import 'dart:typed_data';

import 'package:passive_liveness/src/models/face_bounding_box.dart';
import 'package:passive_liveness/src/models/liveness_image_buffer.dart';

/// Result of evaluating YCbCr color space chrominance distributions.
class ColorSpaceAnalysisResult {
  const ColorSpaceAnalysisResult({
    required this.chrominanceVariance,
    required this.meanCb,
    required this.meanCr,
    required this.isScreenReplaySpoof,
    this.saturationVariance = 0.0,
    this.isEmissiveSaturationSpoof = false,
  });

  /// Calculated chrominance variance ($\sigma^2_{CbCr} = \sigma^2_{Cb} + \sigma^2_{Cr}$).
  final double chrominanceVariance;

  /// Mean Cb channel chrominance value ($0 \dots 255$).
  final double meanCb;

  /// Mean Cr channel chrominance value ($0 \dots 255$).
  final double meanCr;

  /// Whether the chrominance distribution indicates a digital screen
  /// replay attack.
  final bool isScreenReplaySpoof;

  /// HSV Saturation channel variance.
  ///
  /// Screens emit additive RGB light causing unnatural saturation spikes
  /// in backlight scatter, unlike human skin which reflects subtractive light.
  final double saturationVariance;

  /// Whether emissive saturation spike characteristics were detected.
  final bool isEmissiveSaturationSpoof;
}

/// Evaluates YCbCr / YUV chrominance sub-sampling metrics ($\sigma^2_{CbCr}$)
/// to detect emissive RGB digital display screen replay attacks.
class ColorSpaceAnalyzer {
  const ColorSpaceAnalyzer({
    this.maxVarianceThreshold = 380.0,
    this.minVarianceThreshold = 0.5,
  });

  /// Upper bound threshold for chrominance variance (default: 380.0).
  /// Higher values indicate screen sub-pixel dispersion.
  final double maxVarianceThreshold;

  /// Lower bound threshold for chrominance variance (default: 0.5). Extremely
  /// low values indicate flat monochrome / synthetic photos.
  final double minVarianceThreshold;

  /// Evaluates a raw [LivenessImageBuffer] in YUV420 or BGRA format.
  ColorSpaceAnalysisResult analyzeBuffer(
    LivenessImageBuffer buffer, {
    FaceBoundingBox? boundingBox,
  }) {
    final width = buffer.width;
    final height = buffer.height;

    if (width <= 0 || height <= 0 || buffer.planes.isEmpty) {
      return const ColorSpaceAnalysisResult(
        chrominanceVariance: 0,
        meanCb: 128,
        meanCr: 128,
        isScreenReplaySpoof: false,
      );
    }

    var startX = 0;
    var startY = 0;
    var endX = width;
    var endY = height;

    if (boundingBox != null) {
      // Analyze within face region to avoid false background positives
      final bX = math.max(0, boundingBox.x.toInt());
      final bY = math.max(0, boundingBox.y.toInt());
      final bW = math.min(width, (boundingBox.x + boundingBox.width).ceil());
      final bH = math.min(height, (boundingBox.y + boundingBox.height).ceil());

      if (bW > bX && bH > bY) {
        startX = bX;
        startY = bY;
        endX = bW;
        endY = bH;
      }
    }

    final isBgra = buffer.format == LivenessImageFormat.bgra8888;
    final isRgba = buffer.format == LivenessImageFormat.rgba8888;

    var sumCb = 0.0;
    var sumCr = 0.0;
    var sumSat = 0.0;
    var sampleCount = 0;

    // Subsample every 4th pixel for high performance without loss of
    // statistical accuracy
    const step = 4;

    final maxSamples =
        ((width + step - 1) ~/ step) * ((height + step - 1) ~/ step);
    final cbList = Float64List(maxSamples);
    final crList = Float64List(maxSamples);
    final satList = Float64List(maxSamples);

    if (isBgra || isRgba) {
      final plane0 = buffer.planes[0].bytes;
      final stride = buffer.planes[0].bytesPerRow;

      for (var y = startY; y < endY; y += step) {
        final rowOffset = y * stride;
        for (var x = startX; x < endX; x += step) {
          final offset = rowOffset + (x << 2);
          if (offset + 2 < plane0.length) {
            final b = (isBgra ? plane0[offset] : plane0[offset + 2]).toDouble();
            final g = plane0[offset + 1].toDouble();
            final r = (isBgra ? plane0[offset + 2] : plane0[offset]).toDouble();

            // Skip specular glare pixels (high brightness >= 245)
            final luma = 0.299 * r + 0.587 * g + 0.114 * b;
            if (luma >= 245) continue;

            // BT.601 RGB -> YCbCr formulas
            final cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;
            final cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;

            // HSV Saturation: S = (max - min) / max  (0..1 range)
            final maxC = math.max(r, math.max(g, b));
            final minC = math.min(r, math.min(g, b));
            final sat = maxC > 0.0 ? (maxC - minC) / maxC : 0.0;

            cbList[sampleCount] = cb;
            crList[sampleCount] = cr;
            satList[sampleCount] = sat;
            sumCb += cb;
            sumCr += cr;
            sumSat += sat;
            sampleCount++;
          }
        }
      }
    } else {
      // YUV420 / NV21 buffer planes
      final plane0 = buffer.planes[0].bytes;
      final p0Stride = buffer.planes[0].bytesPerRow;

      final plane1 = buffer.planes.length > 1 ? buffer.planes[1].bytes : null;
      final plane2 = buffer.planes.length > 2 ? buffer.planes[2].bytes : null;

      final uvStride =
          buffer.planes.length > 1 ? buffer.planes[1].bytesPerRow : p0Stride;
      final uvPixelStride =
          buffer.planes.length > 1 ? (buffer.planes[1].bytesPerPixel ?? 1) : 2;

      for (var y = startY; y < endY; y += step) {
        final yIdxBase = y * p0Stride;
        final uvRow = (y >> 1) * uvStride;

        for (var x = startX; x < endX; x += step) {
          final uvCol = x >> 1;
          var uVal = 128;
          var vVal = 128;

          if (plane1 != null) {
            final uvOff = uvRow + uvCol * uvPixelStride;
            if (uvPixelStride == 2 && uvOff + 1 < plane1.length) {
              if (buffer.format == LivenessImageFormat.nv21) {
                vVal = plane1[uvOff];
                uVal = plane1[uvOff + 1];
              } else {
                uVal = plane1[uvOff];
                vVal = plane1[uvOff + 1];
              }
            } else if (plane2 != null &&
                uvOff < plane1.length &&
                uvOff < plane2.length) {
              if (buffer.format == LivenessImageFormat.nv21) {
                vVal = plane1[uvOff];
                uVal = plane2[uvOff];
              } else {
                uVal = plane1[uvOff];
                vVal = plane2[uvOff];
              }
            }
          } else if (plane0.length >= width * height * 3 ~/ 2) {
            // Packed single-plane NV21 / NV12 in plane0
            final uvOff =
                p0Stride * height + (y >> 1) * p0Stride + (uvCol << 1);
            if (uvOff + 1 < plane0.length) {
              if (buffer.format == LivenessImageFormat.nv21) {
                vVal = plane0[uvOff];
                uVal = plane0[uvOff + 1];
              } else {
                uVal = plane0[uvOff];
                vVal = plane0[uvOff + 1];
              }
            }
          }

          final cb = uVal.toDouble();
          final cr = vVal.toDouble();

          // Calculate BT.601 YUV -> RGB -> HSV Saturation
          final yIdx = yIdxBase + x;
          final yVal = (yIdx < plane0.length) ? plane0[yIdx].toDouble() : 128.0;
          final u = cb - 128.0;
          final v = cr - 128.0;

          final r = (yVal + 1.402 * v).clamp(0.0, 255.0);
          final g = (yVal - 0.344136 * u - 0.714136 * v).clamp(0.0, 255.0);
          final b = (yVal + 1.772 * u).clamp(0.0, 255.0);

          final maxC = math.max(r, math.max(g, b));
          final minC = math.min(r, math.min(g, b));
          final sat = maxC > 0.0 ? (maxC - minC) / maxC : 0.0;

          cbList[sampleCount] = cb;
          crList[sampleCount] = cr;
          satList[sampleCount] = sat;
          sumCb += cb;
          sumCr += cr;
          sumSat += sat;
          sampleCount++;
        }
      }
    }

    if (sampleCount == 0) {
      return const ColorSpaceAnalysisResult(
        chrominanceVariance: 0,
        meanCb: 128,
        meanCr: 128,
        isScreenReplaySpoof: false,
      );
    }

    final meanCb = sumCb / sampleCount;
    final meanCr = sumCr / sampleCount;
    final meanSat = sumSat / sampleCount;

    var varCb = 0.0;
    var varCr = 0.0;
    var varSat = 0.0;

    for (var i = 0; i < sampleCount; i++) {
      final diffCb = cbList[i] - meanCb;
      final diffCr = crList[i] - meanCr;
      final diffSat = satList[i] - meanSat;
      varCb += diffCb * diffCb;
      varCr += diffCr * diffCr;
      varSat += diffSat * diffSat;
    }

    varCb /= sampleCount;
    varCr /= sampleCount;
    varSat /= sampleCount;

    final totalChrominanceVar = varCb + varCr;
    final isScreenReplaySpoof = totalChrominanceVar > maxVarianceThreshold ||
        totalChrominanceVar < minVarianceThreshold;

    // Emissive saturation spoof: screens emit additive RGB backlight that
    // creates higher saturation variance (varSat >= 0.12) combined with
    // elevated chrominance variance (>= 250.0).
    final isEmissiveSaturationSpoof =
        varSat >= 0.12 && totalChrominanceVar >= 250.0;

    return ColorSpaceAnalysisResult(
      chrominanceVariance: totalChrominanceVar,
      meanCb: meanCb,
      meanCr: meanCr,
      isScreenReplaySpoof: isScreenReplaySpoof,
      saturationVariance: varSat,
      isEmissiveSaturationSpoof: isEmissiveSaturationSpoof,
    );
  }
}
