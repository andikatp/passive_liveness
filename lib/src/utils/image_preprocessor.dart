import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:passive_liveness/src/models/face_bounding_box.dart';
import 'package:passive_liveness/src/models/liveness_image_buffer.dart';
import 'package:passive_liveness/src/utils/liveness_logger.dart';

/// Normalization schemes supported for input tensor preparation.
enum NormalizationScheme {
  /// Standard 0.0 to 1.0 normalization (`val / 255.0`).
  zeroToOne,

  /// Symmetric -1.0 to 1.0 normalization (`(val - 127.5) / 127.5`).
  minusOneToOne,

  /// PyTorch standard ImageNet normalization (`(val/255.0 - mean) / std`).
  /// Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225].
  imageNet,
}

/// Preprocessing utilities for passive face anti-spoofing input.
class ImagePreprocessor {
  const ImagePreprocessor._();

  /// Default input size expected by MiniFAS model (128x128).
  static const int defaultModelSize = 128;

  /// Default bounding box expansion factor recommended by MiniFAS
  /// (1.5x for this custom model).
  static const double defaultExpansionFactor = 1.5;

  /// Normalizes raw 0..255 pixel intensity according to specified
  /// [NormalizationScheme].
  static double normalizePixel(
    double val, {
    required int colorChannel, // 0: Red, 1: Green, 2: Blue
    NormalizationScheme scheme = NormalizationScheme.zeroToOne,
  }) {
    switch (scheme) {
      case NormalizationScheme.minusOneToOne:
        return (val - 127.5) / 127.5;
      case NormalizationScheme.imageNet:
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];
        final cIdx = colorChannel.clamp(0, 2);
        return ((val / 255.0) - mean[cIdx]) / std[cIdx];
      case NormalizationScheme.zeroToOne:
        return val / 255.0;
    }
  }

  /// Reflect101 coordinate mapping (`g fedcba|abcdefgh|gfedcba`) for smooth
  /// border padding.
  ///
  /// Mirrors coordinates past image boundaries `[0, maxVal]` to avoid
  /// artificial solid constant-color edge bands.
  static int _reflect101(int p, int maxVal) {
    if (maxVal <= 0) return 0;
    var val = p;
    while (val < 0 || val > maxVal) {
      if (val < 0) {
        val = -val;
      } else if (val > maxVal) {
        val = 2 * maxVal - val;
      }
    }
    return val;
  }

  /// Preprocesses a raw camera [LivenessImageBuffer] into a Float32 tensor.
  ///
  /// Stores normalized RGB/BGR pixel values in Float32 format.
  /// Out-of-boundary pixels use **Reflect101 Border Padding** (`_reflect101`)
  /// to avoid artificial black border artifacts.
  ///
  /// If [useNchw] is `true` (default), the tensor is structured in NCHW
  /// format `[1, 3, targetSize, targetSize]`. Otherwise, NHWC format
  /// `[1, targetSize, targetSize, 3]` is used.
  static Float32List preprocessBufferToTensor(
    LivenessImageBuffer buffer, {
    FaceBoundingBox? boundingBox,
    int rotation = 0,
    bool? isRotatedBoundingBox,
    double expansionFactor = defaultExpansionFactor,
    int targetSize = defaultModelSize,
    bool useNchw = true,
    bool isBgr = false,
    NormalizationScheme normalizationScheme = NormalizationScheme.zeroToOne,
    bool enableContrastStretch = true,
  }) {
    final rawW = buffer.width;
    final rawH = buffer.height;
    final normRotation = ((rotation % 360) + 360) % 360;

    final faceBbox = boundingBox ??
        FaceBoundingBox(
          x: 0,
          y: 0,
          width: rawW.toDouble(),
          height: rawH.toDouble(),
        );

    // Auto-detect if bounding box is in rotated frame space [0..rotW, 0..rotH]
    final isRotated = isRotatedBoundingBox ??
        (Platform.isAndroid && (normRotation == 90 || normRotation == 270)) ||
            ((normRotation == 90 || normRotation == 270) &&
                (faceBbox.centerY > rawH || faceBbox.centerX > rawW));

    final effectiveBbox = isRotated
        ? faceBbox.toRawBufferSpace(rawW, rawH, normRotation)
        : faceBbox;

    // Strictly 1:1 square crop calculation (baseSide = max(w, h)) preserving
    // trained 37.0% facial scale:
    final boxW = math.max(1, effectiveBbox.width);
    final boxH = math.max(1, effectiveBbox.height);
    final baseSide = math.max(boxW, boxH);
    final cropSize = (baseSide * expansionFactor).toInt().toDouble();

    final newW = cropSize;
    final newH = cropSize;

    final cropLeft =
        (effectiveBbox.centerX - cropSize / 2.0).toInt().toDouble();
    final cropTop = (effectiveBbox.centerY - cropSize / 2.0).toInt().toDouble();

    LivenessLogger.logCropStats(
      rawWidth: rawW,
      rawHeight: rawH,
      rotation: normRotation,
      boundingBox: boundingBox,
      expansionFactor: expansionFactor,
      cropLeft: cropLeft,
      cropTop: cropTop,
      cropWidth: newW,
      cropHeight: newH,
    );

    final tensor = Float32List(1 * targetSize * targetSize * 3);
    final hw = targetSize * targetSize;

    final isBgra = buffer.format == LivenessImageFormat.bgra8888;
    final isRgba = buffer.format == LivenessImageFormat.rgba8888;

    final plane0 = buffer.planes[0].bytes;
    final p0Stride = buffer.planes[0].bytesPerRow;

    final plane1 = buffer.planes.length > 1 ? buffer.planes[1].bytes : null;
    final plane2 = buffer.planes.length > 2 ? buffer.planes[2].bytes : null;

    final uvStride = plane1 != null ? buffer.planes[1].bytesPerRow : 0;
    final uvPixelStride =
        plane1 != null ? (buffer.planes[1].bytesPerPixel ?? 1) : 1;

    final step = cropSize / targetSize;

    for (var y = 0; y < targetSize; y++) {
      final cy1 = y * step;
      final cy2 = (y + 1) * step;

      for (var x = 0; x < targetSize; x++) {
        final cx1 = x * step;
        final cx2 = (x + 1) * step;

        double rawX1;
        double rawX2;
        double rawY1;
        double rawY2;
        switch (normRotation) {
          case 90:
            rawX1 = cropLeft + cy1;
            rawX2 = cropLeft + cy2;
            rawY1 = cropTop + newH - cx2;
            rawY2 = cropTop + newH - cx1;
          case 180:
            rawX1 = cropLeft + newW - cx2;
            rawX2 = cropLeft + newW - cx1;
            rawY1 = cropTop + newH - cy2;
            rawY2 = cropTop + newH - cy1;
          case 270:
            rawX1 = cropLeft + newW - cy2;
            rawX2 = cropLeft + newW - cy1;
            rawY1 = cropTop + cx1;
            rawY2 = cropTop + cx2;
          case 0:
          default:
            rawX1 = cropLeft + cx1;
            rawX2 = cropLeft + cx2;
            rawY1 = cropTop + cy1;
            rawY2 = cropTop + cy2;
        }

        var sumR = 0.0;
        var sumG = 0.0;
        var sumB = 0.0;
        var sumY = 0.0;
        var sumU = 0.0;
        var sumV = 0.0;
        var totalWeight = 0.0;

        final startY = rawY1.floor();
        final endY = rawY2.ceil();
        final startX = rawX1.floor();
        final endX = rawX2.ceil();

        for (var ry = startY; ry < endY; ry++) {
          final ryD = ry.toDouble();
          final ryp1 = ryD + 1.0;
          final yMin = ryp1 < rawY2 ? ryp1 : rawY2;
          final yMax = ryD > rawY1 ? ryD : rawY1;
          final yOverlap = yMin - yMax;
          if (yOverlap <= 0) continue;

          final srcY = _reflect101(ry, rawH - 1);
          final yIdxBase = srcY * p0Stride;
          final uvRow = srcY >> 1;
          final uvBase = uvRow * uvStride;

          for (var rx = startX; rx < endX; rx++) {
            final rxD = rx.toDouble();
            final rxp1 = rxD + 1.0;
            final xMin = rxp1 < rawX2 ? rxp1 : rawX2;
            final xMax = rxD > rawX1 ? rxD : rawX1;
            final xOverlap = xMin - xMax;
            final w = xOverlap * yOverlap;
            if (w <= 0) continue;

            final srcX = _reflect101(rx, rawW - 1);

            if (isBgra || isRgba) {
              var r = 0;
              var g = 0;
              var b = 0;
              final offset = yIdxBase + (srcX << 2);
              if (offset + 2 < plane0.length) {
                b = isBgra ? plane0[offset] : plane0[offset + 2];
                g = plane0[offset + 1];
                r = isBgra ? plane0[offset + 2] : plane0[offset];
              }
              sumR += r * w;
              sumG += g * w;
              sumB += b * w;
            } else {
              // YUV420 / NV21
              final yIdx = yIdxBase + srcX;
              final yVal = yIdx < plane0.length ? plane0[yIdx] : 0;

              var uVal = 128;
              var vVal = 128;

              if (plane1 != null && plane2 != null) {
                final uvCol = srcX >> 1;
                final uvOff = uvBase + uvCol * uvPixelStride;

                if (uvPixelStride == 2) {
                  if (uvOff + 1 < plane1.length) {
                    if (buffer.format == LivenessImageFormat.nv21) {
                      vVal = plane1[uvOff];
                      uVal = plane1[uvOff + 1];
                    } else {
                      uVal = plane1[uvOff];
                      vVal = plane1[uvOff + 1];
                    }
                  } else if (uvOff < plane1.length) {
                    uVal = plane1[uvOff];
                    vVal = uvOff < plane2.length ? plane2[uvOff] : 128;
                  }
                } else {
                  if (uvOff < plane1.length && uvOff < plane2.length) {
                    if (buffer.format == LivenessImageFormat.nv21) {
                      vVal = plane1[uvOff];
                      uVal = plane2[uvOff];
                    } else {
                      uVal = plane1[uvOff];
                      vVal = plane2[uvOff];
                    }
                  }
                }
              } else if (plane1 != null) {
                // Dual plane (Y in plane0, interleaved UV/VU in plane1)
                final uvCol = srcX >> 1;
                final p1PixelStride = buffer.planes[1].bytesPerPixel ?? 2;
                final uvOff = uvBase + uvCol * p1PixelStride;

                if (uvOff + 1 < plane1.length) {
                  if (buffer.format == LivenessImageFormat.nv21) {
                    vVal = plane1[uvOff];
                    uVal = plane1[uvOff + 1];
                  } else {
                    uVal = plane1[uvOff];
                    vVal = plane1[uvOff + 1];
                  }
                }
              } else if (plane0.length >= rawW * rawH * 3 ~/ 2) {
                // Single plane packed NV21 / NV12 in plane0
                final uvOff = p0Stride * rawH +
                    (srcY >> 1) * p0Stride +
                    ((srcX >> 1) << 1);
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

              sumY += yVal * w;
              sumU += uVal * w;
              sumV += vVal * w;
            }
            totalWeight += w;
          }
        }

        if (totalWeight > 0) {
          if (isBgra || isRgba) {
            sumR /= totalWeight;
            sumG /= totalWeight;
            sumB /= totalWeight;
          } else {
            sumY /= totalWeight;
            sumU /= totalWeight;
            sumV /= totalWeight;

            // BT.601 YUV→RGB conversion
            final u = sumU - 128.0;
            final v = sumV - 128.0;
            final r = sumY + 1.402 * v;
            final g = sumY - 0.344136 * u - 0.714136 * v;
            final b = sumY + 1.772 * u;

            sumR = r.clamp(0.0, 255.0);
            sumG = g.clamp(0.0, 255.0);
            sumB = b.clamp(0.0, 255.0);
          }
        }

        // Store normalized Float32 values based on NormalizationScheme
        final spatialIdx = y * targetSize + x;
        final c0Val = isBgr ? sumB : sumR;
        final c1Val = sumG;
        final c2Val = isBgr ? sumR : sumB;

        final c0Channel = isBgr ? 2 : 0;
        const c1Channel = 1;
        final c2Channel = isBgr ? 0 : 2;

        final c0Norm = normalizePixel(
          c0Val,
          colorChannel: c0Channel,
          scheme: normalizationScheme,
        );
        final c1Norm = normalizePixel(
          c1Val,
          colorChannel: c1Channel,
          scheme: normalizationScheme,
        );
        final c2Norm = normalizePixel(
          c2Val,
          colorChannel: c2Channel,
          scheme: normalizationScheme,
        );

        if (useNchw) {
          tensor[spatialIdx] = c0Norm;
          tensor[hw + spatialIdx] = c1Norm;
          tensor[2 * hw + spatialIdx] = c2Norm;
        } else {
          final idx = spatialIdx * 3;
          tensor[idx] = c0Norm;
          tensor[idx + 1] = c1Norm;
          tensor[idx + 2] = c2Norm;
        }
      }
    }

    _applyAdaptiveContrastStretch(
      tensor,
      targetSize: targetSize,
      useNchw: useNchw,
      enableContrastStretch: enableContrastStretch,
    );

    return tensor;
  }

  /// Preprocesses a raw RGBA 32-bit pixel byte buffer into a Float32 tensor.
  ///
  /// Stores normalized RGB pixel values in `[0.0, 1.0]` range.
  /// Out-of-boundary pixels use **Edge Pixel Replication**
  /// (`BORDER_REPLICATE` coordinate clamping).
  static Float32List preprocessRgbaBytesToTensor(
    Uint8List rgbaBytes,
    int rawW,
    int rawH, {
    FaceBoundingBox? boundingBox,
    double expansionFactor = defaultExpansionFactor,
    int targetSize = defaultModelSize,
    bool useNchw = true,
    bool isBgr = false,
    NormalizationScheme normalizationScheme = NormalizationScheme.zeroToOne,
    bool enableContrastStretch = false,
  }) {
    final faceBbox = boundingBox ??
        FaceBoundingBox(
          x: 0,
          y: 0,
          width: rawW.toDouble(),
          height: rawH.toDouble(),
        );

    // Strictly 1:1 square crop calculation (baseSide = max(w, h)) preserving
    // trained 37.0% facial scale:
    final boxW = math.max(1, faceBbox.width);
    final boxH = math.max(1, faceBbox.height);
    final baseSide = math.max(boxW, boxH);
    final cropSize = (baseSide * expansionFactor).toInt().toDouble();

    final newW = cropSize;
    final newH = cropSize;

    final cropLeft = (faceBbox.centerX - cropSize / 2.0).toInt().toDouble();
    final cropTop = (faceBbox.centerY - cropSize / 2.0).toInt().toDouble();

    LivenessLogger.logCropStats(
      rawWidth: rawW,
      rawHeight: rawH,
      rotation: 0,
      boundingBox: boundingBox,
      expansionFactor: expansionFactor,
      cropLeft: cropLeft,
      cropTop: cropTop,
      cropWidth: newW,
      cropHeight: newH,
    );

    final tensor = Float32List(1 * targetSize * targetSize * 3);
    final hw = targetSize * targetSize;

    final step = cropSize / targetSize;

    for (var y = 0; y < targetSize; y++) {
      final cy1 = y * step;
      final cy2 = (y + 1) * step;

      for (var x = 0; x < targetSize; x++) {
        final cx1 = x * step;
        final cx2 = (x + 1) * step;

        final rawX1 = cropLeft + cx1;
        final rawX2 = cropLeft + cx2;
        final rawY1 = cropTop + cy1;
        final rawY2 = cropTop + cy2;

        var sumR = 0.0;
        var sumG = 0.0;
        var sumB = 0.0;
        var totalWeight = 0.0;

        final startY = rawY1.floor();
        final endY = rawY2.ceil();
        final startX = rawX1.floor();
        final endX = rawX2.ceil();

        for (var ry = startY; ry < endY; ry++) {
          final yOverlap =
              math.min(ry + 1.0, rawY2) - math.max(ry.toDouble(), rawY1);
          if (yOverlap <= 0) continue;
          final srcY = _reflect101(ry, rawH - 1);
          final yBaseOffset = srcY * rawW;

          for (var rx = startX; rx < endX; rx++) {
            final xOverlap =
                math.min(rx + 1.0, rawX2) - math.max(rx.toDouble(), rawX1);
            final w = xOverlap * yOverlap;
            if (w <= 0) continue;

            final srcX = _reflect101(rx, rawW - 1);

            final offset = (yBaseOffset + srcX) << 2;
            if (offset + 2 < rgbaBytes.length) {
              sumR += rgbaBytes[offset] * w;
              sumG += rgbaBytes[offset + 1] * w;
              sumB += rgbaBytes[offset + 2] * w;
            }
            totalWeight += w;
          }
        }

        if (totalWeight > 0) {
          sumR /= totalWeight;
          sumG /= totalWeight;
          sumB /= totalWeight;
        }

        final spatialIdx = y * targetSize + x;
        final c0Val = isBgr ? sumB : sumR;
        final c1Val = sumG;
        final c2Val = isBgr ? sumR : sumB;

        final c0Channel = isBgr ? 2 : 0;
        const c1Channel = 1;
        final c2Channel = isBgr ? 0 : 2;

        final c0Norm = normalizePixel(
          c0Val,
          colorChannel: c0Channel,
          scheme: normalizationScheme,
        );
        final c1Norm = normalizePixel(
          c1Val,
          colorChannel: c1Channel,
          scheme: normalizationScheme,
        );
        final c2Norm = normalizePixel(
          c2Val,
          colorChannel: c2Channel,
          scheme: normalizationScheme,
        );

        if (useNchw) {
          tensor[spatialIdx] = c0Norm;
          tensor[hw + spatialIdx] = c1Norm;
          tensor[2 * hw + spatialIdx] = c2Norm;
        } else {
          final idx = spatialIdx * 3;
          tensor[idx] = c0Norm;
          tensor[idx + 1] = c1Norm;
          tensor[idx + 2] = c2Norm;
        }
      }
    }

    _applyAdaptiveContrastStretch(
      tensor,
      targetSize: targetSize,
      useNchw: useNchw,
      enableContrastStretch: enableContrastStretch,
    );

    return tensor;
  }

  /// Luma-Preserving Contrast Stretch to prevent RGB sensor noise
  /// amplification.
  static void _applyAdaptiveContrastStretch(
    Float32List tensor, {
    required int targetSize,
    bool useNchw = true,
    bool enableContrastStretch = false,
  }) {
    if (!enableContrastStretch || tensor.isEmpty) return;

    final hw = targetSize * targetSize;
    if (hw == 0) return;

    var minY = 1.0;
    var maxY = 0.0;
    var sumY = 0.0;

    // 1. Calculate relative luma Y = 0.299*R + 0.587*G + 0.114*B and find min/max
    for (var i = 0; i < hw; i++) {
      double r;
      double g;
      double b;
      if (useNchw) {
        r = tensor[i];
        g = tensor[hw + i];
        b = tensor[2 * hw + i];
      } else {
        final idx = i * 3;
        r = tensor[idx];
        g = tensor[idx + 1];
        b = tensor[idx + 2];
      }

      final yVal = 0.299 * r + 0.587 * g + 0.114 * b;
      if (yVal < minY) minY = yVal;
      if (yVal > maxY) maxY = yVal;
      sumY += yVal;
    }

    final meanY = sumY / hw;

    // 2. If crop is too dark (< 0.40 mean luma), apply gamma correction
    if (meanY < 0.40 && meanY > 0.01) {
      // Calculate gamma to lift the mean luma to roughly 0.5
      final gamma = math.log(0.5) / math.log(meanY);

      for (var i = 0; i < hw; i++) {
        double r;
        double g;
        double b;
        if (useNchw) {
          r = tensor[i];
          g = tensor[hw + i];
          b = tensor[2 * hw + i];
        } else {
          final idx = i * 3;
          r = tensor[idx];
          g = tensor[idx + 1];
          b = tensor[idx + 2];
        }

        final yVal = 0.299 * r + 0.587 * g + 0.114 * b;

        if (yVal > 0) {
          final newY = math.pow(yVal, gamma).toDouble();
          final ratio = newY / (yVal + 1e-7);

          final newR = (r * ratio).clamp(0.0, 1.0);
          final newG = (g * ratio).clamp(0.0, 1.0);
          final newB = (b * ratio).clamp(0.0, 1.0);

          if (useNchw) {
            tensor[i] = newR;
            tensor[hw + i] = newG;
            tensor[2 * hw + i] = newB;
          } else {
            final idx = i * 3;
            tensor[idx] = newR;
            tensor[idx + 1] = newG;
            tensor[idx + 2] = newB;
          }
        }
      }
    }
  }

  /// Saves a Float32List tensor (128x128) as a PNG/PPM image file on disk.
  ///
  /// Supports NCHW (`isNchw: true`) and NHWC (`isNchw: false`) layouts.
  /// Default save path: `/tmp/liveness_tensor_debug.png`.
  static Future<File> saveTensorToDisk(
    Float32List tensor, {
    bool isNchw = true,
    String? filePath,
    int targetSize = defaultModelSize,
  }) async {
    final path = filePath ?? '/tmp/liveness_tensor_debug.png';
    final file = File(path);
    await file.parent.create(recursive: true);

    final hw = targetSize * targetSize;
    final rgbaBytes = Uint8List(hw * 4);

    for (var y = 0; y < targetSize; y++) {
      for (var x = 0; x < targetSize; x++) {
        final spatialIdx = y * targetSize + x;
        final outIdx = spatialIdx * 4;

        double rVal;
        double gVal;
        double bVal;

        if (isNchw) {
          rVal = tensor[spatialIdx];
          gVal = tensor[hw + spatialIdx];
          bVal = tensor[2 * hw + spatialIdx];
        } else {
          final inIdx = spatialIdx * 3;
          rVal = tensor[inIdx];
          gVal = tensor[inIdx + 1];
          bVal = tensor[inIdx + 2];
        }

        // Scale [0.0, 1.0] to [0, 255] if normalized
        final rPixel = rVal <= 1.0 ? (rVal * 255.0) : rVal;
        final gPixel = gVal <= 1.0 ? (gVal * 255.0) : gVal;
        final bPixel = bVal <= 1.0 ? (bVal * 255.0) : bVal;

        rgbaBytes[outIdx] = rPixel.round().clamp(0, 255);
        rgbaBytes[outIdx + 1] = gPixel.round().clamp(0, 255);
        rgbaBytes[outIdx + 2] = bPixel.round().clamp(0, 255);
        rgbaBytes[outIdx + 3] = 255;
      }
    }

    if (path.endsWith('.ppm')) {
      final header = 'P6\n$targetSize $targetSize\n255\n';
      final rgbBytes = Uint8List(hw * 3);
      for (var i = 0; i < hw; i++) {
        rgbBytes[i * 3] = rgbaBytes[i * 4];
        rgbBytes[i * 3 + 1] = rgbaBytes[i * 4 + 1];
        rgbBytes[i * 3 + 2] = rgbaBytes[i * 4 + 2];
      }
      final builder = BytesBuilder()
        ..add(header.codeUnits)
        ..add(rgbBytes);
      await file.writeAsBytes(builder.toBytes());
    } else {
      final descriptor = ui.ImageDescriptor.raw(
        await ui.ImmutableBuffer.fromUint8List(rgbaBytes),
        width: targetSize,
        height: targetSize,
        pixelFormat: ui.PixelFormat.rgba8888,
      );
      final codec = await descriptor.instantiateCodec();
      final frameInfo = await codec.getNextFrame();
      final pngByteData = await frameInfo.image.toByteData(
        format: ui.ImageByteFormat.png,
      );
      if (pngByteData != null) {
        await file.writeAsBytes(pngByteData.buffer.asUint8List());
      }
      frameInfo.image.dispose();
      codec.dispose();
      descriptor.dispose();
    }

    return file;
  }

  /// Extracts an un-downscaled grayscale crop (default target size 256x256)
  /// of the center face region for high-frequency micro-texture analysis.
  static Uint8List extractHighResCrop(
    LivenessImageBuffer buffer, {
    FaceBoundingBox? boundingBox,
    int rotation = 0,
    bool? isRotatedBoundingBox,
    int targetSize = 256,
  }) {
    final rawW = buffer.width;
    final rawH = buffer.height;
    final normRotation = ((rotation % 360) + 360) % 360;

    final faceBbox = boundingBox ??
        FaceBoundingBox(
          x: 0,
          y: 0,
          width: rawW.toDouble(),
          height: rawH.toDouble(),
        );

    final isRotated = isRotatedBoundingBox ??
        (Platform.isAndroid && (normRotation == 90 || normRotation == 270)) ||
            ((normRotation == 90 || normRotation == 270) &&
                (faceBbox.centerY > rawH || faceBbox.centerX > rawW));
    final effectiveBbox = isRotated
        ? faceBbox.toRawBufferSpace(rawW, rawH, normRotation)
        : faceBbox;

    final boxW = math.max(1, effectiveBbox.width);
    final boxH = math.max(1, effectiveBbox.height);
    final baseSide = math.max(boxW, boxH);
    final cropSize =
        baseSide; // 1.0x tight center face crop for micro-texture analysis

    final cropLeft =
        (effectiveBbox.centerX - cropSize / 2.0).toInt().toDouble();
    final cropTop = (effectiveBbox.centerY - cropSize / 2.0).toInt().toDouble();

    final resultBytes = Uint8List(targetSize * targetSize);
    final isBgra = buffer.format == LivenessImageFormat.bgra8888;
    final isRgba = buffer.format == LivenessImageFormat.rgba8888;
    final plane0 = buffer.planes[0].bytes;
    final p0Stride = buffer.planes[0].bytesPerRow;
    final step = cropSize / targetSize;

    for (var y = 0; y < targetSize; y++) {
      final cy = cropTop + y * step;
      for (var x = 0; x < targetSize; x++) {
        final cx = cropLeft + x * step;

        var rawX = cx;
        var rawY = cy;

        switch (normRotation) {
          case 90:
            rawX = cropLeft + cy;
            rawY = cropTop + cropSize - cx;
          case 180:
            rawX = cropLeft + cropSize - cx;
            rawY = cropTop + cropSize - cy;
          case 270:
            rawX = cropLeft + cropSize - cy;
            rawY = cropTop + cx;
          case 0:
          default:
            rawX = cx;
            rawY = cy;
        }

        final srcX = _reflect101(rawX.round(), rawW - 1);
        final srcY = _reflect101(rawY.round(), rawH - 1);

        var gray = 0;
        if (isBgra || isRgba) {
          final offset = srcY * p0Stride + (srcX << 2);
          if (offset + 2 < plane0.length) {
            final b = isBgra ? plane0[offset] : plane0[offset + 2];
            final g = plane0[offset + 1];
            final r = isBgra ? plane0[offset + 2] : plane0[offset];
            gray = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0, 255);
          }
        } else {
          final yIdx = srcY * p0Stride + srcX;
          if (yIdx < plane0.length) {
            gray = plane0[yIdx];
          }
        }

        resultBytes[y * targetSize + x] = gray;
      }
    }

    return resultBytes;
  }
}
