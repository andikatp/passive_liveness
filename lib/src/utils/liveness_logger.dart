import 'package:flutter/foundation.dart';
import 'package:passive_liveness/src/models/face_bounding_box.dart';
import 'package:passive_liveness/src/models/liveness_result.dart';

/// Centralized diagnostic logging utility for passive face liveness detection.
class LivenessLogger {
  const LivenessLogger._();

  /// Global flag to enable or disable diagnostic print/developer logging.
  static bool enableLogging = true;

  /// Log a message with standard `[PASSIVE_LIVENESS]` prefix.
  static void log(String message) {
    if (!enableLogging) return;
    debugPrint('[PASSIVE_LIVENESS] $message');
  }

  /// Log model initialization specs.
  static void logModelInit({
    required List<int>? inputShape,
    required String tensorType,
    required bool isNativeNchw,
    required int targetSize,
    String engineName = 'Native Platform Channel (Play Services / TFLiteSwift)',
  }) {
    if (!enableLogging) return;
    final fmtStr = isNativeNchw
        ? 'NCHW [1, 3, $targetSize, $targetSize]'
        : 'NHWC [1, $targetSize, $targetSize, 3]';
    log(
      'Model Init -> Shape: $inputShape | Type: $tensorType | Format: $fmtStr '
      '| TargetSize: $targetSize | Engine: $engineName',
    );
  }

  /// Log image buffer crop and bounding box metrics.
  static void logCropStats({
    required int rawWidth,
    required int rawHeight,
    required int rotation,
    required FaceBoundingBox? boundingBox,
    required double expansionFactor,
    required double cropLeft,
    required double cropTop,
    required double cropWidth,
    required double cropHeight,
  }) {
    if (!enableLogging) return;
    final bboxStr = boundingBox != null
        ? 'x: ${boundingBox.x.toStringAsFixed(1)}, '
            'y: ${boundingBox.y.toStringAsFixed(1)}, '
            'w: ${boundingBox.width.toStringAsFixed(1)}, '
            'h: ${boundingBox.height.toStringAsFixed(1)}'
        : 'Full Frame';
    final expStr = expansionFactor.toStringAsFixed(2);
    final lStr = cropLeft.toStringAsFixed(1);
    final tStr = cropTop.toStringAsFixed(1);
    final wStr = cropWidth.toStringAsFixed(1);
    final hStr = cropHeight.toStringAsFixed(1);
    log(
      'Crop Stats -> Raw: ${rawWidth}x$rawHeight (rot: $rotation°) | '
      'FaceBox: [$bboxStr] | Expansion: ${expStr}x | '
      'CropRegion: (L: $lStr, T: $tStr, W: $wStr, H: $hStr)',
    );
  }

  /// Log input tensor stats (min, max, mean, length, input shape).
  static void logTensorStats(Float32List tensorData, {List<int>? inputShape}) {
    if (!enableLogging || tensorData.isEmpty) return;
    var minVal = tensorData[0];
    var maxVal = tensorData[0];
    var sumVal = 0.0;
    for (var i = 0; i < tensorData.length; i++) {
      final v = tensorData[i];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
      sumVal += v;
    }
    final meanVal = sumVal / tensorData.length;
    final minStr = minVal.toStringAsFixed(4);
    final maxStr = maxVal.toStringAsFixed(4);
    final meanStr = meanVal.toStringAsFixed(4);
    log(
      'Input Tensor -> Shape: $inputShape | Stats -> min: $minStr, '
      'max: $maxStr, mean: $meanStr, len: ${tensorData.length}',
    );
  }

  /// Log inference logits, softmax scores, EMA smoothing, threshold,
  /// luminance, and time.
  static void logInferenceResult({
    required double realLogit,
    required double spoofLogit,
    required double logitDiff,
    required double currentRealProb,
    required double? emaRealScore,
    required bool isReal,
    required LivenessStatus status,
    required double threshold,
    required Duration inferenceTime,
    double? meanLuminance,
    bool isLowLight = false,
  }) {
    if (!enableLogging) return;
    final emaStr =
        emaRealScore != null ? emaRealScore.toStringAsFixed(4) : 'none';
    final lumaStr = meanLuminance != null
        ? ' | Luma: ${meanLuminance.toStringAsFixed(1)}'
            '${isLowLight ? " (LowLight)" : ""}'
        : '';
    final rLogit = realLogit.toStringAsFixed(4);
    final sLogit = spoofLogit.toStringAsFixed(4);
    final diffStr = logitDiff.toStringAsFixed(4);
    final probStr = currentRealProb.toStringAsFixed(4);
    final thStr = threshold.toStringAsFixed(2);
    log(
      'Inference Result -> Logits: [real: $rLogit, spoof: $sLogit] | '
      'LogitDiff: $diffStr | CurrentProb: $probStr | EMA: $emaStr$lumaStr | '
      'Status: ${status.name.toUpperCase()} (isReal: $isReal, '
      'threshold: $thStr) | Time: ${inferenceTime.inMilliseconds}ms',
    );
  }

  /// Log motion stability metrics for streaming camera frames.
  static void logMotionStability({
    required bool isStable,
    required double dx,
    required double dy,
    required double dw,
    required double dh,
  }) {
    if (!enableLogging) return;
    final statusStr = isStable ? 'STABLE' : 'UNSTABLE (Motion Detected)';
    final dxStr = dx.toStringAsFixed(3);
    final dyStr = dy.toStringAsFixed(3);
    final dwStr = dw.toStringAsFixed(3);
    final dhStr = dh.toStringAsFixed(3);
    log(
      'Motion Check -> $statusStr | Deltas -> dx: $dxStr, dy: $dyStr, '
      'dw: $dwStr, dh: $dhStr',
    );
  }
}
