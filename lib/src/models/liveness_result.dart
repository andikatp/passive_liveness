import 'dart:math' as math;

/// Status of passive liveness detection result.
enum LivenessStatus {
  /// Classified as a real face.
  real,

  /// Classified as a spoof attempt.
  spoof,

  /// Face is too far from the camera lens.
  tooFar,

  /// Face is too close to the camera lens.
  tooClose,

  /// Face bounding box has an unnatural aspect ratio.
  invalidAspectRatio,

  /// Detected high-frequency inkjet/paper print texture patterns (LBP spoof).
  printSpoof,

  /// Detected digital screen sub-pixel grid lines
  /// or YCbCr chrominance anomalies.
  screenReplaySpoof,
}

/// Result of passive liveness detection for a face.
class LivenessResult {
  /// Creates a [LivenessResult] containing complete classification metrics.
  const LivenessResult({
    required this.isReal,
    required this.status,
    required this.realScore,
    required this.spoofScore,
    required this.realLogit,
    required this.spoofLogit,
    required this.logitDiff,
    required this.confidence,
    required this.threshold,
    required this.inferenceTime,
    double? rawRealScore,
    double? rawSpoofScore,
    double? rawLogitDiff,
    bool? rawIsReal,
    this.lbpUniformityScore,
    this.hogGridDominance,
    this.faceAreaRatio,
    this.chrominanceVariance,
    this.laplacianDelta,
    this.saturationVariance,
    this.moireHighFreqRatio,
    this.meanLuminance,
    this.isLowLight = false,
  })  : rawRealScore = rawRealScore ?? realScore,
        rawSpoofScore = rawSpoofScore ?? spoofScore,
        rawLogitDiff = rawLogitDiff ?? logitDiff,
        rawIsReal = rawIsReal ?? isReal;

  /// Factory constructor for pending/unstable frames.
  factory LivenessResult.pending({
    double threshold = 0.0,
    LivenessStatus status = LivenessStatus.spoof,
    double? meanLuminance,
    bool isLowLight = false,
  }) {
    return LivenessResult(
      isReal: false,
      status: status,
      realScore: 0.5,
      spoofScore: 0.5,
      realLogit: 0,
      spoofLogit: 0,
      logitDiff: 0,
      confidence: 0,
      threshold: threshold,
      inferenceTime: Duration.zero,
      meanLuminance: meanLuminance,
      isLowLight: isLowLight,
    );
  }

  /// Factory constructor to calculate probabilities and classification
  /// from raw logits.
  factory LivenessResult.fromLogits({
    required double realLogit,
    required double spoofLogit,
    double threshold = 0.0,
    Duration inferenceTime = Duration.zero,
    double? lbpUniformityScore,
    double? hogGridDominance,
    double? faceAreaRatio,
    double? chrominanceVariance,
    double? laplacianDelta,
    double? saturationVariance,
    double? moireHighFreqRatio,
    double? meanLuminance,
    bool isLowLight = false,
    LivenessStatus? overrideStatus,
  }) {
    final logitDiff = realLogit - spoofLogit;
    final isRealCalculated = logitDiff >= threshold;
    final status = overrideStatus ??
        (isRealCalculated ? LivenessStatus.real : LivenessStatus.spoof);
    final isReal = status == LivenessStatus.real;

    // Numerically stable softmax
    final maxLogit = math.max(realLogit, spoofLogit);
    final expReal = math.exp(realLogit - maxLogit);
    final expSpoof = math.exp(spoofLogit - maxLogit);
    final sumExp = expReal + expSpoof;

    final realScore = expReal / sumExp;
    final spoofScore = expSpoof / sumExp;
    final confidence = logitDiff.abs();

    return LivenessResult(
      isReal: isReal,
      status: status,
      realScore: realScore,
      spoofScore: spoofScore,
      realLogit: realLogit,
      spoofLogit: spoofLogit,
      logitDiff: logitDiff,
      confidence: confidence,
      threshold: threshold,
      inferenceTime: inferenceTime,
      lbpUniformityScore: lbpUniformityScore,
      hogGridDominance: hogGridDominance,
      faceAreaRatio: faceAreaRatio,
      chrominanceVariance: chrominanceVariance,
      laplacianDelta: laplacianDelta,
      saturationVariance: saturationVariance,
      moireHighFreqRatio: moireHighFreqRatio,
      meanLuminance: meanLuminance,
      isLowLight: isLowLight,
    );
  }

  /// Whether the face is classified as real (live person).
  final bool isReal;

  /// Human-readable status classification (`real`, `spoof`, `tooFar`, etc.).
  final LivenessStatus status;

  /// Softmax probability of being a real face (0.0 to 1.0).
  final double realScore;

  /// Softmax probability of being a spoof face (0.0 to 1.0).
  final double spoofScore;

  /// Raw logit score for real class from ONNX model.
  final double realLogit;

  /// Raw logit score for spoof class from ONNX model.
  final double spoofLogit;

  /// Logit difference (`realLogit - spoofLogit`).
  final double logitDiff;

  /// Absolute confidence metric based on logit difference.
  final double confidence;

  /// Logit threshold used for classification.
  final double threshold;

  /// Time taken to execute preprocessing and model inference.
  final Duration inferenceTime;

  /// Instant single-frame softmax probability of being real
  /// (before EMA smoothing).
  final double rawRealScore;

  /// Instant single-frame softmax probability of being spoof
  /// (before EMA smoothing).
  final double rawSpoofScore;

  /// Instant single-frame logit difference (`realLogit - spoofLogit`).
  final double rawLogitDiff;

  /// Whether the instant single-frame model output is real
  /// (`rawLogitDiff >= threshold`).
  final bool rawIsReal;

  /// Optional LBP micro-texture non-uniform pattern ratio score
  /// (high = print artifact).
  final double? lbpUniformityScore;

  /// Optional HOG dominant orientation grid energy score
  /// (high = screen grid artifact).
  final double? hogGridDominance;

  /// Ratio of face bounding box area to total camera frame area.
  final double? faceAreaRatio;

  /// Chrominance variance ($\sigma^2_{CbCr}$) metric in YCbCr color space.
  final double? chrominanceVariance;

  /// Normalized Laplacian delta between face and background regions
  /// ($0.0 \dots \infty$).
  ///
  /// Low values ($< 0.15$) indicate a 2D flat focal plane (screen/print).
  /// High values indicate natural 3D depth-of-field variations.
  final double? laplacianDelta;

  /// HSV Saturation channel variance ($0.0 \dots 1.0$).
  ///
  /// High values ($\ge 0.025$) with elevated chrominance indicate
  /// emissive screen backlight.
  final double? saturationVariance;

  /// High-frequency energy ratio from FFT moiré analysis ($0.0 \dots 1.0$).
  ///
  /// High values ($\ge 0.35$) with structural regularity peaks indicate
  /// screen moiré.
  final double? moireHighFreqRatio;

  /// Mean luminance (brightness) level of analyzed face crop
  /// ($0.0 \dots 255.0$).
  final double? meanLuminance;

  /// Whether the frame was captured under low-light conditions
  /// (below the configured threshold).
  final bool isLowLight;

  /// Convert result to a JSON map representation.
  Map<String, dynamic> toJson() => {
        'isReal': isReal,
        'status': status.name,
        'realScore': realScore,
        'spoofScore': spoofScore,
        'realLogit': realLogit,
        'spoofLogit': spoofLogit,
        'logitDiff': logitDiff,
        'confidence': confidence,
        'threshold': threshold,
        'inferenceTimeMs': inferenceTime.inMilliseconds,
        'isLowLight': isLowLight,
        if (meanLuminance != null) 'meanLuminance': meanLuminance,
        if (lbpUniformityScore != null)
          'lbpUniformityScore': lbpUniformityScore,
        if (hogGridDominance != null) 'hogGridDominance': hogGridDominance,
        if (faceAreaRatio != null) 'faceAreaRatio': faceAreaRatio,
        if (chrominanceVariance != null)
          'chrominanceVariance': chrominanceVariance,
        if (laplacianDelta != null) 'laplacianDelta': laplacianDelta,
        if (saturationVariance != null)
          'saturationVariance': saturationVariance,
        if (moireHighFreqRatio != null)
          'moireHighFreqRatio': moireHighFreqRatio,
      };

  @override
  String toString() {
    final lum = meanLuminance?.toStringAsFixed(1) ?? 'N/A';
    return 'LivenessResult(isReal: $isReal, status: ${status.name}, '
        'realScore: ${realScore.toStringAsFixed(4)}, '
        'logitDiff: ${logitDiff.toStringAsFixed(4)}, '
        'luminance: $lum, isLowLight: $isLowLight, '
        'inferenceTimeMs: ${inferenceTime.inMilliseconds})';
  }
}

/// Alias for [LivenessResult] to prevent class name collisions
/// in host applications.
typedef PassiveLivenessResult = LivenessResult;
