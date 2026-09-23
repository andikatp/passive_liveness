import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/services.dart';

import 'package:passive_liveness/src/models/face_bounding_box.dart';
import 'package:passive_liveness/src/models/liveness_image_buffer.dart';
import 'package:passive_liveness/src/models/liveness_result.dart';
import 'package:passive_liveness/src/native_liveness_engine.dart';
import 'package:passive_liveness/src/utils/color_space_analyzer.dart';
import 'package:passive_liveness/src/utils/face_proximity_gate.dart';
import 'package:passive_liveness/src/utils/fft_moire_analyzer.dart';
import 'package:passive_liveness/src/utils/high_res_screen_analyzer.dart';
import 'package:passive_liveness/src/utils/image_preprocessor.dart';
import 'package:passive_liveness/src/utils/lbp_hog_analyzer.dart';
import 'package:passive_liveness/src/utils/liveness_logger.dart';

/// Core passive liveness detector engine using native TFLite inference.
///
/// Uses native platform channels for model inference:
/// - **Android**: Google Play Services TFLite (0 MB size impact)
/// - **iOS**: TensorFlowLiteSwift (~3-5 MB size impact)
///
/// All image preprocessing and heuristic analysis runs in Dart.
/// Only the neural network inference is delegated to native code.
class PassiveLivenessDetector {
  /// Creates a new [PassiveLivenessDetector] instance.
  PassiveLivenessDetector();

  /// Default asset path for the TFLite model package asset.
  static const String defaultAssetPath =
      'packages/passive_liveness/assets/best_model.tflite';

  /// Fallback asset path when loaded directly within host application.
  static const String fallbackAssetPath = 'assets/best_model.tflite';

  /// Default Exponential Moving Average (EMA) alpha for score smoothing.
  static const double defaultEmaAlpha = 0.3;

  /// Default recommended luminance threshold (0..255) for low-light face
  /// auto-acceptance.
  static const double defaultLowLightThreshold = 70;

  final NativeLivenessEngine _engine = NativeLivenessEngine();

  bool _isInitialized = false;

  /// Input tensor shape of the loaded model
  /// (e.g. [1, 128, 128, 3] or [1, 3, 128, 128]).
  List<int>? get modelInputShape => _engine.modelInputShape;

  /// Whether the model natively expects NCHW format (`[1, 3, H, W]`).
  bool get isNativeNchw => _engine.isNativeNchw;

  /// Target spatial resolution expected by model (e.g. 128 or 80).
  int get modelTargetSize => _engine.modelTargetSize;

  /// The current Exponential Moving Average (EMA) of the real score.
  double? _emaRealScore;

  /// Returns the current EMA real score.
  double? get emaRealScore => _emaRealScore;

  /// Resets the EMA real score tracker.
  void resetEma() {
    _emaRealScore = null;
  }

  /// Whether the detector is initialized and ready for inference.
  bool get isInitialized => _isInitialized && _engine.isModelLoaded;

  FaceBoundingBox? _resolveRawBoundingBox(
    LivenessImageBuffer buffer, {
    required int rotation,
    FaceBoundingBox? boundingBox,
    bool? isRotatedBoundingBox,
  }) {
    if (boundingBox == null) return null;

    final normRotation = ((rotation % 360) + 360) % 360;
    final isRotated = isRotatedBoundingBox ??
        (Platform.isAndroid && (normRotation == 90 || normRotation == 270)) ||
            ((normRotation == 90 || normRotation == 270) &&
                (boundingBox.centerY > buffer.height ||
                    boundingBox.centerX > buffer.width));

    return isRotated
        ? boundingBox.toRawBufferSpace(
            buffer.width,
            buffer.height,
            normRotation,
          )
        : boundingBox;
  }

  /// Initialize the native TFLite model for passive liveness detection.
  ///
  /// Loads the model bytes and sends them to the native platform (Android/iOS)
  /// for interpreter initialization. On Android, this uses Google Play Services
  /// TFLite runtime (0 MB APK impact). On iOS, it uses TensorFlowLiteSwift.
  Future<void> initialize({
    String? assetPath,
    String? filePath,
    Uint8List? modelBytes,
    ModelClassOrder classOrder = ModelClassOrder.realFirst,
  }) async {
    if (_isInitialized) return;

    Uint8List bytes;

    if (modelBytes != null) {
      bytes = modelBytes;
    } else if (filePath != null) {
      bytes = await File(filePath).readAsBytes();
    } else {
      final path = assetPath ?? defaultAssetPath;
      try {
        final bd = await rootBundle.load(path);
        bytes = bd.buffer.asUint8List(bd.offsetInBytes, bd.lengthInBytes);
      } on Exception catch (_) {
        final bd = await rootBundle.load(fallbackAssetPath);
        bytes = bd.buffer.asUint8List(bd.offsetInBytes, bd.lengthInBytes);
      }
    }

    await _engine.loadModel(bytes, classOrder: classOrder);

    LivenessLogger.logModelInit(
      inputShape: _engine.modelInputShape,
      tensorType: 'float32',
      isNativeNchw: _engine.isNativeNchw,
      targetSize: _engine.modelTargetSize,
    );

    _isInitialized = true;
  }

  Future<List<double>> _runInference({required Float32List tensorData}) async {
    return await _engine.runInference(tensorData);
  }

  /// Evaluates liveness directly from a Flutter `CameraImage` instance.
  ///
  /// Converts [cameraImage] to [LivenessImageBuffer] and invokes
  /// [detectLivenessFromBuffer].
  Future<LivenessResult> detectLivenessFromCameraImage(
    dynamic cameraImage, {
    FaceBoundingBox? boundingBox,
    int rotation = 0,
    bool? isRotatedBoundingBox,
    double threshold = 0.0,
    double expansionFactor = ImagePreprocessor.defaultExpansionFactor,
    double emaAlpha = defaultEmaAlpha,
    bool enableProximityGate = true,
    bool enableTextureAnalysis = true,
    bool enableColorSpaceAnalysis = true,
    bool enableHighResScreenAnalysis = true,
    bool enableMoireAnalysis = true,
    double? lowLightThreshold,
    bool? isBgr,
    NormalizationScheme normalizationScheme = NormalizationScheme.zeroToOne,
    bool? enableContrastStretch,
  }) {
    final buffer = LivenessImageBuffer.fromCameraImage(cameraImage);
    return detectLivenessFromBuffer(
      buffer,
      boundingBox: boundingBox,
      rotation: rotation,
      isRotatedBoundingBox: isRotatedBoundingBox,
      threshold: threshold,
      expansionFactor: expansionFactor,
      emaAlpha: emaAlpha,
      enableProximityGate: enableProximityGate,
      enableTextureAnalysis: enableTextureAnalysis,
      enableColorSpaceAnalysis: enableColorSpaceAnalysis,
      enableHighResScreenAnalysis: enableHighResScreenAnalysis,
      enableMoireAnalysis: enableMoireAnalysis,
      lowLightThreshold: lowLightThreshold,
      isBgr: isBgr,
      normalizationScheme: normalizationScheme,
      enableContrastStretch: enableContrastStretch,
    );
  }

  /// Runs passive face liveness and anti-spoofing detection directly on a
  /// raw camera image buffer ([LivenessImageBuffer]).
  ///
  /// Returns a [LivenessResult] containing `isReal`, classification `status`,
  /// real/spoof confidence scores, and detailed physical heuristic metrics.
  ///
  /// ### Simple Usage (Recommended)
  /// All multi-layer anti-spoofing heuristic engines (Proximity Gate,
  /// Micro-Texture Analysis, YCbCr Color Space Analysis, and 2D Laplacian
  /// High-Res Screen Analysis) are **enabled by default** with optimal
  /// production settings. You only need to pass the camera [buffer] and
  /// optional [boundingBox] & [rotation]:
  ///
  /// ```dart
  /// final result = await detector.detectLivenessFromBuffer(
  ///   buffer,
  ///   boundingBox: faceBbox, // Optional (crops face if provided)
  ///   rotation: sensorRotation, // Sensor rotation in degrees (0, 90, 180, 270)
  /// );
  ///
  /// if (result.isReal) {
  ///   print('Real face verified!');
  /// } else {
  ///   print('Spoof detected: ${result.status.name}');
  /// }
  /// ```
  ///
  /// ### Parameters:
  /// - [buffer]: The raw camera frame image plane buffer (`NV21`, `YUV420`,
  ///   or `BGRA8888`).
  /// - [boundingBox]: Optional facial bounding box from a face detector.
  /// - [rotation]: Sensor rotation angle in degrees (`0`, `90`, `180`, `270`).
  /// - [isRotatedBoundingBox]: Whether bounding box coordinates are in rotated
  ///   frame space.
  /// - [expansionFactor]: Square crop margin around bounding box
  ///   (default `1.5x`).
  /// - [emaAlpha]: Exponential Moving Average smoothing factor (default `0.3`).
  /// - [enableProximityGate]: Enable face area coverage validation.
  /// - [enableTextureAnalysis]: Enable LBP halftone & HOG grid analysis.
  /// - [enableColorSpaceAnalysis]: Enable YCbCr chrominance variance analysis.
  /// - [enableHighResScreenAnalysis]: Enable 2D Laplacian focus depth analysis.
  Future<LivenessResult> detectLivenessFromBuffer(
    LivenessImageBuffer buffer, {
    FaceBoundingBox? boundingBox,
    int rotation = 0,
    bool? isRotatedBoundingBox,
    double threshold = 0.0,
    double expansionFactor = ImagePreprocessor.defaultExpansionFactor,
    double emaAlpha = defaultEmaAlpha,
    bool enableProximityGate = true,
    bool enableTextureAnalysis = true,
    bool enableColorSpaceAnalysis = true,
    bool enableHighResScreenAnalysis = true,
    bool enableMoireAnalysis = true,
    double? lowLightThreshold,
    bool? isBgr,
    NormalizationScheme normalizationScheme = NormalizationScheme.zeroToOne,
    bool? enableContrastStretch,
  }) async {
    const proximityGate = FaceProximityGate();
    const textureAnalyzer = LbpHogAnalyzer();
    const colorSpaceAnalyzer = ColorSpaceAnalyzer();
    const highResScreenAnalyzer = HighResScreenAnalyzer();
    const moireAnalyzer = FftMoireAnalyzer();
    if (!isInitialized) {
      throw StateError(
        'PassiveLivenessDetector is not initialized. Call initialize() first.',
      );
    }

    final stopwatch = Stopwatch()..start();
    final rawBoundingBox = _resolveRawBoundingBox(
      buffer,
      boundingBox: boundingBox,
      rotation: rotation,
      isRotatedBoundingBox: isRotatedBoundingBox,
    );

    Uint8List? cachedHighResCrop;
    Uint8List getHighResCrop() {
      return cachedHighResCrop ??= ImagePreprocessor.extractHighResCrop(
        buffer,
        boundingBox: boundingBox,
        rotation: rotation,
        isRotatedBoundingBox: isRotatedBoundingBox,
      );
    }

    // Measure mean luminance (brightness) of the face crop
    final faceCrop = getHighResCrop();
    var sumLuma = 0;
    for (var i = 0; i < faceCrop.length; i++) {
      sumLuma += faceCrop[i];
    }
    final meanLuminance =
        faceCrop.isNotEmpty ? sumLuma / faceCrop.length : null;
    final effectiveLowLightThreshold = lowLightThreshold != null
        ? (lowLightThreshold <= 1.0
            ? lowLightThreshold * 255.0
            : lowLightThreshold)
        : null;
    final isLowLight = effectiveLowLightThreshold != null &&
        meanLuminance != null &&
        meanLuminance < effectiveLowLightThreshold;

    // 1. Proximity & Aspect Ratio Gate Check
    double? faceAreaRatio;
    if (enableProximityGate && rawBoundingBox != null) {
      final gateResult = proximityGate.evaluate(
        boundingBox: rawBoundingBox,
        frameWidth: buffer.width,
        frameHeight: buffer.height,
        rotation: rotation,
      );
      faceAreaRatio = gateResult.faceAreaRatio;

      if (!gateResult.isValid) {
        stopwatch.stop();
        return LivenessResult.pending(
          threshold: threshold,
          status: gateResult.status,
          meanLuminance: meanLuminance,
          isLowLight: isLowLight,
        );
      }
    }

    // 2. Micro-Texture LBP / HOG Post-Processing
    double? lbpRatio;
    double? hogDominance;
    var isPrintSpoof = false;
    var isScreenGridSpoof = false;

    if (enableTextureAnalysis) {
      final highResCrop = getHighResCrop();
      final textureResult = textureAnalyzer.analyzeGrayscaleCrop(
        highResCrop,
        256,
        256,
      );
      lbpRatio = textureResult.lbpNonUniformRatio;
      hogDominance = textureResult.hogPeakDominance;
      isPrintSpoof = textureResult.isPrintSpoof;
      isScreenGridSpoof = textureResult.isScreenGridSpoof;
    }

    // 3. YCbCr Chrominance Variance & HSV Saturation Analysis
    double? chrominanceVar;
    double? saturationVar;
    var isScreenReplaySpoof = false;
    var isEmissiveSaturationSpoof = false;

    if (enableColorSpaceAnalysis) {
      final colorResult = colorSpaceAnalyzer.analyzeBuffer(
        buffer,
        boundingBox: rawBoundingBox,
      );
      chrominanceVar = colorResult.chrominanceVariance;
      isScreenReplaySpoof = colorResult.isScreenReplaySpoof;
      saturationVar = colorResult.saturationVariance;
      isEmissiveSaturationSpoof = colorResult.isEmissiveSaturationSpoof;
    }

    // Flat paper print photo attack detection (low chrominance variance +
    // LBP degradation + HOG halftone grid):
    if (lbpRatio != null &&
        lbpRatio < 0.250 &&
        chrominanceVar != null &&
        chrominanceVar >= 50.0 &&
        chrominanceVar < 80.0 &&
        hogDominance != null &&
        hogDominance >= 0.160) {
      isPrintSpoof = true;
    }

    // 4. 2D Laplacian Frequency & Focus Depth Analysis for High-Res Screens
    var isHighResScreenSpoof = false;
    var is2DFlatSpoof = false;
    double? laplacianVar;
    double? specularRatio;
    double? laplacianDelta;

    if (enableHighResScreenAnalysis) {
      final highResCrop = getHighResCrop();
      final highResResult = highResScreenAnalyzer.analyzeGrayscaleCrop(
        highResCrop,
        256,
        256,
      );
      isHighResScreenSpoof = highResResult.isHighResScreenSpoof;
      laplacianVar = highResResult.laplacianVariance;
      specularRatio = highResResult.specularHighlightRatio;
      is2DFlatSpoof = highResResult.is2DFlatSpoof;
      laplacianDelta = (highResResult.faceLaplacianVariance +
                  highResResult.backgroundLaplacianVariance) >
              0.0
          ? (highResResult.faceLaplacianVariance -
                      highResResult.backgroundLaplacianVariance)
                  .abs() /
              ((highResResult.faceLaplacianVariance +
                      highResResult.backgroundLaplacianVariance) /
                  2.0)
          : 0.0;
    }

    // 5. FFT Moiré Pattern Frequency Analysis
    double? moireHighFreqRatio;
    var isMoireSpoof = false;

    if (enableMoireAnalysis) {
      final highResCrop = getHighResCrop();
      final moireResult = moireAnalyzer.analyzeGrayscaleCrop(
        highResCrop,
        256,
        256,
      );
      moireHighFreqRatio = moireResult.highFrequencyRatio;
      isMoireSpoof = moireResult.isMoireSpoof;
    }

    // Emissive screen replay detection (MacBook / OLED / LCD digital display
    // re-photography):
    // Digital displays emit sub-pixel high-frequency energy combined with
    // chrominance dispersion.
    final isEmissiveScreenSpoof = chrominanceVar != null &&
        chrominanceVar >= 80.0 &&
        ((laplacianVar != null && laplacianVar >= 2500.0) ||
            (specularRatio != null &&
                specularRatio >= 0.0050 &&
                laplacianVar != null &&
                laplacianVar >= 2000.0));

    final effectiveUseNchw = isNativeNchw;
    final effectiveTargetSize = modelTargetSize;

    final effectiveExpansion =
        (effectiveTargetSize == 80) ? 2.7 : expansionFactor;
    final effectiveIsBgr = isBgr ?? (effectiveTargetSize == 80);
    final effectiveContrastStretch = enableContrastStretch ?? false;

    final tensorData = ImagePreprocessor.preprocessBufferToTensor(
      buffer,
      boundingBox: boundingBox,
      rotation: rotation,
      isRotatedBoundingBox: isRotatedBoundingBox,
      expansionFactor: effectiveExpansion,
      targetSize: effectiveTargetSize,
      useNchw: effectiveUseNchw,
      isBgr: effectiveIsBgr,
      normalizationScheme: normalizationScheme,
      enableContrastStretch: effectiveContrastStretch,
    );

    LivenessLogger.logTensorStats(tensorData, inputShape: modelInputShape);

    final logits = await _runInference(tensorData: tensorData);
    stopwatch.stop();

    const realIdx = 0;
    const spoofIdx = 1;

    final realLogit = logits[realIdx];
    final spoofLogit = logits[spoofIdx];

    final rawResult = LivenessResult.fromLogits(
      realLogit: realLogit,
      spoofLogit: spoofLogit,
      threshold: threshold,
      inferenceTime: stopwatch.elapsed,
      lbpUniformityScore: lbpRatio,
      hogGridDominance: hogDominance,
      faceAreaRatio: faceAreaRatio,
      chrominanceVariance: chrominanceVar,
      laplacianDelta: laplacianDelta,
      saturationVariance: saturationVar,
      moireHighFreqRatio: moireHighFreqRatio,
      meanLuminance: meanLuminance,
      isLowLight: isLowLight,
    );

    // Balanced EMA Calculation:
    final currentRealProb = 1.0 / (1.0 + math.exp(spoofLogit - realLogit));

    final double effectiveAlpha;
    if (_emaRealScore == null) {
      _emaRealScore = currentRealProb;
      effectiveAlpha = 1.0;
    } else {
      effectiveAlpha = emaAlpha;
      _emaRealScore = (currentRealProb * effectiveAlpha) +
          (_emaRealScore! * (1.0 - effectiveAlpha));
    }

    final safeEma = _emaRealScore!.clamp(1e-7, 1.0 - 1e-7);
    final spoofScoreEma = 1.0 - safeEma;
    final smoothedDiff = math.log(safeEma / (1.0 - safeEma));

    final isThresholdPassed = (threshold > 0.0 && threshold < 1.0)
        ? (safeEma >= threshold)
        : (smoothedDiff >= threshold);

    var calculatedStatus =
        isThresholdPassed ? LivenessStatus.real : LivenessStatus.spoof;

    // Low-Light Auto-Acceptance Safeguard:
    // When lowLightThreshold is provided and face luminance is below it,
    // auto-accept as real.
    if (isLowLight) {
      calculatedStatus = LivenessStatus.real;
    }

    // Multi-Factor Decision Fusion Engine:
    // Calibrated physical spoof indicators override neural real score only
    // when unequivocal attack signals exist.
    if (calculatedStatus == LivenessStatus.real && !isLowLight) {
      final isScreenSubPixelGrid = hogDominance != null &&
          hogDominance >= 0.380 &&
          lbpRatio != null &&
          lbpRatio < 0.280 &&
          chrominanceVar != null &&
          chrominanceVar >= 200.0;

      final isEmissiveDisplayLighting =
          chrominanceVar != null && chrominanceVar >= 220.0;

      final isAnySpoofSignal = isPrintSpoof ||
          (isEmissiveDisplayLighting &&
              (isScreenGridSpoof ||
                  isScreenReplaySpoof ||
                  isHighResScreenSpoof ||
                  isEmissiveScreenSpoof ||
                  is2DFlatSpoof ||
                  isEmissiveSaturationSpoof ||
                  isMoireSpoof ||
                  isScreenSubPixelGrid));

      // Neural certainty safeguard:
      // When the neural network rates the frame as real
      // (rawResult.logitDiff >= 0.0 or realLogit > spoofLogit),
      // prioritize genuine real users and do not allow soft micro-texture,
      // saturation variance, or chrominance variation to cause false
      // rejections. Only unequivocal physical attack signals (Moiré FFT
      // interference fringes) can override neural real scores.
      final isExtremeAttackSignal = isMoireSpoof;

      final isConfidentNeuralReal =
          rawResult.logitDiff >= 0.0 && !isExtremeAttackSignal;

      if (isAnySpoofSignal && !isConfidentNeuralReal) {
        if (isPrintSpoof &&
            !isHighResScreenSpoof &&
            !isEmissiveScreenSpoof &&
            !isScreenGridSpoof &&
            !isScreenReplaySpoof &&
            !isScreenSubPixelGrid) {
          calculatedStatus = LivenessStatus.printSpoof;
        } else {
          calculatedStatus = LivenessStatus.screenReplaySpoof;
        }
      }
    }

    final result = LivenessResult(
      isReal: calculatedStatus == LivenessStatus.real,
      status: calculatedStatus,
      realScore: safeEma,
      spoofScore: spoofScoreEma,
      realLogit: rawResult.realLogit,
      spoofLogit: rawResult.spoofLogit,
      logitDiff: smoothedDiff,
      confidence: smoothedDiff.abs(),
      threshold: threshold,
      inferenceTime: stopwatch.elapsed,
      rawRealScore: rawResult.realScore,
      rawSpoofScore: rawResult.rawSpoofScore,
      rawLogitDiff: rawResult.rawLogitDiff,
      rawIsReal: rawResult.rawIsReal,
      lbpUniformityScore: lbpRatio,
      hogGridDominance: hogDominance,
      faceAreaRatio: faceAreaRatio,
      chrominanceVariance: chrominanceVar,
      laplacianDelta: laplacianDelta,
      saturationVariance: saturationVar,
      moireHighFreqRatio: moireHighFreqRatio,
      meanLuminance: meanLuminance,
      isLowLight: isLowLight,
    );

    LivenessLogger.logInferenceResult(
      realLogit: realLogit,
      spoofLogit: spoofLogit,
      logitDiff: smoothedDiff,
      currentRealProb: currentRealProb,
      emaRealScore: safeEma,
      isReal: result.isReal,
      status: result.status,
      threshold: threshold,
      inferenceTime: stopwatch.elapsed,
      meanLuminance: meanLuminance,
      isLowLight: isLowLight,
    );

    return result;
  }

  /// Runs passive face liveness and anti-spoofing detection directly on raw
  /// image bytes ([Uint8List]).
  ///
  /// Evaluates static photo bytes (e.g. JPEG, PNG) using the full anti-spoofing
  /// heuristic suite (micro-texture, YCbCr chrominance, 2D Laplacian screen).
  ///
  /// ### Parameters:
  /// - [imageBytes]: The raw image file bytes (`Uint8List`).
  /// - [boundingBox]: Optional facial bounding box. If `null`, evaluates frame.
  /// - [threshold]: Logit decision threshold (default `0.0`).
  /// - [expansionFactor]: Square crop margin around face box (default `1.5x`).
  /// - [lowLightThreshold]: When specified, frames with mean luminance below
  ///   this threshold are auto-accepted as live.
  Future<LivenessResult> detectLivenessFromImageBytes(
    Uint8List imageBytes, {
    FaceBoundingBox? boundingBox,
    double threshold = 0.0,
    double expansionFactor = ImagePreprocessor.defaultExpansionFactor,
    double? lowLightThreshold,
    bool? isBgr,
    NormalizationScheme normalizationScheme = NormalizationScheme.zeroToOne,
    bool? enableContrastStretch,
  }) async {
    if (!isInitialized) {
      throw StateError(
        'PassiveLivenessDetector is not initialized. Call initialize() first.',
      );
    }

    final codec = await ui.instantiateImageCodec(imageBytes);
    final frameInfo = await codec.getNextFrame();
    final image = frameInfo.image;
    final byteData = await image.toByteData();

    if (byteData == null) {
      image.dispose();
      codec.dispose();
      throw ArgumentError('Failed to extract raw RGBA pixel bytes from image.');
    }

    final rgbaBytes = byteData.buffer.asUint8List();
    final width = image.width;
    final height = image.height;

    image.dispose();
    codec.dispose();

    final buffer = LivenessImageBuffer(
      width: width,
      height: height,
      format: LivenessImageFormat.rgba8888,
      planes: [
        LivenessImagePlane(
          bytes: rgbaBytes,
          bytesPerRow: width * 4,
          bytesPerPixel: 4,
        ),
      ],
    );

    return await detectLivenessFromBuffer(
      buffer,
      boundingBox: boundingBox,
      threshold: threshold,
      expansionFactor: expansionFactor,
      enableProximityGate:
          false, // Proximity gate is disabled for static image crops
      lowLightThreshold: lowLightThreshold,
      isBgr: isBgr,
      normalizationScheme: normalizationScheme,
      enableContrastStretch: enableContrastStretch,
    );
  }

  /// Runs passive face liveness and anti-spoofing detection directly on a
  /// static image [File].
  ///
  /// Evaluates static photo files using the full anti-spoofing heuristic suite.
  ///
  /// ### Parameters:
  /// - [file]: The static image file to analyze.
  /// - [boundingBox]: Optional facial bounding box. If `null`, evaluates frame.
  /// - [threshold]: Logit decision threshold (default `0.0`).
  /// - [expansionFactor]: Square crop margin around face box (default `1.5x`).
  /// - [lowLightThreshold]: When specified, frames with mean luminance below
  ///   this threshold are auto-accepted as live.
  Future<LivenessResult> detectLivenessFromImageFile(
    File file, {
    FaceBoundingBox? boundingBox,
    double threshold = 0.0,
    double expansionFactor = ImagePreprocessor.defaultExpansionFactor,
    double? lowLightThreshold,
    bool? isBgr,
    NormalizationScheme normalizationScheme = NormalizationScheme.zeroToOne,
    bool? enableContrastStretch,
  }) async {
    final bytes = await file.readAsBytes();
    return await detectLivenessFromImageBytes(
      bytes,
      boundingBox: boundingBox,
      threshold: threshold,
      expansionFactor: expansionFactor,
      lowLightThreshold: lowLightThreshold,
      isBgr: isBgr,
      normalizationScheme: normalizationScheme,
      enableContrastStretch: enableContrastStretch,
    );
  }

  /// Close native TFLite interpreter and clear EMA tracker.
  Future<void> dispose() async {
    resetEma();
    await _engine.close();
    _isInitialized = false;
  }
}
