## 0.1.2

- **Optimized Real User Pass Rate & Reduced False Screen Replay Spoof Rejections**:
  - **Neural Certainty Safeguard Calibration**: Re-calibrated multi-factor heuristic override rules to prevent soft micro-texture variations, screen grid highlights, or ambient chrominance shifts from causing false spoof rejections on genuine live faces.
  - **Protected Positive Neural Real Scores**: When the neural model classifies a face as live (`rawResult.logitDiff >= 0.0`), genuine real user selfie frames are preserved as `LivenessStatus.real`, reserving spoof status overrides strictly for unequivocal physical attack signals like Moiré FFT interference fringes (`isMoireSpoof`) or print halftone patterns (`isPrintSpoof`).
  - **Calibrated Sub-Pixel Grid & Emissive Thresholds**: Raised HOG grid dominance cutoff ($\ge 0.380$) and chrominance variance requirement ($\ge 200.0 / 220.0$) to eliminate false positives on high-end modern smartphones (e.g. OLED screen reflections under bright office/indoor lighting).
  - **Removed Borderline Screen Replay Overrides**: Eliminated aggressive borderline heuristics that previously triggered false `screenReplaySpoof` classifications on high-detail facial features.
- **Diagnostics & Code Quality**:
  - **Enhanced Diagnostic Logging**: Refactored `LivenessLogger` format outputs for cleaner crop stats, tensor data, and inference logging.
  - **Static Analysis & Lint Standardizations**: Fixed code lints and type annotations across all core heuristic analyzers (`ImagePreprocessor`, `HighResScreenAnalyzer`, `ColorSpaceAnalyzer`, `FftMoireAnalyzer`, `LbpHogAnalyzer`, `FaceProximityGate`).

## 0.1.1

- **Expanded Flutter & Dart Compatibility**: Widened environment constraints to support Dart SDK `sdk: ">=3.0.0 <4.0.0"` and Flutter `flutter: ">=3.10.0"`.

## 0.1.0

- **Low-Light Face Liveness Auto-Acceptance (`lowLightThreshold`)**: Added configurable low-light auto-acceptance to eliminate false spoof rejections and errors in dark/dimly-lit environments (such as night shifts or low-light work spaces).
  - **`lowLightThreshold` (`double?`, default `null`)**: When specified (e.g. `70.0` or `0.30`), frames captured with mean luminance below this threshold are automatically accepted as live (`isReal: true`, `status: LivenessStatus.real`), bypassing dark-induced neural logit suppression. Supports both 0..255 direct scale and normalized 0..1 scale.
  - **Exposed Luminance Metrics**: `LivenessResult` provides `meanLuminance` ($0.0 \dots 255.0$) and `isLowLight` for diagnostic logging and UI display.
- **Upgraded Multi-Modal Anti-Spoofing Heuristics & Decision Fusion**: Enhanced resilience against high-resolution OLED / Retina / MacBook screen replay attacks without increasing package size or native binary footprint.
  - **2D Flatness Check (`is2DFlatSpoof`)**: Evaluates Laplacian depth-of-field variance delta between center face crop ($\sigma^2_{\text{Face}}$) vs background region ($\sigma^2_{\text{Background}}$) to detect flat 2D focal planes ($\Delta < 0.08$).
  - **Emissive Saturation Spikes (`isEmissiveSaturationSpoof`)**: Converted color space crops to HSV to analyze Saturation ($S$) channel variance ($\text{varSat} \ge 0.045$) relative to Hue, identifying additive RGB display backlight scatter.
  - **Fast 2D Radix-2 FFT Moiré Analyzer (`FftMoireAnalyzer`)**: Pure Dart-only Cooley-Tukey FFT algorithm ($O(n \log n)$) to detect high-frequency sub-pixel display grid peaks ($\text{regularity} \ge 25.0$) beyond 60% Nyquist radius with zero binary size overhead.
  - **Calibrated Multi-Factor Decision Fusion Engine**: Re-calibrated decision override rules to reliably catch screen photo attacks (including low-brightness MacBook photos) while protecting genuine live face selfies under indoor lighting.
  - **Enhanced Debug & Watermark Metrics**: Exposed `laplacianDelta`, `saturationVariance`, and `moireHighFreqRatio` metrics in `LivenessResult` for detailed debugging and watermark logging.

## 0.0.6

- **Direct `CameraImage` Liveness API Support (`detectLivenessFromCameraImage`)**: Added direct `CameraImage` evaluation support to `PassiveLivenessDetector`, eliminating redundant developer boilerplate for converting camera frames to `LivenessImageBuffer`.

## 0.0.5

- **Native MethodChannel TFLite Architecture (~0MB Android APK Impact)**: Migrated model inference from `flutter_litert` Dart FFI to native platform channels (`com.andikatp.passiveLiveness`), reducing plugin binary size footprint from **~40MB** down to almost **0MB**.
  - **Android**: Uses Google Play Services TFLite runtime (`play-services-tflite-java:16.5.0` & `play-services-tflite-gpu:16.5.0`), eliminating **~40MB** APK/AAB size bloat by utilizing system-shared TFLite binaries with automatic GPU acceleration & CPU fallback.
  - **iOS**: Uses Apple-thinning optimized `TensorFlowLiteSwift (~> 2.14)` targeting iOS 12.0+.
- **Zero-Dependency Plugin Architecture**: Completely removed third-party `flutter_litert` dependency.
- **Pure Dart Heuristic Processing**: All anti-spoofing heuristic layers (`FaceProximityGate`, `LbpHogAnalyzer`, `ColorSpaceAnalyzer`, `HighResScreenAnalyzer`, and `ImagePreprocessor`) remain 100% in Dart for zero-overhead performance.
- **Lower-end Device Compatibility (Low Light & High Sensor Noise)**: Re-enabled automatic brightness adjustment (`enableContrastStretch`) in the main inference pipeline to improve neural model accuracy on budget phones with dark environments.
- **Chrominance Variance & Heuristic Decision Tuning**: Calibrated `ColorSpaceAnalyzer.maxVarianceThreshold` to `140.0` and upgraded multi-factor decision fusion to reliably capture subtle OLED screen replays (e.g. `spoof4`) while eliminating false spoof rejections on genuine faces.
- **Cleaner Diagnostics**: Streamlined `LivenessLogger` to reflect Native Platform Channel engine initialization.
- **Enhanced `.gitignore`**: Added full coverage for Android and iOS build output directories (`.gradle/`, `Pods/`, `DerivedData/`, etc.).

## 0.0.4

- **High-Resolution Screen Replay Anti-Spoofing (`HighResScreenAnalyzer`)**: Added 2D Laplacian frequency variance and patch focus depth dispersal ($\sigma^2_{\text{PatchLap}}$) analysis to catch high-resolution OLED/Retina/4K screen replay presentation attacks.
- **Glasses Glare & Frame Edge False Positive Resolution**: Added specular glare highlight masking ($\ge 245$ brightness) and multi-region upper-face HOG peak de-biasing (`LbpHogAnalyzer` & `ColorSpaceAnalyzer`) to prevent linear glasses frames and anti-reflective lens glare from triggering false spoof classifications on genuine users.
- **Multi-Factor Liveness Decision Fusion Engine**: Upgraded `PassiveLivenessDetector` decision engine to use multi-factor fusion scoring with adaptive neural model thresholding.
- **Zero-Config Default Heuristic Engines**: All anti-spoofing heuristic layers (`enableTextureAnalysis`, `enableColorSpaceAnalysis`, `enableHighResScreenAnalysis`) are now **enabled by default (`true`)**.
- **Streamlined API Parameter Signatures**: Cleaned up redundant internal low-level flags across `detectLivenessFromImageBytes`, `detectLivenessFromImageFile`, `detectLivenessFromBuffer`, and `LivenessFrameProcessor.processBufferFrame` for a zero-boilerplate developer experience.
- **Face Aspect Ratio & Proximity Gate**: Added early pre-inference rejection gate (`FaceProximityGate`) to discard presentation attacks with small cropped photos ($<5\%$ area), extreme close-ups ($>85\%$ area), or distorted aspect ratios ($0.50 \dots 1.25$).
- **Micro-Texture LBP / HOG Analysis Engine**: Added `LbpHogAnalyzer` to evaluate $256 \times 256$ un-downscaled face crops for paper inkjet print halftone patterns (Local Binary Patterns) and screen grid alignments (Histogram of Oriented Gradients).
- **YCbCr / YUV Color Space Analysis**: Added `ColorSpaceAnalyzer` to calculate chrominance sub-sampling variance ($\sigma^2_{CbCr}$) directly from YUV camera streams to detect emissive RGB digital display screen replay attacks (iPad/tablet video replays).
- **Adaptive Screen Flash Overlay**: Added `LivenessFlashController` and `AdaptiveScreenFlashOverlay` UI components for active photometric stereo assist.
- **New `LivenessStatus` Diagnostic Reason Codes**: Introduced `tooFar`, `tooClose`, `invalidAspectRatio`, `printSpoof`, and `screenReplaySpoof` to give developers detailed status feedback.

## 0.0.3

- **Android Sensor Coordinate Rotation & Accuracy Fixes**: Fixed coordinate transformation for Android ML Kit portrait face bounding boxes across all sensor angles (`0°`, `90°`, `180°`, `270°`), ensuring accurate upright crops and resolving false spoof detections on Android devices.
- **Low-Light & Dark Environments Compensation**: Enhanced anti-spoofing accuracy in dim lighting using adaptive power-law gamma expansion ($\gamma \approx 0.60 - 0.88$) and dynamic contrast stretching without corrupting facial texture liveness signals.
- **Glasses Glare & Specular Reflection Resilience**: Implemented Asymmetric Exponential Moving Average (EMA) score filtering ($\alpha=0.1$ for score drops, $\alpha=0.4$ for recovery) to prevent momentary specular lens reflections on glasses from triggering false spoof classifications.
- **LiteRT Next `CompiledModel` Zero-Copy Acceleration**: Integrated LiteRT Next `CompiledModel` support for zero-copy GPU/CPU hardware acceleration with automatic fallback to classic `Interpreter`.
- **Flexible Model Tensor Layout Support**: Added automatic shape inspection to support both NCHW (`[1, 3, H, W]`) and NHWC (`[1, H, W, 3]`) tensor models.
- **Edge Pixel Replication (`BORDER_REPLICATE`)**: Improved image preprocessor boundary handling to eliminate pitch-black border artifacts on tight face crops.
- **Pending Result & EMA Tracker Controls**: Added `LivenessResult.pending()` factory constructor and `resetEma()` controls for handling motion transition frames cleanly.

## 0.0.2

- **Fix Android ML Kit Bounding Box Coordinate System Mismatch**: Fixed false spoof classification on Android caused by portrait bounding box offsets. Added `isRotatedBoundingBox` parameter and `FaceBoundingBox.toRawBufferSpace()` transformation for `0°`, `90°`, `180°`, and `270°` sensor angles.
- **Fix Android 270° Front Camera Upside-Down Crop**: Corrected `case 270` coordinate mapping in `ImagePreprocessor` to ensure MiniFAS receives upright face crops.
- **Offload Inference to Background Isolate (`IsolateInterpreter`)**: Prevented main UI isolate frame drops by running LiteRT inference asynchronously on a dedicated background Dart isolate.
- **Add XNNPack ARM NEON SIMD Hardware Vectorization**: Enabled `XNNPackDelegate` for 2x–4x CPU matrix multiplication speedup on mobile chipsets.
- **Fast NV21 Plane Copy**: Optimized `_getNv21Bytes` to replace 518,400 Dart loop iterations with native `setRange` array copies.
- **Low-Light Adaptive Gamma Contrast Expansion**: Replaced linear boost with adaptive power-law gamma curve ($\gamma \approx 0.60 - 0.88$) to expand 3D skin texture gradients in dim lighting.

## 0.0.1

- Initial release of `passive_liveness`.
- Ultra-lightweight passive face anti-spoofing engine powered by LiteRT (TensorFlow Lite) edge inference.
- Real-time zero-copy camera frame processing (`LivenessImageBuffer`) for high FPS video streams (iOS `BGRA8888`, Android `NV21`/`YUV420`).
- Support for static image file (`File`) and byte array (`Uint8List`) liveness detection (e.g. from `takePicture()` or `image_picker`).
- Closed-form `reflect101` boundary padding and camera sensor rotation sampling (`0°`, `90°`, `180°`, `270°`).
- Built-in low-light shadow-lift compensation for underexposed camera environments.
