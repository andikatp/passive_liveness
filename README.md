# passive_liveness

[![pub package](https://img.shields.io/pub/v/passive_liveness.svg)](https://pub.dev/packages/passive_liveness)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-flutter%20%7C%20android%20%7C%20ios-blue.svg)](https://pub.dev/packages/passive_liveness)
[![style: very good analysis](https://img.shields.io/badge/style-very_good_analysis-B22C89.svg)](https://pub.dev/packages/very_good_analysis)

An ultra-lightweight, high-performance passive face anti-spoofing (liveness) detection package for Flutter powered by **Native TensorFlow Lite** edge inference on Android (Google Play Services) and iOS (`TensorFlowLiteSwift`).

---

## Features

- ⚡ **Native TFLite Inference & Zero-MB Android Footprint**: Uses Google Play Services TFLite on Android (0 MB APK size impact) with GPU acceleration and automatic CPU fallback, and `TensorFlowLiteSwift` on iOS.
- 🔓 **100% Standalone (No ML Kit Required)**: Works directly out-of-the-box on raw images or camera streams without needing Google ML Kit or third-party face detectors.
- 📷 **Zero-Copy Camera Stream Processing**: Preprocesses raw Flutter `CameraImage` byte buffers (`NV21` / `YUV420` on Android, `BGRA8888` on iOS) directly to TFLite tensors without main-thread image decoding.
- 👓 **Glasses Glare Resistance (Asymmetric EMA)**: Uses Asymmetric Exponential Moving Average filtering ($\alpha=0.1$ for score drops, $\alpha=0.4$ for recovery) to resist momentary specular reflections on glasses.
- 🏃 **Motion Gate & Stability Heuristic**: Tracks bounding box delta shifts to bypass inference on motion-blurred or out-of-focus frames during user movement.
- 🖼️ **Static Photo & File Detection**: Uses Flutter's built-in C++ Skia engine codecs (`dart:ui`) to evaluate liveness from static images (`File` or `Uint8List`) with **zero external image package dependencies**.
- 📱 **Android Rotated Bounding Box Mapping**: Built-in `isRotatedBoundingBox` auto-detection and `FaceBoundingBox.toRawBufferSpace()` transformation for portrait ML Kit face detection bounding boxes on Android (`0°`, `- 💡 **Adaptive Contrast Stretching & Edge Clamping**: Automatically normalizes dark backlit faces and uses Edge Pixel Replication (`BORDER_REPLICATE`) to eliminate black border artifacts.
- 🛡️ **Multi-Layer Anti-Spoofing Protection Engine**:
  - **Proximity & Aspect Ratio Gate**: Rejects presentation attacks with small cropped photos or extreme lens close-ups ($5\% \le \text{faceAreaRatio} \le 85\%$).
  - **Micro-Texture LBP / HOG Analyzer**: Evaluates un-downscaled $256 \times 256$ face crops to detect inkjet paper print noise (LBP) and screen sub-pixel grid lines (HOG).
  - **YCbCr Chrominance & HSV Saturation Analysis**: Inspects $Cb/Cr$ sub-pixel color space dispersion ($\sigma^2_{CbCr}$) and HSV saturation variance to separate digital LCD/OLED/Retina screen replays from natural skin reflectance.
  - **2D Laplacian Depth-of-Field Flatness Check**: Compares center face focus vs outer background region to identify flat 2D focal planes ($\Delta < 0.08$).
  - **Fast 2D Radix-2 FFT Moiré Analyzer**: Pure Dart Cooley-Tukey FFT algorithm ($O(n \log n)$) to detect high-frequency sub-pixel display grid peaks ($\text{regularity} \ge 25.0$) beyond 60% Nyquist radius with zero package size impact.
  - **Adaptive Screen Flash (Active Photometric Stereo)**: Brief high-brightness screen flash UI overlay (`LivenessFlashController` & `AdaptiveScreenFlashOverlay`) to verify 3D skin reflectance bounce.

---

## Multi-Layer Anti-Spoofing Protection Layers

### 1. Face Aspect Ratio & Proximity Gate
Prevents attackers from tricking the detector by holding up a tiny printed photo card or ID card close to the lens.
```dart
final result = await detector.detectLivenessFromBuffer(
  buffer,
  boundingBox: faceBbox,
  enableProximityGate: true, // Automatically filters tooFar, tooClose, invalidAspectRatio
);

if (result.status == LivenessStatus.tooFar) {
  print('Please move closer to the camera');
} else if (result.status == LivenessStatus.tooClose) {
  print('Please move slightly further away');
}
```

### 2. Micro-Texture LBP / HOG Analysis Engine
Evaluates high-frequency spatial gradients on un-downscaled $256 \times 256$ crops to capture halftone inkjet printer patterns (LBP) and screen grid lines (HOG):
```dart
final result = await detector.detectLivenessFromBuffer(
  buffer,
  boundingBox: faceBbox,
  enableTextureAnalysis: true,
);

if (result.status == LivenessStatus.printSpoof) {
  print('Paper printout attack detected!');
}
```

### 3. YCbCr / YUV Color Space Transformation
Analyzes chrominance sub-sampling variance ($\sigma^2_{CbCr}$) directly from YUV camera streams to detect emissive RGB digital display screen replays (iPad/tablet video replays):
```dart
final result = await detector.detectLivenessFromBuffer(
  buffer,
  enableColorSpaceAnalysis: true,
);

if (result.status == LivenessStatus.screenReplaySpoof) {
  print('Digital screen replay attack detected!');
}
```

### 4. Adaptive Screen Flash Overlay (Photometric Stereo Assist)
Triggers a brief full-screen flash overlay to bounce light on the user's face:
```dart
final flashController = LivenessFlashController();

// Wrap camera preview in UI:
AdaptiveScreenFlashOverlay(
  controller: flashController,
  child: CameraPreview(cameraController),
);

// Trigger flash on face detection:
await flashController.triggerFlash();
```

---

## Do I Need Google ML Kit or a Face Detector?

### Short Answer: **NO!** 

The `boundingBox` parameter is **100% OPTIONAL**. 

If you don't pass a `boundingBox`, `passive_liveness` automatically uses the **entire image frame** as the face region. You can run liveness detection in just **one line of code**!

### How Bounding Box Works:

- **Without Face Detector (Default - Super Easy):**
  Simply pass your camera buffer, photo file, or bytes. `passive_liveness` evaluates the entire image automatically:
  ```dart
  // Works out of the box without ML Kit!
  final result = await detector.detectLivenessFromImageBytes(imageBytes);
  ```

- **With Face Detector (Optional - Advanced Precision):**
  If your app already uses a face detector (like `google_mlkit_face_detection`), passing `FaceBoundingBox.fromRect(face.boundingBox)` crops directly onto the detected face for pinpoint accuracy:
  ```dart
  // Optional: pass boundingBox if you have ML Kit
  final result = await detector.detectLivenessFromImageBytes(
    imageBytes,
    boundingBox: FaceBoundingBox.fromRect(faceRect),
  );
  ```

---

## Installation

Add `passive_liveness` to your project:

```bash
flutter pub add passive_liveness
```

---

## Usage

### 1. Initialize Detector Engine

Initialize `PassiveLivenessDetector`. The bundled MiniFAS model is loaded automatically:

```dart
import 'package:passive_liveness/passive_liveness.dart';

final detector = PassiveLivenessDetector();
await detector.initialize();
```

---

### 2. Real-Time Camera Stream Detection (`CameraImage`)

#### Option A: Simple (Without ML Kit - Full Frame)
```dart
import 'package:camera/camera.dart';
import 'package:passive_liveness/passive_liveness.dart';

void processCameraFrame(CameraImage cameraImage, int sensorRotation) async {
  // Pass CameraImage directly — no manual buffer mapping required!
  final LivenessResult result = await detector.detectLivenessFromCameraImage(
    cameraImage,
    rotation: sensorRotation,
  );

  if (result.isReal) {
    print('Real human face!');
  } else {
    print('Spoof face detected: ${result.status.name}');
  }
}
```

#### Option B: Advanced (With ML Kit Face Detector)
```dart
  // Pass faceRect from ML Kit directly with CameraImage
  final boundingBox = faceRect != null ? FaceBoundingBox.fromRect(faceRect) : null;

  final LivenessResult result = await detector.detectLivenessFromCameraImage(
    cameraImage,
    boundingBox: boundingBox,
    rotation: sensorRotation,
    isRotatedBoundingBox: Platform.isAndroid,
  );
```

---

### 3. Using `LivenessFrameProcessor` for Smooth Streaming

For stream processing with automatic motion gating and frame throttling:

```dart
final processor = LivenessFrameProcessor(
  detector: detector,
  throttleInterval: const Duration(milliseconds: 150),
);

// In your camera stream listener — pass CameraImage directly:
final LivenessResult? result = await processor.processCameraFrame(
  cameraImage,
  rotation: sensorRotation,
);
```

---

### 4. Detect Liveness from Static Photo File (`File`)

Evaluate a photo file picked via `image_picker` or taken with `takePicture()`:

```dart
import 'dart:io';
import 'package:passive_liveness/passive_liveness.dart';

Future<void> checkPhotoLiveness(File imageFile) async {
  // Works directly on the file - boundingBox is optional!
  final LivenessResult result = await detector.detectLivenessFromImageFile(
    imageFile,
  );

  print('Is Real: ${result.isReal}');
  print('Real Score: ${result.realScore}');
  print('Inference Time: ${result.inferenceTime.inMilliseconds}ms');
}
```

---

### 5. Detect Liveness from Image Bytes (`Uint8List`)

Evaluate liveness directly from in-memory image bytes:

```dart
import 'dart:typed_data';
import 'package:passive_liveness/passive_liveness.dart';

Future<void> checkBytesLiveness(Uint8List imageBytes) async {
  // Works directly on raw bytes - boundingBox is optional!
  final LivenessResult result = await detector.detectLivenessFromImageBytes(
    imageBytes,
  );

  print('Is Real: ${result.isReal}');
  print('Real Score: ${result.realScore}');
}
```

---

## Clean Up

When you are done using the detector (e.g. in a State's `dispose` method):

```dart
@override
void dispose() {
  detector.dispose();
  super.dispose();
}
```

---

## Core Classes Reference

| Class | Description |
|---|---|
| `PassiveLivenessDetector` | Main engine class for initializing the Native TFLite model and running inferences. |
| `LivenessFrameProcessor` | Stream processor with motion-gating heuristic and frame throttling. |
| `FaceProximityGate` | Evaluates face bounding box area coverage and aspect ratio to prevent close-up print attacks. |
| `LbpHogAnalyzer` | Micro-texture analysis engine for Local Binary Patterns (LBP) and Histogram of Oriented Gradients (HOG). |
| `ColorSpaceAnalyzer` | Calculates YCbCr chrominance variance ($\sigma^2_{CbCr}$) to spot digital screen replay attacks. |
| `LivenessFlashController` | UI controller for triggering momentary high-brightness photometric stereo screen flashes. |
| `AdaptiveScreenFlashOverlay` | Flutter UI overlay widget rendering full-screen flash bursts during camera stream capture. |
| `LivenessImageBuffer` | Lightweight container for camera raw byte planes (`NV21`, `YUV420`, `BGRA8888`). |
| `FaceBoundingBox` | Coordinates (`x`, `y`, `width`, `height`) defining the face area. **Optional**. |
| `LivenessResult` | Detailed result containing `isReal`, `status`, `realScore`, `meanLuminance`, `isLowLight`, `lbpUniformityScore`, `hogGridDominance`, `chrominanceVariance`, and `inferenceTime`. |

---

### 🌙 Low-Light Face Liveness Auto-Acceptance (v0.1.0)
- **`lowLightThreshold`**: When specified (e.g. `70.0` or `0.30` or `PassiveLivenessDetector.defaultLowLightThreshold`), faces captured under dark/dimly-lit conditions (`meanLuminance < lowLightThreshold`) are automatically accepted as live (`isReal: true`, `status: LivenessStatus.real`), eliminating false spoof rejections on night shifts or in low-light workspaces. Supports both 0..255 direct scale and normalized 0..1 scale.
```dart
final result = await detector.detectLivenessFromCameraImage(
  cameraImage,
  lowLightThreshold: 70.0, // When passed, auto-accepts frames with luminance < 70.0
);

print('Mean luminance: ${result.meanLuminance}');
print('Low light detected: ${result.isLowLight}');
```

---

### 🛡️ Upgraded Multi-Modal Anti-Spoofing & Decision Fusion (v0.1.0)
> *Note: Passive anti-spoofing is a continuous cat-and-mouse game. While no single passive heuristic is 100% infallible against every possible high-end display in every extreme lighting condition, spoofing is significantly harder now!*

- **2D Laplacian Depth-of-Field Flatness Check**: Evaluates Laplacian variance delta between center face crop ($\sigma^2_{\text{Face}}$) vs outer background ($\sigma^2_{\text{Background}}$) to flag flat 2D focal planes ($\Delta < 0.08$).
- **Emissive Saturation Spikes (HSV Analysis)**: Analyzes Saturation ($S$) channel variance ($\text{varSat} \ge 0.045$) relative to Hue ($H$) to detect additive RGB screen backlight scatter.
- **Fast Radix-2 FFT Moiré Analyzer**: Pure Dart Cooley-Tukey FFT algorithm ($O(n \log n)$) measuring high-frequency sub-pixel display grid peaks ($\text{regularity} \ge 25.0$) beyond 60% Nyquist radius. Zero binary bloat!
- **Calibrated Decision Fusion Engine**: Multi-factor decision override rules reliably catch high-res OLED / Retina / MacBook screen replays (including low-brightness screen photos) while protecting genuine live face selfies under indoor lighting.

### ⚡ Native MethodChannel TFLite Architecture (v0.0.5)
- **~0 MB Android APK Size Impact**: Migrated model inference from `flutter_litert` Dart FFI to native platform channels (`com.andikatp.passiveLiveness`), decreasing plugin binary footprint from **~40 MB** down to almost **0 MB**.
- **Google Play Services TFLite (Android)**: Dynamically uses system-shared TFLite runtime (`play-services-tflite-java:16.5.0` & `play-services-tflite-gpu:16.5.0`), eliminating ~40 MB APK/AAB size bloat with automatic GPU acceleration & CPU fallback.
- **iOS `TensorFlowLiteSwift` Integration**: Native Swift platform channel wrapper (`~> 2.14`) targeting iOS 12.0+.
- **Zero Third-Party Plugin Dependencies**: Removed `flutter_litert` dependency completely for zero-conflict integration.
- **100% Pure Dart Heuristics**: All anti-spoofing heuristic analysis layers remain 100% in Dart for maximum execution speed and zero main-thread overhead.

### 🚀 Multi-Layer Anti-Spoofing & Accuracy Upgrades (v0.0.4)
- **High-Resolution Screen Replay Anti-Spoofing:** Added 2D Laplacian frequency variance and patch focus depth dispersal ($\sigma^2_{\text{PatchLap}}$) to catch high-density OLED/Retina screen replays.
- **Glasses Glare & Frame Edge False Positive Fix:** Specular glare highlight masking ($\ge 245$ brightness) and multi-region HOG peak de-biasing (`LbpHogAnalyzer` & `ColorSpaceAnalyzer`) to prevent linear glasses frames and anti-reflective lens glare from causing false positive spoof classifications.
- **Multi-Factor Liveness Decision Fusion Engine:** Upgraded decision engine with adaptive neural model thresholding and multi-heuristic validation.
- **Zero-Config Default Heuristic Engines:** All anti-spoofing heuristic layers are enabled by default (`true`).
- **Proximity & Aspect Ratio Gate:** Early rejection gate (`FaceProximityGate`) to discard small photo cards ($<5\%$) or extreme close-ups ($>85\%$).
- **Micro-Texture LBP / HOG Engine:** Extracted $256 \times 256$ un-downscaled crops to detect paper inkjet patterns (`printSpoof`) and screen grid alignments (`screenReplaySpoof`).
- **YCbCr Chrominance Variance Analysis:** Analyzes YUV/YCbCr color space dispersion ($\sigma^2_{CbCr}$) to catch emissive RGB screen replay attacks.
- **Adaptive Screen Flash Overlay:** Added `LivenessFlashController` and `AdaptiveScreenFlashOverlay` for active photometric stereo assist.
- **Native Hardware Acceleration:** Zero-copy hardware acceleration with GPU / NPU / CPU fallback.
- **Asymmetric EMA Glare Resistance:** Asymmetric Exponential Moving Average ($\alpha=0.1$ for drops, $\alpha=0.4$ for recovery) to prevent glasses reflections from triggering false spoof drops.
- **Bounding Box Motion Stability Gate:** Bypasses TFLite inference during motion blur.
- **Edge Pixel Replication (`BORDER_REPLICATE`):** Smooth mirror coordinate clamping (`_reflect101`) to eliminate black border artifacts.

---

## Model Attribution & Credits

This package utilizes the **MiniFASNet v2 SE** passive face anti-spoofing model architecture, derived and converted from [facenox/face-antispoof-onnx](https://github.com/facenox/face-antispoof-onnx) (MiniFASNetV2 SE trained with Fourier Transform frequency spectrum loss).

---

## Author

**Andika Tri Prasetya**
- GitHub: [github.com/andikatp](https://github.com/andikatp)

---

## License

This project is licensed under the OSI-approved **MIT License** - see the [LICENSE](LICENSE) file for details.
