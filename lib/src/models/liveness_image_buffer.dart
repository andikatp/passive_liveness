// Duck-typing is used for cameraImage parameter to support flexible
// camera objects.
// ignore_for_file: avoid_dynamic_calls

import 'dart:typed_data';

/// Image format group supported for passive liveness detection.
enum LivenessImageFormat {
  /// YUV 4:2:0 format (typical for Android camera streams).
  yuv420,

  /// NV21 interleaved format.
  nv21,

  /// BGRA 8888 32-bit format (typical for iOS camera streams).
  bgra8888,

  /// RGBA 8888 32-bit format (typical for decoded static Flutter images).
  rgba8888,
}

/// Represents a single plane in a raw camera frame buffer.
class LivenessImagePlane {
  /// Creates a [LivenessImagePlane] with [bytes], [bytesPerRow],
  /// and optional [bytesPerPixel].
  const LivenessImagePlane({
    required this.bytes,
    required this.bytesPerRow,
    this.bytesPerPixel,
  });

  /// Byte array for this plane.
  final Uint8List bytes;

  /// Number of bytes per row (row stride).
  final int bytesPerRow;

  /// Number of bytes per pixel (pixel stride), if applicable.
  final int? bytesPerPixel;
}

/// Pure Dart lightweight model representing a raw camera image buffer.
///
/// Designed to decouple `passive_liveness` from
/// any specific Flutter camera package.
class LivenessImageBuffer {
  /// Creates a [LivenessImageBuffer] with
  /// [width], [height], [format], and [planes].
  const LivenessImageBuffer({
    required this.width,
    required this.height,
    required this.format,
    required this.planes,
  });

  /// Creates a [LivenessImageBuffer] directly from a Flutter
  /// `CameraImage` instance.
  factory LivenessImageBuffer.fromCameraImage(dynamic cameraImage) {
    if (cameraImage is LivenessImageBuffer) return cameraImage;

    final formatGroup = cameraImage.format.group.toString();
    final LivenessImageFormat format;
    if (formatGroup.contains('bgra8888')) {
      format = LivenessImageFormat.bgra8888;
    } else if ((cameraImage.planes as List).length == 1) {
      format = LivenessImageFormat.nv21;
    } else {
      format = LivenessImageFormat.yuv420;
    }

    final planes = (cameraImage.planes as List)
        .map(
          (p) => LivenessImagePlane(
            bytes: p.bytes as Uint8List,
            bytesPerRow: p.bytesPerRow as int,
            bytesPerPixel: p.bytesPerPixel as int?,
          ),
        )
        .toList();

    return LivenessImageBuffer(
      width: cameraImage.width as int,
      height: cameraImage.height as int,
      format: format,
      planes: planes,
    );
  }

  /// Buffer width in pixels.
  final int width;

  /// Buffer height in pixels.
  final int height;

  /// Buffer color format (`yuv420`, `nv21`, `bgra8888`, `rgba8888`).
  final LivenessImageFormat format;

  /// List of image planes.
  final List<LivenessImagePlane> planes;
}
