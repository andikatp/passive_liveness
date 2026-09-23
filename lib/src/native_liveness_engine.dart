import 'dart:typed_data';

import 'package:flutter/services.dart';

/// Target class index ordering produced by binary neural model classification
/// logits.
enum ModelClassOrder {
  /// Index 0: Real Face, Index 1: Spoof Face (default).
  realFirst,

  /// Index 0: Spoof Face, Index 1: Real Face (common in PyTorch ImageFolder
  /// sorted datasets).
  spoofFirst,
}

/// Lightweight platform channel engine that delegates TFLite neural inference
/// to native OS runtime.
///
/// On Android, uses Google Play Services TFLite module (0MB APK size increase).
/// On iOS, uses TensorFlowLiteSwift (~3-5MB download impact).
class NativeLivenessEngine {
  static const MethodChannel _channel = MethodChannel(
    'com.andikatp.passiveLiveness',
  );

  bool _isModelLoaded = false;

  /// Whether the native TFLite interpreter model has been loaded.
  bool get isModelLoaded => _isModelLoaded;

  /// Model input tensor shape returned from native runtime
  /// (e.g. [1, 3, 128, 128]).
  List<int>? modelInputShape;

  /// Whether the model natively expects NCHW format (`[1, 3, H, W]`).
  bool isNativeNchw = false;

  /// Target spatial resolution expected by model (e.g. 128).
  int modelTargetSize = 128;

  /// Class order mapping for 2-class binary model logits.
  ModelClassOrder classOrder = ModelClassOrder.realFirst;

  /// Initialize the native TFLite engine by passing raw model bytes.
  Future<void> loadModel(
    Uint8List modelBytes, {
    ModelClassOrder classOrder = ModelClassOrder.realFirst,
  }) async {
    this.classOrder = classOrder;

    final result = await _channel.invokeMapMethod<String, dynamic>(
      'initModel',
      {'modelBytes': modelBytes},
    );

    if (result != null) {
      final rawShape = result['inputShape'] as List?;
      if (rawShape != null) {
        modelInputShape = rawShape.map((e) => (e as num).toInt()).toList();
      }
      isNativeNchw = result['isNchw'] as bool? ?? false;
      modelTargetSize = result['targetSize'] as int? ?? 128;
    }

    _isModelLoaded = true;
  }

  /// Runs neural model inference on a preprocessed Float32 tensor buffer.
  ///
  /// Returns raw output classification logits `[realLogit, spoofLogit]`.
  Future<List<double>> runInference(Float32List tensorData) async {
    if (!_isModelLoaded) {
      throw StateError(
        'NativeLivenessEngine is not loaded. Call loadModel() first.',
      );
    }

    final result = await _channel.invokeMethod<List<dynamic>>('runInference', {
      'inputData': tensorData.buffer.asUint8List(),
    });

    if (result == null || result.length < 2) {
      return [0.0, 0.0];
    }

    final double realLogit;
    final double spoofLogit;

    if (result.length == 3) {
      // 3-class MiniFASNet specification:
      // index 0: 2D Spoof, index 1: 3D Spoof, index 2: Real Face
      realLogit = (result[2] as num).toDouble();
      final spoof2d = (result[0] as num).toDouble();
      final spoof3d = (result[1] as num).toDouble();
      spoofLogit = spoof2d > spoof3d ? spoof2d : spoof3d;
    } else if (classOrder == ModelClassOrder.spoofFirst) {
      // 2-class binary specification (index 0: Spoof, index 1: Real)
      spoofLogit = (result[0] as num).toDouble();
      realLogit = (result[1] as num).toDouble();
    } else {
      // 2-class binary specification (index 0: Real, index 1: Spoof)
      realLogit = (result[0] as num).toDouble();
      spoofLogit = (result[1] as num).toDouble();
    }

    return [realLogit, spoofLogit];
  }

  /// Close the native TFLite interpreter and free platform resources.
  Future<void> close() async {
    if (_isModelLoaded) {
      try {
        await _channel.invokeMethod('closeModel');
      } on Exception catch (_) {}
      _isModelLoaded = false;
    }
  }
}
