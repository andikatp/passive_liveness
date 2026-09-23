import 'dart:math' as math;
import 'dart:typed_data';

/// Result of evaluating moiré pattern frequency analysis via FFT.
class MoireAnalysisResult {
  const MoireAnalysisResult({
    required this.highFrequencyRatio,
    required this.peakHighFrequencyMagnitude,
    required this.structuralRegularity,
    required this.isMoireSpoof,
  });

  /// Ratio of high-frequency energy to total spectral energy ($0.0 \dots 1.0$).
  ///
  /// Digital screen re-photography produces moiré interference patterns
  /// that concentrate energy in the high-frequency spectral bands.
  final double highFrequencyRatio;

  /// Peak magnitude value in the high-frequency region of the spectrum.
  final double peakHighFrequencyMagnitude;

  /// Structural regularity score (peak-to-mean ratio in high-freq band).
  ///
  /// Screen sub-pixel grids produce sharp spectral peaks (high regularity),
  /// while organic skin texture has diffuse frequency distribution
  /// (low regularity).
  final double structuralRegularity;

  /// Whether moiré interference patterns characteristic of digital screens
  /// were detected.
  final bool isMoireSpoof;
}

/// Lightweight Dart-only moiré pattern analyzer using radix-2 Fast
/// Fourier Transform.
///
/// Detects structural high-frequency peaks in grayscale crops that are
/// characteristic
/// of digital screen sub-pixel grids photographed by digital cameras
/// (moiré interference).
/// Zero package size impact — pure Dart computation on 128×128 or
/// 256×256 crops.
class FftMoireAnalyzer {
  const FftMoireAnalyzer({
    this.highFreqRatioThreshold = 0.45,
    this.structuralRegularityThreshold = 25.0,
  });

  /// Minimum high-frequency energy ratio to flag moiré (default: 0.45).
  final double highFreqRatioThreshold;

  /// Minimum structural regularity (peak/mean ratio) to flag moiré (default: 25.0).
  final double structuralRegularityThreshold;

  /// Analyzes a grayscale crop ([width] × [height] Uint8List) for
  /// moiré patterns.
  ///
  /// Input dimensions should ideally be power-of-2 (128, 256).
  /// Non-power-of-2 inputs are zero-padded to the next power of 2.
  MoireAnalysisResult analyzeGrayscaleCrop(
    Uint8List grayBytes,
    int width,
    int height,
  ) {
    if (width < 16 || height < 16 || grayBytes.length < width * height) {
      return const MoireAnalysisResult(
        highFrequencyRatio: 0,
        peakHighFrequencyMagnitude: 0,
        structuralRegularity: 0,
        isMoireSpoof: false,
      );
    }

    // Use the smaller dimension and pad to next power of 2
    final size = _nextPowerOf2(math.min(width, height));
    // Cap at 256 to limit compute time
    final fftSize = math.min(size, 256);

    // Compute row-wise FFT, then column-wise FFT for 2D transform
    // Only compute magnitude spectrum (no inverse needed)

    // Step 1: Extract center square crop and convert to Float64
    final startX = (width - fftSize) ~/ 2;
    final startY = (height - fftSize) ~/ 2;

    final real2D = Float64List(fftSize * fftSize);
    final imag2D = Float64List(fftSize * fftSize);

    for (var y = 0; y < fftSize; y++) {
      final srcY = startY + y;
      for (var x = 0; x < fftSize; x++) {
        final srcX = startX + x;
        final srcIdx = srcY * width + srcX;
        final val =
            srcIdx < grayBytes.length ? grayBytes[srcIdx].toDouble() : 0.0;
        real2D[y * fftSize + x] = val;
      }
    }

    // Step 2: Row-wise FFT
    final rowReal = Float64List(fftSize);
    final rowImag = Float64List(fftSize);

    for (var y = 0; y < fftSize; y++) {
      final offset = y * fftSize;
      for (var x = 0; x < fftSize; x++) {
        rowReal[x] = real2D[offset + x];
        rowImag[x] = 0.0;
      }
      _fft(rowReal, rowImag);
      for (var x = 0; x < fftSize; x++) {
        real2D[offset + x] = rowReal[x];
        imag2D[offset + x] = rowImag[x];
      }
    }

    // Step 3: Column-wise FFT
    final colReal = Float64List(fftSize);
    final colImag = Float64List(fftSize);

    for (var x = 0; x < fftSize; x++) {
      for (var y = 0; y < fftSize; y++) {
        colReal[y] = real2D[y * fftSize + x];
        colImag[y] = imag2D[y * fftSize + x];
      }
      _fft(colReal, colImag);
      for (var y = 0; y < fftSize; y++) {
        real2D[y * fftSize + x] = colReal[y];
        imag2D[y * fftSize + x] = colImag[y];
      }
    }

    // Step 4: Compute magnitude spectrum and analyze frequency bands
    final half = fftSize ~/ 2;

    var totalEnergy = 0.0;
    var highFreqEnergy = 0.0;
    var peakHighFreq = 0.0;
    var highFreqSum = 0.0;
    var highFreqCount = 0.0;

    // The high-frequency region is defined as frequencies beyond 60% of
    //Nyquist (sub-pixel grid range)
    final highFreqRadiusMin = half * 0.60;
    // Exclude the DC component and very low frequencies
    const dcExcludeRadius = 3.0;

    for (var y = 0; y < fftSize; y++) {
      for (var x = 0; x < fftSize; x++) {
        final idx = y * fftSize + x;
        final mag = math.sqrt(
          real2D[idx] * real2D[idx] + imag2D[idx] * imag2D[idx],
        );

        // Convert to centered frequency coordinates
        final fx = (x < half) ? x : x - fftSize;
        final fy = (y < half) ? y : y - fftSize;
        final radius = math.sqrt(fx * fx + fy * fy);

        // Skip DC and very low frequencies
        if (radius < dcExcludeRadius) continue;

        totalEnergy += mag;

        if (radius >= highFreqRadiusMin) {
          highFreqEnergy += mag;
          highFreqSum += mag;
          highFreqCount++;
          if (mag > peakHighFreq) {
            peakHighFreq = mag;
          }
        }
      }
    }

    final highFreqRatio =
        totalEnergy > 0.0 ? highFreqEnergy / totalEnergy : 0.0;
    final meanHighFreq = highFreqCount > 0 ? highFreqSum / highFreqCount : 0.0;
    final structuralRegularity =
        meanHighFreq > 0.0 ? peakHighFreq / meanHighFreq : 0.0;

    // Moiré patterns produce both elevated high-frequency ratio AND sharp
    //structural peaks.
    // Both conditions must be met to avoid false positives from naturally
    //high-texture images.
    final isMoireSpoof = highFreqRatio >= highFreqRatioThreshold &&
        structuralRegularity >= structuralRegularityThreshold;

    return MoireAnalysisResult(
      highFrequencyRatio: highFreqRatio,
      peakHighFrequencyMagnitude: peakHighFreq,
      structuralRegularity: structuralRegularity,
      isMoireSpoof: isMoireSpoof,
    );
  }

  /// Cooley-Tukey in-place radix-2 FFT on complex arrays.
  ///
  /// Both [real] and [imag] must have length that is a power of 2.
  static void _fft(Float64List real, Float64List imag) {
    final n = real.length;
    if (n <= 1) return;

    // Bit-reversal permutation
    var j = 0;
    for (var i = 0; i < n - 1; i++) {
      if (i < j) {
        final tempR = real[i];
        final tempI = imag[i];
        real[i] = real[j];
        imag[i] = imag[j];
        real[j] = tempR;
        imag[j] = tempI;
      }
      var k = n >> 1;
      while (k <= j) {
        j -= k;
        k >>= 1;
      }
      j += k;
    }

    // Cooley-Tukey butterfly stages
    var step = 2;
    while (step <= n) {
      final halfStep = step >> 1;
      final angle = -2.0 * math.pi / step;

      for (var groupStart = 0; groupStart < n; groupStart += step) {
        var wReal = 1.0;
        var wImag = 0.0;
        final cosA = math.cos(angle);
        final sinA = math.sin(angle);

        for (var k = 0; k < halfStep; k++) {
          final evenIdx = groupStart + k;
          final oddIdx = groupStart + k + halfStep;

          final tReal = wReal * real[oddIdx] - wImag * imag[oddIdx];
          final tImag = wReal * imag[oddIdx] + wImag * real[oddIdx];

          real[oddIdx] = real[evenIdx] - tReal;
          imag[oddIdx] = imag[evenIdx] - tImag;
          real[evenIdx] = real[evenIdx] + tReal;
          imag[evenIdx] = imag[evenIdx] + tImag;

          final newWReal = wReal * cosA - wImag * sinA;
          final newWImag = wReal * sinA + wImag * cosA;
          wReal = newWReal;
          wImag = newWImag;
        }
      }
      step <<= 1;
    }
  }

  /// Returns the next power of 2 >= [n].
  static int _nextPowerOf2(int n) {
    if (n <= 1) return 1;
    var p = 1;
    while (p < n) {
      p <<= 1;
    }
    return p;
  }
}
