import 'dart:ui';

/// Represents a bounding box for a detected face in an image.
class FaceBoundingBox {
  /// Creates a [FaceBoundingBox] with top-left coordinates ([x], [y])
  /// and dimensions ([width], [height]).
  const FaceBoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });

  /// Factory constructor from Flutter [Rect].
  factory FaceBoundingBox.fromRect(Rect rect) {
    return FaceBoundingBox(
      x: rect.left,
      y: rect.top,
      width: rect.width,
      height: rect.height,
    );
  }

  /// Factory constructor from Left, Top, Right, Bottom coordinates.
  factory FaceBoundingBox.fromLTRB(
    double left,
    double top,
    double right,
    double bottom,
  ) {
    return FaceBoundingBox(
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
    );
  }

  /// X-coordinate of top-left corner of the face bounding box.
  final double x;

  /// Y-coordinate of top-left corner of the face bounding box.
  final double y;

  /// Width of the face bounding box.
  final double width;

  /// Height of the face bounding box.
  final double height;

  /// Convert to Flutter [Rect].
  Rect toRect() => Rect.fromLTWH(x, y, width, height);

  /// Center X coordinate.
  double get centerX => x + (width / 2);

  /// Center Y coordinate.
  double get centerY => y + (height / 2);

  /// Right boundary X coordinate.
  double get right => x + width;

  /// Bottom boundary Y coordinate.
  double get bottom => y + height;

  /// Converts a bounding box defined in rotated frame space
  /// `[0..rotW, 0..rotH]`
  /// back into raw unrotated buffer space `[0..rawW, 0..rawH]`.
  FaceBoundingBox toRawBufferSpace(int rawW, int rawH, int rotation) {
    final normRotation = ((rotation % 360) + 360) % 360;
    if (normRotation == 0) return this;

    final rotCx = centerX;
    final rotCy = centerY;

    double rawCx;
    double rawCy;
    double rawWBox;
    double rawHBox;

    switch (normRotation) {
      case 90:
        rawCx = rotCy;
        rawCy = rawH.toDouble() - rotCx;
        rawWBox = height;
        rawHBox = width;
      case 180:
        rawCx = rawW.toDouble() - rotCx;
        rawCy = rawH.toDouble() - rotCy;
        rawWBox = width;
        rawHBox = height;
      case 270:
        rawCx = rawW.toDouble() - rotCy;
        rawCy = rotCx;
        rawWBox = height;
        rawHBox = width;
      default:
        rawCx = rotCx;
        rawCy = rotCy;
        rawWBox = width;
        rawHBox = height;
    }

    return FaceBoundingBox(
      x: rawCx - rawWBox / 2.0,
      y: rawCy - rawHBox / 2.0,
      width: rawWBox,
      height: rawHBox,
    );
  }

  /// Convert bounding box to a JSON map representation.
  Map<String, dynamic> toJson() => {
        'x': x,
        'y': y,
        'width': width,
        'height': height,
      };

  @override
  String toString() =>
      'FaceBoundingBox(x: $x, y: $y, width: $width, height: $height)';
}
