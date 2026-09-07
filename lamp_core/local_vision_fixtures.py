"""Manually annotated local-detector outputs for the supplied desk images.

These values are an offline Pi-detector simulation fixture.  They deliberately
live apart from production code so a future OpenNI + TFLite detector can replace
them without changing the autonomous centering pipeline.
"""

from lamp_core.object_localization import SimulatedLocalTargetDetector, TargetDetection


DESK_TARGET_DETECTIONS = {
    "book_x015.jpeg": TargetDetection("book_x015", "Crucial BX500 package", (0.27, 0.10, 0.71, 0.29), 0.93),
    "book_x020.jpeg": TargetDetection("book_x020", "Crucial BX500 package", (0.30, 0.21, 0.73, 0.38), 0.94),
    "book_x035.jpeg": TargetDetection("book_x035", "Crucial BX500 package", (0.32, 0.34, 0.74, 0.50), 0.94),
    "book_x050.jpeg": TargetDetection("book_x050", "Crucial BX500 package", (0.28, 0.54, 0.69, 0.70), 0.95),
    "book_x070.jpeg": TargetDetection("book_x070", "Crucial BX500 package", (0.27, 0.66, 0.71, 0.85), 0.94),
    "book_x090.jpeg": TargetDetection("book_x090", "Crucial BX500 package", (0.05, 0.83, 0.56, 0.99), 0.91),
    "book_x0901.jpeg": TargetDetection("book_x0901", "Crucial BX500 package", (0.44, 0.70, 0.94, 0.95), 0.92),
}


SIMULATED_PI_LOCAL_DETECTOR = SimulatedLocalTargetDetector(DESK_TARGET_DETECTIONS)
