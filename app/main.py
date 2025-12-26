import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------- Load segmentation model ----------
seg_base_options = python.BaseOptions(
    model_asset_path="selfie_segmenter.tflite"
)
seg_options = vision.ImageSegmenterOptions(
    base_options=seg_base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_confidence_masks=True
)
segmenter = vision.ImageSegmenter.create_from_options(seg_options)

# ---------- Load face landmarker model ----------
face_base_options = python.BaseOptions(
    model_asset_path='face_landmarker.task'
)
face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=10
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

# ---------- Video capture ----------
cap = cv2.VideoCapture(0)
prev_time = time.time()

try:
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)

        # ---------- Human Segmentation ----------
        seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
        mask = seg_result.confidence_masks[0].numpy_view()  # (H, W)

        # Post-processing segmentation
        binary_mask = (mask > 0.5).astype(np.float32)
        binary_mask = np.squeeze(binary_mask)
        binary_mask = binary_mask[:, :, None]   # (H, W, 1)

        # Color overlay for segmentation
        alpha = 0.4  # transparency strength
        seg_color = np.array([0, 100, 255], dtype=np.uint8)  # Orange-red
        color_layer = (frame_bgr * (1 - alpha) + seg_color * alpha).astype(np.uint8)
        segmented_frame = (frame_bgr * (1 - binary_mask) + color_layer * binary_mask).astype(np.uint8)

        # ---------- Face Detection ----------
        face_result = face_detector.detect_for_video(mp_image, timestamp_ms)

        face_crop_resized = None
        if face_result.face_landmarks:
            height, width, _ = frame_bgr.shape

            for face_landmarks in face_result.face_landmarks:
                # Collect landmark points
                points = []
                for lm in face_landmarks:
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    points.append((x, y))

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                left, right = min(xs), max(xs)
                top, bottom = min(ys), max(ys)

                # Optional padding
                pad = 20
                left   = max(0, left - pad)
                top    = max(0, top - pad)
                right  = min(width, right + pad)
                bottom = min(height, bottom + pad)

                # Cropping the face
                face_crop = frame_bgr[top:bottom, left:right]
                if face_crop.size > 0:  # Check if crop is valid
                    face_crop_resized = cv2.resize(face_crop, (112, 112))

                # Draw bounding box on segmented frame
                cv2.rectangle(
                    segmented_frame,
                    (left, top),
                    (right, bottom),
                    (0, 255, 0),
                    2
                )

        # ---------- FPS Calculation ----------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(
            segmented_frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # ---------- Display ----------
        cv2.imshow("Human Segmentation + Face Detection", segmented_frame)

        if face_crop_resized is not None:
            cv2.imshow("Cropped Face 112x112", face_crop_resized)

        # Exit on 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    segmenter.close()
    face_detector.close()
