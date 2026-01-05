from ultralytics import YOLO
import cv2
import numpy as np


class ObjectDetectorSegmenter:
    """
    Wrapper class for YOLOv8-Seg model to perform object detection and instance segmentation.

    This class simplifies inference, filtering by class, and mask handling.
    """

    def __init__(self, model_path: str = 'yolov8m-seg.pt', conf_threshold: float = 0.25):
        """
        Initialize the YOLOv8-Seg model.

        Parameters
        ----------
        model_path : str, optional
            Path to the YOLOv8 segmentation model weights file.
            Common options: 'yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt', etc.
            (default: 'yolov8m-seg.pt')
        conf_threshold : float, optional
            Confidence threshold for keeping detections (default: 0.25)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names  # dict: idx → class name

    def detect_and_segment(self, image: np.ndarray, target_classes: list[str] | None = None) -> list[dict]:
        """
        Run detection and segmentation on the input image.

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format (shape: H×W×3)
        target_classes : list of str, optional
            If provided, only return detections for these class names.
            If None, return all detected classes.

        Returns
        -------
        list of dict
            Each dict contains:
            - 'class'       : str              → class name
            - 'confidence'  : float            → detection confidence
            - 'bbox'        : list[float]      → [x1, y1, x2, y2]
            - 'mask'        : np.ndarray       → binary mask (H×W, uint8), values 0 or 255
        """
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            verbose=False
        )[0]

        detections = []

        # Early exit if no detections or no masks
        if results.masks is None or len(results.masks) == 0:
            return detections

        # Extract data to CPU/numpy
        masks = results.masks.data.cpu().numpy()          # shape: (N, H_model, W_model)
        boxes = results.boxes.xyxy.cpu().numpy()          # (N, 4)
        classes = results.boxes.cls.cpu().numpy()         # (N,)
        confs = results.boxes.conf.cpu().numpy()          # (N,)

        orig_h, orig_w = image.shape[:2]

        for mask, box, cls_id, conf in zip(masks, boxes, classes, confs):
            class_name = self.class_names[int(cls_id)]

            # Skip if we're filtering by target classes
            if target_classes is not None and class_name not in target_classes:
                continue

            # Resize mask to original image resolution
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            )

            # Optional: binarize (some implementations threshold here)
            # mask_resized = (mask_resized > 0).astype(np.uint8) * 255

            detections.append({
                'class': class_name,
                'confidence': float(conf),
                'bbox': box.tolist(),
                'mask': mask_resized
            })

        return detections

    def get_combined_mask(self, detections: list[dict]) -> np.ndarray | None:
        """
        Merge all instance masks into a single binary mask (union of all detected objects).

        Parameters
        ----------
        detections : list of dict
            Output from detect_and_segment()

        Returns
        -------
        np.ndarray or None
            Combined binary mask (H×W, uint8), values 0 or 255
            Returns None if no detections
        """
        if not detections:
            return None

        # Start with the first mask
        combined = detections[0]['mask'].copy()

        # Union (logical OR) with all remaining masks
        for det in detections[1:]:
            combined = np.maximum(combined, det['mask'])

        return combined


# ────────────────────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick test / demo
    detector = ObjectDetectorSegmenter(
        model_path='yolov8m-seg.pt',
        conf_threshold=0.25
    )

    img = cv2.imread("test_image.jpg")
    if img is None:
        print("Could not load image")
    else:
        # Detect everything
        results = detector.detect_and_segment(img)

        # Or detect only specific classes
        # results = detector.detect_and_segment(img, target_classes=["person", "car"])

        print(f"Found {len(results)} objects")

        if results:
            combined_mask = detector.get_combined_mask(results)
            if combined_mask is not None:
                cv2.imwrite("combined_mask.jpg", combined_mask)
                print("Saved combined mask → combined_mask.jpg")
