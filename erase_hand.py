import argparse
from pathlib import Path

import cv2
import mediapipe as mp


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MARGIN_RATIO = 0.15


def collect_images(images_dir):
    return sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def crop_single_hand(image_bgr, hands_detector):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(image_rgb)
    if not result.multi_hand_landmarks:
        return None

    image_h, image_w = image_bgr.shape[:2]
    best_bbox = None
    best_area = -1

    for hand_landmarks in result.multi_hand_landmarks:
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]

        min_x = max(int(min(xs) * image_w), 0)
        max_x = min(int(max(xs) * image_w), image_w - 1)
        min_y = max(int(min(ys) * image_h), 0)
        max_y = min(int(max(ys) * image_h), image_h - 1)

        box_w = max_x - min_x + 1
        box_h = max_y - min_y + 1
        margin_x = int(box_w * MARGIN_RATIO)
        margin_y = int(box_h * MARGIN_RATIO)

        min_x = max(min_x - margin_x, 0)
        max_x = min(max_x + margin_x, image_w - 1)
        min_y = max(min_y - margin_y, 0)
        max_y = min(max_y + margin_y, image_h - 1)

        area = (max_x - min_x + 1) * (max_y - min_y + 1)
        if area > best_area:
            best_area = area
            best_bbox = (min_x, min_y, max_x, max_y)

    if best_bbox is None:
        return None

    min_x, min_y, max_x, max_y = best_bbox
    return image_bgr[min_y:max_y + 1, min_x:max_x + 1]


def process_directory(images_dir, save_root):
    images_dir = images_dir.resolve()
    output_dir = save_root.resolve() / images_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(images_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    processed = 0
    skipped = 0

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as hands_detector:
        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                skipped += 1
                print(f"Skip unreadable file: {image_path}")
                continue

            cropped_hand = crop_single_hand(image_bgr, hands_detector)
            if cropped_hand is None:
                skipped += 1
                print(f"Skip no-hand image: {image_path}")
                continue

            output_path = output_dir / image_path.name
            ok = cv2.imwrite(str(output_path), cropped_hand)
            if not ok:
                skipped += 1
                print(f"Skip failed save: {output_path}")
                continue

            processed += 1
            print(f"Saved: {output_path}")

    print(f"Done. Processed: {processed}, skipped: {skipped}, output: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Crop one hand from each image")
    parser.add_argument(
        "--images-dir",
        required=True,
        type=Path,
        help="Directory with input images",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        type=Path,
        help="Root directory for saving cropped hand images",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_directory(args.images_dir, args.save_dir)
