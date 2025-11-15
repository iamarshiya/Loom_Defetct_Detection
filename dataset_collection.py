import cv2
import numpy as np
import os
import pandas as pd

BASE_DIR = r"F:\CopperCloud\LOOM\3_Datasets"

def get_next_index(folder, label):
    path = os.path.join(folder, label)
    os.makedirs(path, exist_ok=True)
    files = [f for f in os.listdir(path) if f.startswith(label)]
    if not files:
        return 1
    nums = []
    for f in files:
        num_part = ''.join([c for c in f if c.isdigit()])
        if num_part.isdigit():
            nums.append(int(num_part))
    return max(nums, default=0) + 1

def save_datasets(b_norm, roi_binary, line_detected, force_label=None):
    # Allow forced label from keypress (s/n)
    if force_label is not None:
        label = force_label
    else:
        label = "silver" if line_detected else "non_silver"

    # Paths
    paths = {
        "y_channel_b": os.path.join(BASE_DIR, "Y_channel_B"),
        "binary_images": os.path.join(BASE_DIR, "Binary_images"),
        "binary_csv": os.path.join(BASE_DIR, "Binary_csv")
    }

    for path in paths.values():
        os.makedirs(os.path.join(path, label), exist_ok=True)

    index = get_next_index(paths["y_channel_b"], label)
    index_str = f"{index:03d}"

    # --- Save Y-channel (b_norm) ---
    yb_path = os.path.join(paths["y_channel_b"], label, f"{label}_{index_str}.png")
    cv2.imwrite(yb_path, b_norm)

    # --- Save Binary Image ---
    binary_path = os.path.join(paths["binary_images"], label, f"{label}_{index_str}.png")
    cv2.imwrite(binary_path, roi_binary)

    # --- Save Binary CSV ---
    binary_matrix = (roi_binary // 255).astype(int)
    csv_path = os.path.join(paths["binary_csv"], label, f"{label}_{index_str}.csv")
    pd.DataFrame(binary_matrix).to_csv(csv_path, index=False, header=False)

    print(f"✅ Saved: {label}_{index_str}")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Convert to Lab color space ---
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)

        # --- CLAHE on L-channel ---
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        frame_eq = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)

        # --- Extract ROI from b-channel ---
        h, w = b.shape
        start_col = w // 3
        end_col = 2 * w // 3
        roi_b = b[:, start_col:end_col]
        crop_top = h // 4
        roi_b_cropped = roi_b[crop_top:, :]

        # --- Normalize + denoise (this becomes Y-channel-B) ---
        b_norm = cv2.normalize(roi_b_cropped, None, 0, 255, cv2.NORM_MINMAX)
        b_norm = cv2.medianBlur(b_norm, 5)

        # --- Binary threshold ---
        _, roi_binary = cv2.threshold(
            b_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # --- Morphological vertical closure ---
        vertical_kernel = np.ones((30, 3), np.uint8)
        roi_closed = cv2.morphologyEx(roi_binary, cv2.MORPH_CLOSE, vertical_kernel)

        # --- Connected component analysis ---
        num_labels, labels = cv2.connectedComponents(roi_closed)
        line_detected = False
        for label in range(1, num_labels):
            ys, xs = np.where(labels == label)
            if len(ys) == 0:
                continue
            min_y, max_y = ys.min(), ys.max()
            if min_y == 0 and max_y == roi_closed.shape[0] - 1:
                line_detected = True
                break

        # --- Display (smaller windows) ---
        display_scale = 0.6
        display_bnorm = cv2.resize(b_norm, (0, 0), fx=display_scale, fy=display_scale)
        display_binary = cv2.resize(roi_closed, (0, 0), fx=display_scale, fy=display_scale)

        cv2.imshow("Y Channel B (b_norm)", display_bnorm)
        cv2.imshow("Binary ROI", display_binary)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Save as silver regardless of detection
            save_datasets(b_norm, roi_closed, line_detected, force_label="silver")
        elif key == ord('n'):
            # Save as non_silver regardless of detection
            save_datasets(b_norm, roi_closed, line_detected, force_label="non_silver")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
