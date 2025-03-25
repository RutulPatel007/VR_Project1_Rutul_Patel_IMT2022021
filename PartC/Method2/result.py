import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_image(image, method='threshold'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 'threshold':
        _, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'edge':
        segmented = cv2.Canny(gray, 50, 150)
    else:
        raise ValueError("Invalid segmentation method")
    
    return segmented

def evaluate_segmentation(ground_truth, predicted):
    intersection = np.logical_and(ground_truth, predicted).sum()
    union = np.logical_or(ground_truth, predicted).sum()
    
    iou = intersection / union if union > 0 else 0
    dice = (2 * intersection) / (ground_truth.sum() + predicted.sum()) if (ground_truth.sum() + predicted.sum()) > 0 else 0
    
    return iou, dice

def process_images(image_folder, ground_truth_folder, output_folder, method='threshold'):
    iou_scores, dice_scores = [], []
    
    method_folder = os.path.join(output_folder, method)
    os.makedirs(method_folder, exist_ok=True)
    
    filenames = sorted(os.listdir(image_folder))
    
    for filename in filenames:
        img_path = os.path.join(image_folder, filename)
        gt_path = os.path.join(ground_truth_folder, filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found - {img_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth not found - {gt_path}")
            continue
        
        image = cv2.imread(img_path)
        ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or ground_truth is None:
            print(f"Error loading files: {filename}")
            continue

        try:
            segmented = segment_image(image, method)
            output_path = os.path.join(method_folder, filename)
            cv2.imwrite(output_path, segmented)
            
            resized_gt = cv2.resize(ground_truth, (segmented.shape[1], segmented.shape[0]))
            iou, dice = evaluate_segmentation(resized_gt > 0, segmented > 0)
            iou_scores.append(iou)
            dice_scores.append(dice)

            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')

            plt.subplot(1, 2, 2)
            plt.imshow(segmented, cmap='gray')
            plt.title(f'Segmented ({method})')

            vis_path = os.path.join(method_folder, f"vis_{filename}.png")
            plt.savefig(vis_path)
            plt.close()

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    
    print(f"Mean IoU ({method}): {mean_iou:.4f}")
    print(f"Mean Dice Score ({method}): {mean_dice:.4f}")
    
    with open(os.path.join(output_folder, f"accuracy_{method}.txt"), "w") as file:
        file.write(f"Mean IoU ({method}): {mean_iou:.4f}\n")
        file.write(f"Mean Dice Score ({method}): {mean_dice:.4f}\n")

image_folder = "../../MSFD/1/face_crop"
ground_truth_folder = "../../MSFD/1/face_crop_segmentation"
output_folder = "./segmented_output"

process_images(image_folder, ground_truth_folder, output_folder, method='threshold')
process_images(image_folder, ground_truth_folder, output_folder, method='edge')