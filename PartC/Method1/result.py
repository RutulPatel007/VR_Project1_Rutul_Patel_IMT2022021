import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    return df[df['with_mask'] == True]  

def segment_image(image, method='threshold'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    o
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
    
    iou = intersection / union if union > 0 else 0  # IoU calculation
    dice = (2 * intersection) / (ground_truth.sum() + predicted.sum()) if (ground_truth.sum() + predicted.sum()) > 0 else 0  # Dice Score
    
    return iou, dice

def process_images(csv_path, image_folder, output_folder, method='threshold'):
    df = load_dataset(csv_path)
    iou_scores, dice_scores = [], []
    
    os.makedirs(output_folder, exist_ok=True)
    method_folder = os.path.join(output_folder, method)
    os.makedirs(method_folder, exist_ok=True)
    
    for index, row in df.iterrows():
        img_path = os.path.join(image_folder, row['filename'])
        if not os.path.exists(img_path):
            print(f"Warning: Image not found - {img_path}")
            continue
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error loading image: {img_path}")
            continue
        
        try:
            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
            face_crop = image[y1:y2, x1:x2]
            
            segmented = segment_image(face_crop, method)
            
            output_path = os.path.join(method_folder, row['filename'])
            cv2.imwrite(output_path, segmented)
            
            ground_truth = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) > 128  
            
            iou, dice = evaluate_segmentation(ground_truth, segmented > 0)
            iou_scores.append(iou)
            dice_scores.append(dice)
            
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            plt.title('Original Face')
            
            plt.subplot(1, 2, 2)
            plt.imshow(segmented, cmap='gray')
            plt.title(f'Segmented ({method})')
            
            vis_path = os.path.join(method_folder, f"vis_{row['filename']}.png")
            plt.savefig(vis_path)
            plt.close()
        
        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")
            continue
    
    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    
    print(f"Mean IoU ({method}): {mean_iou:.4f}")
    print(f"Mean Dice Score ({method}): {mean_dice:.4f}")
    
    with open(os.path.join(output_folder, f"accuracy_{method}.txt"), "w") as file:
        file.write(f"Mean IoU ({method}): {mean_iou:.4f}\n")
        file.write(f"Mean Dice Score ({method}): {mean_dice:.4f}\n")

csv_path = "../../MSFD/1/dataset.csv"
image_folder = "../../MSFD/1/img"
output_folder = "./segmented_output"

process_images(csv_path, image_folder, output_folder, method='threshold')
process_images(csv_path, image_folder, output_folder, method='edge')
