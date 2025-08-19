import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import torchvision.transforms as transforms

def make_predictions(model: torch.nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    device: torch.device,
                    num_samples: int = 5) -> Tuple[List, List, List]:
    """
    Make predictions on a subset of data and return images, true corners, and predicted corners.
    
    Args:
        model: Trained model
        dataloader: DataLoader containing test data
        device: Device to run inference on
        num_samples: Number of samples to predict on
    
    Returns:
        Tuple of (images, true_corners, pred_corners)
    """
    model.eval()
    images, true_corners, pred_corners = [], [], []
    
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if len(images) >= num_samples:
                break
                
            X = batch["image"].to(device)
            y = batch["corners"].to(device)
            
            # Make predictions
            pred_logits = model(X)
            
            # Convert to numpy for visualization
            batch_images = X.cpu().numpy()
            batch_true = y.cpu().numpy()
            batch_pred = pred_logits.cpu().numpy()
            
            # Add samples from this batch
            for i in range(min(X.size(0), num_samples - len(images))):
                images.append(batch_images[i])
                true_corners.append(batch_true[i])
                pred_corners.append(batch_pred[i])
    
    return images, true_corners, pred_corners

def denormalize_image(image: np.ndarray, 
                     mean: List[float] = [0.485, 0.456, 0.406], 
                     std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Denormalize image for visualization (adjust mean/std based on your preprocessing).
    """
    # Convert from CHW to HWC
    if image.shape[0] == 3:  # if channels first
        image = np.transpose(image, (1, 2, 0))
    
    # Denormalize
    mean = np.array(mean)
    std = np.array(std)
    image = (image * std) + mean
    image = np.clip(image, 0, 1)
    
    return image

def plot_predictions(images: List[np.ndarray], 
                    true_corners: List[np.ndarray], 
                    pred_corners: List[np.ndarray],
                    original_img_size: Tuple[int, int] = (1200, 800),  # Original image size (width, height)
                    resized_img_size: Tuple[int, int] = (224, 224),   # Model input size (height, width)
                    figsize: Tuple[int, int] = (15, 10)):
    """
    Plot images with true and predicted corners overlaid.
    
    Args:
        images: List of image arrays (model inputs)
        true_corners: List of true corner coordinates (normalized or absolute)
        pred_corners: List of predicted corner coordinates
        original_img_size: Original image dimensions (width, height) from your JSON
        resized_img_size: Resized image dimensions (height, width) that the model uses
        figsize: Figure size for matplotlib
    """
    num_samples = len(images)
    cols = min(3, num_samples)  # Max 3 columns
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Prepare image for display
        img = denormalize_image(images[i])
        ax.imshow(img)
        
        # Get actual displayed image dimensions
        img_height, img_width = img.shape[:2]
        
        # Reshape corners from flat to (4, 2) if needed
        true_pts = true_corners[i].reshape(4, 2)
        pred_pts = pred_corners[i].reshape(4, 2)
        
        # Handle coordinate scaling based on your preprocessing
        # Case 1: If corners are normalized to [0,1] during preprocessing
        if true_pts.max() <= 1.0:
            # Scale from [0,1] to display image size
            true_pts = true_pts * np.array([img_width, img_height])
            pred_pts = pred_pts * np.array([img_width, img_height])
        
        # Case 2: If corners are in original image coordinates, scale to display size
        elif true_pts.max() > resized_img_size[0]:  # Likely in original coordinates
            # Scale from original image size to display size
            scale_x = img_width / original_img_size[0]
            scale_y = img_height / original_img_size[1]
            true_pts = true_pts * np.array([scale_x, scale_y])
            pred_pts = pred_pts * np.array([scale_x, scale_y])
        
        # Case 3: Already in the correct coordinate system
        # (no scaling needed)
        
        # Plot true corners (green circles)
        ax.scatter(true_pts[:, 0], true_pts[:, 1], 
                  c='green', s=100, marker='o', alpha=0.8, label='True corners', edgecolors='darkgreen', linewidth=2)
        
        # Plot predicted corners (red crosses)
        ax.scatter(pred_pts[:, 0], pred_pts[:, 1], 
                  c='red', s=120, marker='x', alpha=0.8, label='Predicted corners', linewidth=3)
        
        # Draw lines connecting corners to show the quadrilateral
        # True corners (green lines) - connect in order: top-left, top-right, bottom-right, bottom-left
        true_quad = np.vstack([true_pts, true_pts[0]])  # Close the shape
        ax.plot(true_quad[:, 0], true_quad[:, 1], 'g-', alpha=0.7, linewidth=2, label='True quad')
        
        # Predicted corners (red lines)  
        pred_quad = np.vstack([pred_pts, pred_pts[0]])  # Close the shape
        ax.plot(pred_quad[:, 0], pred_quad[:, 1], 'r--', alpha=0.7, linewidth=2, label='Pred quad')
        
        # Add corner labels
        corner_labels = ['TL', 'TR', 'BR', 'BL']  # Assuming this order
        for j, (label, true_pt, pred_pt) in enumerate(zip(corner_labels, true_pts, pred_pts)):
            ax.annotate(f'{label}', (true_pt[0], true_pt[1]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, color='green', weight='bold')
        
        # Calculate and display error (scale back to original image coordinates for meaningful error)
        if true_pts.max() <= img_width:  # If coordinates are in display space
            # Scale back to original image space for error calculation
            scale_x = original_img_size[0] / img_width
            scale_y = original_img_size[1] / img_height
            true_orig = true_pts * np.array([scale_x, scale_y])
            pred_orig = pred_pts * np.array([scale_x, scale_y])
            error = np.mean(np.linalg.norm(true_orig - pred_orig, axis=1))
        else:
            error = np.mean(np.linalg.norm(true_pts - pred_pts, axis=1))
        
        ax.set_title(f'Sample {i+1}\nMean Error: {error:.2f} pixels', fontsize=10)
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Set axis limits to show full image
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Invert y-axis for image coordinates
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def calculate_corner_errors(true_corners: List[np.ndarray], 
                           pred_corners: List[np.ndarray]) -> dict:
    """
    Calculate various error metrics for corner predictions.
    """
    errors = []
    for true_pts, pred_pts in zip(true_corners, pred_corners):
        true_pts = true_pts.reshape(4, 2)
        pred_pts = pred_pts.reshape(4, 2)
        
        # Calculate per-corner errors
        corner_errors = np.linalg.norm(true_pts - pred_pts, axis=1)
        errors.append(corner_errors)
    
    errors = np.array(errors)
    
    return {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'per_corner_mean': np.mean(errors, axis=0)  # Average error for each corner
    }

# Example usage:
"""
# Load your trained model
model.load_state_dict(torch.load('your_model.pth'))

# Make predictions
images, true_corners, pred_corners = make_predictions(
    model=model,
    dataloader=test_dataloader,  # or val_dataloader
    device=device,
    num_samples=6
)

# Plot results with correct image dimensions
plot_predictions(images, true_corners, pred_corners,
                original_img_size=(1200, 800),  # Your original image size (width, height)
                resized_img_size=(224, 224))    # Your model input size (height, width)

# Calculate error metrics
error_stats = calculate_corner_errors(true_corners, pred_corners)
print("Error Statistics:")
for key, value in error_stats.items():
    if isinstance(value, np.ndarray):
        print(f"{key}: {value}")
    else:
        print(f"{key}: {value:.4f}")
"""

# If you want to debug coordinate systems, add this helper function:
def debug_coordinates(true_corners, pred_corners, sample_idx=0):
    """Debug function to check coordinate ranges and scaling"""
    true_pts = true_corners[sample_idx].reshape(4, 2)
    pred_pts = pred_corners[sample_idx].reshape(4, 2)
    
    print(f"Sample {sample_idx}:")
    print(f"True corners range: X=[{true_pts[:, 0].min():.2f}, {true_pts[:, 0].max():.2f}], Y=[{true_pts[:, 1].min():.2f}, {true_pts[:, 1].max():.2f}]")
    print(f"Pred corners range: X=[{pred_pts[:, 0].min():.2f}, {pred_pts[:, 0].max():.2f}], Y=[{pred_pts[:, 1].min():.2f}, {pred_pts[:, 1].max():.2f}]")
    print(f"True corners:\n{true_pts}")
    print(f"Pred corners:\n{pred_pts}")
    print("-" * 50)