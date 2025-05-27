import torch
    


def calculate_iou(logits, targets, num_classes=4):
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)  # Convert logits to class predictions
        targets = torch.argmax(targets, dim=1)  # Convert one-hot targets to class labels

        #print(f"Shape of logits: {logits.shape}")
        #print(f"Shape of preds: {preds.shape}")
        #print(f"Shape of targets: {targets.shape}")

        iou = []
        for cls in range(num_classes):
            pred_mask = (preds == cls)
            target_mask = (targets == cls)

            #print(f"Class {cls}:")
            #print(f"pred_mask shape: {pred_mask.shape}")
            #print(f"target_mask shape: {target_mask.shape}")

            intersection = (pred_mask & target_mask).float().sum((1, 2))
            union = (pred_mask | target_mask).float().sum((1, 2))

            #print(f"intersection: {intersection}")
            #print(f"union: {union}")

            iou.append((intersection + 1e-6) / (union + 1e-6))

        mean_iou = torch.stack(iou).mean(dim=0)  # Mean IoU across all classes
        #print(f"Mean IoU: {mean_iou}")
        return mean_iou

