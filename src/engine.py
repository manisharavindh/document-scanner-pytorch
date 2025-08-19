import torch
# from tqdm.auto import tqdm
from typing import List, Tuple, Dict

# device agnostic code
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
)

# train step
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim,
               device: torch.device = device) -> Tuple[float, float]:
    train_loss, train_acc = 0, 0
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        X = batch["image"].to(device)
        y = batch["corners"].to(device)
        y_logits = model(X)

        y = y.view(y.size(0), -1)
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            threshold = 0.05  # pixels
            pred_points = y_logits.view(-1, 4, 2)
            true_points = y.view(-1, 4, 2)
            
            # DEBUG: Print actual values
            if batch_idx == 0:  # Only print for first batch
                print(f"Pred range: [{pred_points.min():.3f}, {pred_points.max():.3f}]")
                print(f"True range: [{true_points.min():.3f}, {true_points.max():.3f}]")
                print(f"Sample pred corner: {pred_points[0, 0].cpu().numpy()}")
                print(f"Sample true corner: {true_points[0, 0].cpu().numpy()}")
            
            point_distances = torch.norm(pred_points - true_points, dim=2)
            
            # DEBUG: Print distances
            if batch_idx == 0:
                print(f"Distance range: [{point_distances.min():.3f}, {point_distances.max():.3f}]")
                print(f"Points within {threshold} pixels: {(point_distances < threshold).sum().item()}/{point_distances.numel()}")
            
            accuracy = (point_distances < threshold).float().mean().item() * 100
            train_acc += accuracy

    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc

# test step (validation)
def val_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device = device) -> Tuple[float, float]:
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            X = batch["image"].to(device)
            y = batch["corners"].to(device)
            # X, y = X.to(device), y.to(device)
            test_logits = model(X)

            y = y.view(y.size(0), -1)
            loss = loss_fn(test_logits, y)
            test_loss += loss.item()

            with torch.no_grad():
                threshold = 0.05  # pixels
                pred_points = test_logits.view(-1, 4, 2)
                true_points = y.view(-1, 4, 2)
                
                # DEBUG: Print actual values
                if batch_idx == 0:  # Only print for first batch
                    print(f"Pred range: [{pred_points.min():.3f}, {pred_points.max():.3f}]")
                    print(f"True range: [{true_points.min():.3f}, {true_points.max():.3f}]")
                    print(f"Sample pred corner: {pred_points[0, 0].cpu().numpy()}")
                    print(f"Sample true corner: {true_points[0, 0].cpu().numpy()}")
                
                point_distances = torch.norm(pred_points - true_points, dim=2)
                
                # DEBUG: Print distances
                if batch_idx == 0:
                    print(f"Distance range: [{point_distances.min():.3f}, {point_distances.max():.3f}]")
                    print(f"Points within {threshold} pixels: {(point_distances < threshold).sum().item()}/{point_distances.numel()}")
                
                accuracy = (point_distances < threshold).float().mean().item() * 100
                test_acc += accuracy

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc

# train function
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim,
          epochs: int,
          device: torch.device = device) -> Dict[str, List[float]]:
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    model.to(device)
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        test_loss, test_acc = val_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(
            f"Epochs: {epoch} | "
            f"Train Loss: {train_loss:.4f} |"
            f"Train Acc: {train_acc:.2f}% |"
            f"Test Loss: {test_loss:.4f} |"
            f"Test Acc: {test_acc:.2f}%"
         )
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    
    return results