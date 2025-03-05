import torch
from ultralytics import YOLO

def create_model():
    return YOLO("yolov8n.pt")

def set_weights(yolo, parameters):
    state_dict = yolo.model.state_dict()
    for (name, old_val), new_val in zip(state_dict.items(), parameters):
        state_dict[name] = torch.tensor(new_val, dtype=old_val.dtype)
    yolo.model.load_state_dict(state_dict, strict=True)

def get_weights(yolo):
    return [val.cpu().numpy() for _, val in yolo.model.state_dict().items()]

def train(yolo: YOLO, epochs: int, lr: float, device: str, cid):
    dataset_path = f"/app/YOLOV8N/datasets/taco/client_{cid}/taco.yaml"
    results = yolo.train(data=dataset_path, epochs=100, imgsz=640)

    

def test(yolo: YOLO, device: str, cid):
    dataset_path = f"/app/YOLOV8N/datasets/taco/client_{cid}/taco.yaml"
    metrics = yolo.val(
        data=dataset_path,
        device=device,
        imgsz=640,
        batch=1,
    )
    
    # metrics.results_dict is the actual dictionary of results
    rd = metrics.results_dict  # -> dict

    # Now you can extract what you want:
    map50 = rd.get("metrics/mAP50(B)", 0.0)
    # For loss, you might see these keys in rd. Adjust if they differ:
    loss = rd.get("val/box_loss", 0.0) + rd.get("val/cls_loss", 0.0)

    return loss, {"map50": map50}
