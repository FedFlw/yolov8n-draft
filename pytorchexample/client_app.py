"""pytorchexample: A Flower / PyTorch YOLO detection client."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from ultralytics import YOLO
from pytorchexample.train import DetectionTrainer
from pytorchexample.val import DetectionValidator
from pytorchexample.task import create_model, get_weights, set_weights, train, test

class YOLODetectionClient(NumPyClient):
    def __init__(self, local_epochs, learning_rate,cid):
        # Keep the entire YOLO object
        self.net = YOLO("yolov8n.pt")
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.cid = cid
        print("CUDA AVAILABLE", torch.cuda.is_available())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        results = train(self.net, epochs=self.local_epochs, lr=self.lr, device=self.device, cid=self.cid)
        updated_weights = get_weights(self.net)
        # For demonstration, number of examples might just be 8 for coco8
        return updated_weights, 8, {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, metrics = test(self.net, self.device, cid= self.cid)
        return loss, 8, metrics

def client_fn(context: Context):
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    cid = context.node_id
    return YOLODetectionClient( local_epochs, learning_rate, cid).to_client()

app = ClientApp(client_fn)
