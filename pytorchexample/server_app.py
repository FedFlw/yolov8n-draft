"""pytorchexample: A Flower / PyTorch YOLO detection server."""

from typing import List, Tuple
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from pytorchexample.task import create_model, get_weights
import time

def map50_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Weighted average aggregator for 'map50'. 
    Each tuple is (num_examples, {"map50": val, ...}).
    """
    # Collect mAP * examples 
    total_map50 = 0.0
    total_examples = 0

    for num_examples, m in metrics:
        if "map50" in m:
            total_map50 += num_examples * m["map50"]
            total_examples += num_examples

    if total_examples == 0:
        return {"map50": 0.0}

    return {"map50": total_map50 / total_examples}


def server_fn(context: Context):
    """Construct components defining server behavior."""
    num_rounds = context.run_config["num-server-rounds"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    start_time = time.time()

    # 1) Create an initial YOLO detection model
    model = create_model()
    init_weights = get_weights(model)
    parameters = ndarrays_to_parameters(init_weights)

    # 2) Define FedAvg strategy with custom aggregator for mAP
    strategy = FedAvg(
        fraction_fit=1,
        fraction_evaluate=0.5,
        min_available_clients=1,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=map50_weighted_average,
    )

    # 3) Create server config
    config = ServerConfig(num_rounds=num_rounds)

    server_app_components = ServerAppComponents(strategy=strategy, config=config)

    # Track end time
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total simulation time: {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds)")

    return server_app_components



app = ServerApp(server_fn=server_fn)
