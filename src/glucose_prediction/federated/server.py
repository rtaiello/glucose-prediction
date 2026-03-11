"""Federated server: FedAvg aggregation, client selection, and federation orchestration."""

import logging
import random
from typing import Dict, List

import torch
from tqdm import tqdm

from glucose_prediction.federated.client import FederatedClient
from glucose_prediction.federated.train_utils import make_model, set_model_weights
from glucose_prediction.modules.cnn_lstm import CNN_LSTM

pylogger = logging.getLogger(__name__)


class FederatedServer:
    """Coordinates FedAvg with optional client selection per round."""

    def __init__(
        self,
        clients: List[FederatedClient],
        device: torch.device = torch.device("cpu"),
        input_length: int = 12,
        seed: int = 42,
    ) -> None:
        self.clients = clients
        self.device = device
        self.global_model: CNN_LSTM = make_model(input_length=input_length)
        self.round_history: List[float] = []
        self._rng = random.Random(seed)

    def _select_clients(self, fraction: float) -> List[FederatedClient]:
        """Return a random subset of clients for this round."""
        n_select = max(1, int(len(self.clients) * fraction))
        return self._rng.sample(self.clients, n_select)

    def aggregate(self, client_weights: List[Dict], n_samples: List[int]) -> Dict:
        """FedAvg: weighted average of client state dicts.

        Uses float64 accumulators for numerical stability, casts back to float32.
        """
        total = sum(n_samples)
        aggregated: Dict = {}
        for key in client_weights[0]:
            acc = torch.zeros_like(client_weights[0][key], dtype=torch.float64)
            for weights, n in zip(client_weights, n_samples):
                acc += weights[key].double() * n
            aggregated[key] = (acc / total).float()
        return aggregated

    def run_round(self, n_local_steps: int, client_fraction: float = 1.0) -> dict:
        """Broadcast global weights, select clients, run local training, aggregate.

        Returns a dict with round stats: avg_loss, n_selected, selected_ids.
        """
        global_weights = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
        selected = self._select_clients(client_fraction)

        # Broadcast to selected clients only
        for client in selected:
            client.set_weights(global_weights)

        # Local training
        client_losses = []
        client_weights = []
        n_samples = []
        for client in selected:
            loss = client.local_train(n_local_steps)
            client_losses.append(loss)
            client_weights.append(client.get_weights())
            n_samples.append(client.get_n_samples())

        # Aggregate
        new_weights = self.aggregate(client_weights, n_samples)
        set_model_weights(self.global_model, new_weights)

        return {
            "avg_loss": sum(client_losses) / len(client_losses),
            "n_selected": len(selected),
            "selected_ids": [c.patient_id for c in selected],
        }

    def run_federation(
        self,
        n_rounds: int,
        n_local_steps: int,
        client_fraction: float = 1.0,
        verbose: bool = True,
    ) -> None:
        """Run the full federated training loop."""
        n_total = len(self.clients)
        n_select = max(1, int(n_total * client_fraction))
        if verbose:
            print(f"  Client selection: {n_select}/{n_total} per round (fraction={client_fraction:.2f})")

        iterator = tqdm(range(1, n_rounds + 1), desc="FL Rounds") if verbose else range(1, n_rounds + 1)
        for rnd in iterator:
            stats = self.run_round(n_local_steps, client_fraction)
            self.round_history.append(stats["avg_loss"])
            if verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    loss=f"{stats['avg_loss']:.3f}",
                    selected=f"{stats['n_selected']}/{n_total}",
                )
            pylogger.info(
                f"Round {rnd}/{n_rounds} — loss: {stats['avg_loss']:.4f} "
                f"({stats['n_selected']}/{n_total} clients)"
            )
