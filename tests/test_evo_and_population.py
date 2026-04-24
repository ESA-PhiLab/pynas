import random
import threading
import time
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from pynas.core.population import Population
from pynas.core.individual import Individual
from pynas.opt.evo import single_point_crossover


class DummyIndividual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.parsed_layers = chromosome[:]

    def _reparse_layers(self):
        self.parsed_layers = self.chromosome[:]


def test_single_point_crossover_does_not_mutate_parents():
    random.seed(4)
    parent0 = DummyIndividual(["L0", "P0", "L1", "P1"])
    parent1 = DummyIndividual(["L2", "P2", "L3", "P3"])
    parent0_chromosome = parent0.chromosome[:]
    parent1_chromosome = parent1.chromosome[:]

    children = single_point_crossover([parent0, parent1])

    assert parent0.chromosome == parent0_chromosome
    assert parent1.chromosome == parent1_chromosome
    assert children[0] is not parent0
    assert children[1] is not parent1


def test_population_extracts_segmentation_metrics_from_lightning_results():
    metric, fps = Population._extract_fitness_inputs(
        [{"test_iou": torch.tensor(0.25), "fps": 40.0}],
        task="segmentation",
    )

    assert float(metric) == pytest.approx(0.25)
    assert float(fps) == pytest.approx(40.0)


def test_population_extracts_classification_metrics_from_lightning_results():
    metric, fps = Population._extract_fitness_inputs(
        [{"test_accuracy": torch.tensor(0.75), "fps": torch.tensor(120.0)}],
        task="classification",
    )

    assert float(metric) == pytest.approx(0.75)
    assert float(fps) == pytest.approx(120.0)


def test_create_random_individual_respects_parameter_limit():
    random.seed(123)
    torch.manual_seed(123)
    dm = SimpleNamespace(input_shape=(3, 16, 16), num_classes=2)
    population = Population(
        n_individuals=1,
        max_layers=4,
        dm=dm,
        max_parameters=5_000_000,
    )

    for _ in range(6):
        individual = population.create_random_individual(max_attempts=20)

        assert individual.model_size is not None
        assert 0 < individual.model_size < population.max_parameters


def test_individual_allows_small_max_layers():
    random.seed(11)

    individual = Individual(max_layers=2)

    convolution_layers = [
        layer
        for layer in individual.parsed_layers
        if layer["layer_type"] not in {"AvgPool", "MaxPool"}
    ]
    assert 1 <= len(convolution_layers) <= 2


def test_train_generation_runs_pending_individuals_in_parallel():
    class ParallelPopulation(Population):
        def __init__(self):
            self.population = [SimpleNamespace(fitness=0.0) for _ in range(4)]
            self.df = None
            self.generation = 0
            self._state_lock = threading.RLock()
            self.logger = SimpleNamespace(
                info=lambda *args, **kwargs: None,
                error=lambda *args, **kwargs: None,
            )
            self.active = 0
            self.max_active = 0
            self.calls = []
            self.call_lock = threading.Lock()

        def _update_df(self):
            self.df = pd.DataFrame(
                {
                    "Fitness": [0.0 for _ in self.population],
                    "Metric": [None for _ in self.population],
                    "FPS": [None for _ in self.population],
                }
            )

        def train_individual(self, idx, task, epochs=20, lr=1e-3, batch_size=None):
            with self.call_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.05)
            with self.call_lock:
                self.calls.append((idx, task, epochs, lr, batch_size))
                self.active -= 1

    population = ParallelPopulation()

    population.train_generation(
        task="segmentation",
        lr=0.02,
        epochs=3,
        batch_size=7,
        training_workers=2,
    )

    assert sorted(call[0] for call in population.calls) == [0, 1, 2, 3]
    assert {call[1:] for call in population.calls} == {("segmentation", 3, 0.02, 7)}
    assert population.max_active == 2


def test_train_generation_rejects_invalid_worker_count():
    population = Population.__new__(Population)
    population.population = []
    population.df = None

    with pytest.raises(ValueError, match="training_workers"):
        population.train_generation(training_workers=0)
