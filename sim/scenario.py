"""Builds the two architectures and drives traffic + update events through them.

Baselines are tuned so both pipelines have roughly the same steady-state
latency/throughput -- the point of the experiment is what happens *around a
model update*, not who's faster at rest.
"""
from __future__ import annotations

import random

import simpy

from .core import Interface, Module, Pipeline
from .metrics import MetricsCollector


def build_monolithic(env: simpy.Environment) -> Pipeline:
    module = Module(
        env=env,
        name="monolith",
        base_latency=30.0,
        jitter=3.0,
        capacity=4,
        retrain_time=150.0,   # full model retrain/redeploy: slow
        degraded_quality=0.05,  # effectively down
        retrain_penalty_factor=25.0,
        )
    return Pipeline(env=env, name="monolithic", modules=[module], interfaces=[])


def build_modular(env: simpy.Environment) -> Pipeline:
    modules = [
        Module(
            env=env,
            name=f"module_{i}",
            base_latency=9.0,
            jitter=1.0,
            capacity=4,
            retrain_time=1.0,       # hot-swappable chip: near-instant
            degraded_quality=1.0,   # swapping the module itself costs ~nothing
            retrain_penalty_factor=0.0,
        )
        for i in range(3)
    ]
    interfaces = [
        Interface(
            env=env,
            name=f"iface_{i}",
            comm_latency=1.0,
            translation_cost=1.0,
            retrain_time=40.0,      # only the adapter needs re-aligning: fast
            degraded_quality=0.5,
            retrain_penalty_factor=3.0,
        )
        for i in range(len(modules) - 1)
    ]
    return Pipeline(env=env, name="modular", modules=modules, interfaces=interfaces)


def traffic_generator(env, pipeline: Pipeline, metrics: MetricsCollector, mean_interarrival: float):
    while True:
        yield env.timeout(random.expovariate(1.0 / mean_interarrival))
        env.process(pipeline.handle_request(metrics))


def update_scheduler(env, pipeline: Pipeline, update_times: list, module_index: int = 0):
    for t in update_times:
        yield env.timeout(max(0.0, t - env.now))
        pipeline.apply_update(module_index)


def run_scenario(total_time: float = 1000.0, mean_interarrival: float = 4.0,
                  update_times=(200.0, 500.0, 800.0), seed: int = 42):
    random.seed(seed)
    env = simpy.Environment()
    mono = build_monolithic(env)
    modular = build_modular(env)
    metrics = MetricsCollector()

    env.process(traffic_generator(env, mono, metrics, mean_interarrival))
    env.process(traffic_generator(env, modular, metrics, mean_interarrival))
    env.process(update_scheduler(env, mono, list(update_times), module_index=0))
    # rotate which module gets updated each time, to show it doesn't matter which one
    env.process(update_scheduler(env, modular, [update_times[0]], module_index=0))
    env.process(update_scheduler(env, modular, [update_times[1]], module_index=1))
    env.process(update_scheduler(env, modular, [update_times[2]], module_index=2))

    env.run(until=total_time)
    return metrics, list(update_times)
