"""Core simulation primitives: Module, Interface, RetrainState, Pipeline.

Mental model:
- Module   = one physical "chip" running a chunk of the computation.
- Interface= a learned translation layer connecting two modules' representation
             spaces (see LLaVA-style projector). It has to be retrained whenever
             either module it touches changes.
- Monolithic architecture = a single Module, zero Interfaces. Updating it means
  retraining/redeploying the whole thing (no isolation).
- Modular architecture   = a chain of Modules + Interfaces. Updating one Module
  only forces its adjacent Interfaces to re-align; the rest of the chain is
  untouched.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import simpy


@dataclass
class RetrainState:
    """Shared 'this thing just got destabilized by a swap' state machine.

    Quality decays to `degraded_quality` the instant a swap happens, then
    recovers linearly back to 1.0 over `retrain_time`.
    """

    env: simpy.Environment
    retrain_time: float
    degraded_quality: float
    _retrain_until: float = field(default=-1.0, init=False)

    def notify(self) -> None:
        self._retrain_until = self.env.now + self.retrain_time

    @property
    def is_retraining(self) -> bool:
        return self.env.now < self._retrain_until

    def quality(self) -> float:
        if not self.is_retraining:
            return 1.0
        remaining = self._retrain_until - self.env.now
        recovered = 1.0 - (remaining / self.retrain_time)
        return self.degraded_quality + (1.0 - self.degraded_quality) * recovered


@dataclass
class Module:
    env: simpy.Environment
    name: str
    base_latency: float
    jitter: float
    capacity: int
    retrain_time: float
    degraded_quality: float
    retrain_penalty_factor: float
    version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        self.resource = simpy.Resource(self.env, capacity=self.capacity)
        self.retrain = RetrainState(self.env, self.retrain_time, self.degraded_quality)

    def process(self):
        q = self.retrain.quality()
        penalty = 1.0 + (1.0 - q) * self.retrain_penalty_factor
        delay = max(0.001, random.gauss(self.base_latency, self.jitter)) * penalty
        yield self.env.timeout(delay)
        return q

    def swap(self) -> None:
        self.version += 1
        self.retrain.notify()


@dataclass
class Interface:
    env: simpy.Environment
    name: str
    comm_latency: float
    translation_cost: float
    retrain_time: float
    degraded_quality: float
    retrain_penalty_factor: float

    def __post_init__(self) -> None:
        self.retrain = RetrainState(self.env, self.retrain_time, self.degraded_quality)

    def process(self):
        q = self.retrain.quality()
        penalty = 1.0 + (1.0 - q) * self.retrain_penalty_factor
        delay = self.comm_latency + self.translation_cost * penalty
        yield self.env.timeout(delay)
        return q


@dataclass
class Pipeline:
    env: simpy.Environment
    name: str
    modules: list
    interfaces: list  # len(modules) - 1, connecting modules[i] -> modules[i+1]

    def handle_request(self, metrics):
        start = self.env.now
        quality = 1.0
        for i, module in enumerate(self.modules):
            with module.resource.request() as req:
                yield req
                q = yield from module.process()
                quality = min(quality, q)
            if i < len(self.interfaces):
                q = yield from self.interfaces[i].process()
                quality = min(quality, q)
        latency = self.env.now - start
        metrics.record(self.name, self.env.now, latency, quality)

    def apply_update(self, module_index: int) -> None:
        """Simulate shipping a new version of modules[module_index]."""
        module = self.modules[module_index]
        module.swap()
        if self.interfaces:
            if module_index > 0:
                self.interfaces[module_index - 1].retrain.notify()
            if module_index < len(self.interfaces):
                self.interfaces[module_index].retrain.notify()
        # else: monolithic — the module's own retrain state IS the outage.
