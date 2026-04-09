"""
BridgeTwin Resource Scheduler — Multi-agent construction logistics optimizer.

Models construction repair logistics as a Common Pool Resource (CPR) problem
and solves it with a lightweight tabular Q-learning approach inspired by
PufferLib's Ocean environment suite.

Key concepts:
  - **Agents**: Distinct operational entities (crew, equipment, traffic control).
  - **Slots**: Discrete time windows (e.g. 1-hour blocks in a 7-day horizon).
  - **Constraints**: No overlapping bookings for the same physical resource.
  - **Reward**: Maximise coverage of required tasks; heavily penalise overlaps.

The scheduler runs entirely offline and in pure Python (no external RL library
needed on the edge device).  For high-throughput offline training on the
RTX 5090 or M4 Max, swap the ``TabularQScheduler`` for a PufferLib-backed
implementation — the ``SchedulerEnv`` class is designed to be drop-in
compatible with PufferLib's ``PufferEnv`` interface.

Typical usage:
    from scheduler import SchedulerEnv, TabularQScheduler

    env = SchedulerEnv(
        n_slots=168,           # 7 days × 24 hours
        resources=["bridge-S47", "road-closure-S47", "concrete-crew-A"],
        tasks=[
            {"id": "T1", "resource": "concrete-crew-A", "duration": 4, "earliest": 0},
            {"id": "T2", "resource": "road-closure-S47", "duration": 6, "earliest": 0},
        ],
    )
    sched = TabularQScheduler(env)
    sched.train(episodes=500)
    schedule = sched.best_schedule()
    print(sched.explain(schedule))
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single maintenance or repair task that must be scheduled."""

    id: str
    resource: str          # Which resource (crew / equipment / road) this task consumes
    duration: int          # Number of time slots required
    earliest: int = 0      # Earliest start slot
    latest: Optional[int] = None  # Latest start slot (None = no constraint)
    priority: int = 1      # Higher = more important

    def __post_init__(self) -> None:
        if self.duration < 1:
            raise ValueError(f"Task {self.id}: duration must be ≥ 1")


@dataclass
class ScheduleEntry:
    """A committed booking of a resource for a task."""

    task_id: str
    resource: str
    start_slot: int
    end_slot: int          # exclusive — slots [start, end)

    @property
    def slots(self) -> range:
        return range(self.start_slot, self.end_slot)

    def overlaps(self, other: "ScheduleEntry") -> bool:
        """Return True if two entries use the same resource in overlapping slots."""
        if self.resource != other.resource:
            return False
        return self.start_slot < other.end_slot and other.start_slot < self.end_slot


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SchedulerEnv:
    """Discrete scheduling environment.

    State:   dict mapping task_id → assigned start_slot (or -1 = unscheduled)
    Action:  (task_index, start_slot)
    Reward:
      +priority   for scheduling a task without overlap
      -10         for each overlap (same resource, same slot window)
      +5          for completing the full schedule (all tasks assigned)
    """

    UNSCHEDULED = -1

    def __init__(
        self,
        n_slots: int,
        resources: list[str],
        tasks: list[dict[str, Any] | Task],
    ) -> None:
        self.n_slots = n_slots
        self.resources = list(resources)

        # Normalise tasks
        self._tasks: list[Task] = []
        for t in tasks:
            if isinstance(t, Task):
                self._tasks.append(t)
            else:
                self._tasks.append(Task(**t))

        self.reset()

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, int]:
        """Reset the environment and return the initial state."""
        self._state: dict[str, int] = {t.id: self.UNSCHEDULED for t in self._tasks}
        return copy.copy(self._state)

    @property
    def n_tasks(self) -> int:
        return len(self._tasks)

    def step(
        self, task_index: int, start_slot: int
    ) -> tuple[dict[str, int], float, bool]:
        """Apply an action and return (new_state, reward, done).

        Args:
            task_index:  Index into the tasks list.
            start_slot:  Proposed start slot for the task.

        Returns:
            new_state:   Updated state dict.
            reward:      Scalar reward for this step.
            done:        True when all tasks have been assigned.
        """
        if task_index < 0 or task_index >= len(self._tasks):
            return copy.copy(self._state), -1.0, False

        task = self._tasks[task_index]

        # Validate slot range
        end_slot = start_slot + task.duration
        if start_slot < task.earliest or end_slot > self.n_slots:
            return copy.copy(self._state), -5.0, False
        if task.latest is not None and start_slot > task.latest:
            return copy.copy(self._state), -5.0, False

        # Build the proposed entry
        proposed = ScheduleEntry(
            task_id=task.id,
            resource=task.resource,
            start_slot=start_slot,
            end_slot=end_slot,
        )

        # Count overlaps with already-scheduled tasks
        overlap_penalty = 0.0
        for other_task in self._tasks:
            if other_task.id == task.id:
                continue
            other_start = self._state.get(other_task.id, self.UNSCHEDULED)
            if other_start == self.UNSCHEDULED:
                continue
            other_entry = ScheduleEntry(
                task_id=other_task.id,
                resource=other_task.resource,
                start_slot=other_start,
                end_slot=other_start + other_task.duration,
            )
            if proposed.overlaps(other_entry):
                overlap_penalty -= 10.0

        # Reward for a valid (possibly overlapping) booking
        reward = float(task.priority) + overlap_penalty

        # Commit the assignment
        self._state[task.id] = start_slot

        # Check completion
        done = all(v != self.UNSCHEDULED for v in self._state.values())
        if done:
            reward += 5.0

        return copy.copy(self._state), reward, done

    def current_entries(self) -> list[ScheduleEntry]:
        """Return ScheduleEntry objects for all currently scheduled tasks."""
        entries = []
        for task in self._tasks:
            start = self._state.get(task.id, self.UNSCHEDULED)
            if start != self.UNSCHEDULED:
                entries.append(
                    ScheduleEntry(
                        task_id=task.id,
                        resource=task.resource,
                        start_slot=start,
                        end_slot=start + task.duration,
                    )
                )
        return entries

    def count_overlaps(self) -> int:
        """Return the total number of overlapping task pairs in current state."""
        entries = self.current_entries()
        overlaps = 0
        for i, a in enumerate(entries):
            for b in entries[i + 1:]:
                if a.overlaps(b):
                    overlaps += 1
        return overlaps

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def state(self) -> dict[str, int]:
        return copy.copy(self._state)


# ---------------------------------------------------------------------------
# Tabular Q-learning scheduler
# ---------------------------------------------------------------------------

class TabularQScheduler:
    """Lightweight tabular Q-learning agent for the SchedulerEnv.

    Suitable for use on edge devices (iPhone, Pixel, Mac Mini) where a full
    PufferLib / PyTorch stack is not available.  For larger action spaces or
    continuous state encodings, replace with PufferLib's vectorised trainer.

    Args:
        env:           The SchedulerEnv instance to train on.
        alpha:         Learning rate (default 0.1).
        gamma:         Discount factor (default 0.9).
        epsilon_start: Initial exploration rate (default 1.0).
        epsilon_min:   Minimum exploration rate (default 0.05).
        epsilon_decay: Multiplicative decay per episode (default 0.995).
        seed:          Random seed for reproducibility.
    """

    def __init__(
        self,
        env: SchedulerEnv,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng = random.Random(seed)

        # Q-table: maps (task_index, start_slot) → Q-value
        self._q: dict[tuple[int, int], float] = {}

        # Best schedule found so far
        self._best_state: dict[str, int] = {}
        self._best_reward: float = float("-inf")

    # ------------------------------------------------------------------
    def _q_val(self, task_idx: int, slot: int) -> float:
        return self._q.get((task_idx, slot), 0.0)

    def _greedy_action(self) -> tuple[int, int]:
        """Return the (task_index, slot) with the highest Q-value among
        unscheduled tasks; break ties randomly."""
        state = self.env.state()
        candidates: list[tuple[int, int]] = []
        for i, task in enumerate(self.env.tasks()):
            if state.get(task.id, SchedulerEnv.UNSCHEDULED) == SchedulerEnv.UNSCHEDULED:
                max_q = max(
                    (self._q_val(i, s) for s in range(task.earliest, self.env.n_slots - task.duration + 1)),
                    default=0.0,
                )
                best_slots = [
                    s for s in range(task.earliest, self.env.n_slots - task.duration + 1)
                    if abs(self._q_val(i, s) - max_q) < 1e-9
                ]
                if best_slots:
                    candidates.append((i, self._rng.choice(best_slots)))
        if not candidates:
            # All scheduled — pick any task/slot
            i = self._rng.randrange(self.env.n_tasks)
            task = self.env.tasks()[i]
            slot = self._rng.randint(task.earliest, max(task.earliest, self.env.n_slots - task.duration))
            return i, slot
        return self._rng.choice(candidates)

    def _random_action(self) -> tuple[int, int]:
        """Return a uniformly random (task_index, slot) for an unscheduled task."""
        state = self.env.state()
        unscheduled = [
            i for i, t in enumerate(self.env.tasks())
            if state.get(t.id, SchedulerEnv.UNSCHEDULED) == SchedulerEnv.UNSCHEDULED
        ]
        if not unscheduled:
            i = self._rng.randrange(self.env.n_tasks)
        else:
            i = self._rng.choice(unscheduled)
        task = self.env.tasks()[i]
        max_start = max(task.earliest, self.env.n_slots - task.duration)
        slot = self._rng.randint(task.earliest, max_start)
        return i, slot

    # ------------------------------------------------------------------

    def train(self, episodes: int = 500) -> None:
        """Run Q-learning for ``episodes`` episodes.

        Each episode resets the environment and greedily assigns tasks one
        by one (with ε-greedy exploration), accumulating rewards and updating
        the Q-table after each step.
        """
        for _ in range(episodes):
            self.env.reset()
            episode_reward = 0.0
            steps = 0
            max_steps = self.env.n_tasks * 2  # guard against infinite loops

            while steps < max_steps:
                state = self.env.state()
                unscheduled = [
                    i for i, t in enumerate(self.env.tasks())
                    if state.get(t.id, SchedulerEnv.UNSCHEDULED) == SchedulerEnv.UNSCHEDULED
                ]
                if not unscheduled:
                    break

                # ε-greedy action selection
                if self._rng.random() < self.epsilon:
                    task_idx, slot = self._random_action()
                else:
                    task_idx, slot = self._greedy_action()

                new_state, reward, done = self.env.step(task_idx, slot)
                episode_reward += reward

                # Q-update
                key = (task_idx, slot)
                current_q = self._q_val(task_idx, slot)

                # Max Q for next state: use best slot for any still-unscheduled task
                next_unscheduled = [
                    j for j, t in enumerate(self.env.tasks())
                    if new_state.get(t.id, SchedulerEnv.UNSCHEDULED) == SchedulerEnv.UNSCHEDULED
                ]
                if next_unscheduled:
                    next_max_q = max(
                        (
                            self._q_val(j, s)
                            for j in next_unscheduled
                            for s in range(
                                self.env.tasks()[j].earliest,
                                self.env.n_slots - self.env.tasks()[j].duration + 1,
                            )
                        ),
                        default=0.0,
                    )
                else:
                    next_max_q = 0.0

                self._q[key] = current_q + self.alpha * (
                    reward + self.gamma * next_max_q - current_q
                )

                steps += 1
                if done:
                    break

            # Track best
            if episode_reward > self._best_reward:
                self._best_reward = episode_reward
                self._best_state = self.env.state()

            # Decay exploration
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def best_schedule(self) -> dict[str, int]:
        """Return the best task → start_slot mapping found during training."""
        return dict(self._best_state)

    def explain(
        self,
        schedule: Optional[dict[str, int]] = None,
        slot_duration_hours: float = 1.0,
        epoch: Optional[str] = None,
    ) -> str:
        """Generate a human-readable dispatch schedule string.

        Args:
            schedule:            Task → start_slot dict (defaults to best).
            slot_duration_hours: How many wall-clock hours one slot represents.
            epoch:               ISO timestamp for slot 0 (e.g. "2026-04-09T08:00:00").

        Returns:
            A formatted, sorted schedule listing each task, resource, and
            time window in plain English.
        """
        from datetime import datetime, timedelta, timezone

        if schedule is None:
            schedule = self.best_schedule()

        task_map = {t.id: t for t in self.env.tasks()}
        lines: list[str] = ["=== BridgeTwin Dispatch Schedule ==="]

        base_dt: Optional[datetime] = None
        if epoch:
            try:
                base_dt = datetime.fromisoformat(epoch)
                if base_dt.tzinfo is None:
                    base_dt = base_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                base_dt = None

        sorted_items = sorted(
            schedule.items(), key=lambda kv: (kv[1] if kv[1] != SchedulerEnv.UNSCHEDULED else 9999)
        )
        for task_id, start_slot in sorted_items:
            task = task_map.get(task_id)
            if task is None:
                continue
            if start_slot == SchedulerEnv.UNSCHEDULED:
                lines.append(f"  [{task.priority}] {task_id}  →  UNSCHEDULED  (resource: {task.resource})")
                continue
            end_slot = start_slot + task.duration

            if base_dt is not None:
                start_dt = base_dt + timedelta(hours=start_slot * slot_duration_hours)
                end_dt = base_dt + timedelta(hours=end_slot * slot_duration_hours)
                time_str = f"{start_dt.strftime('%a %Y-%m-%d %H:%M')} → {end_dt.strftime('%H:%M')}"
            else:
                time_str = f"slot {start_slot}–{end_slot}"

            lines.append(
                f"  [{task.priority}] {task_id}  →  {time_str}  (resource: {task.resource})"
            )

        n_overlaps = self._count_overlaps_in(schedule)
        lines.append("")
        lines.append(f"Total tasks: {len(schedule)}  |  Overlaps: {n_overlaps}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _count_overlaps_in(self, schedule: dict[str, int]) -> int:
        task_map = {t.id: t for t in self.env.tasks()}
        entries: list[ScheduleEntry] = []
        for task_id, start in schedule.items():
            if start == SchedulerEnv.UNSCHEDULED:
                continue
            t = task_map.get(task_id)
            if t is None:
                continue
            entries.append(
                ScheduleEntry(
                    task_id=task_id,
                    resource=t.resource,
                    start_slot=start,
                    end_slot=start + t.duration,
                )
            )
        overlaps = 0
        for i, a in enumerate(entries):
            for b in entries[i + 1:]:
                if a.overlaps(b):
                    overlaps += 1
        return overlaps
