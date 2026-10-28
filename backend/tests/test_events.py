from __future__ import annotations

import asyncio
import threading

from framelab.events import EventBus
from framelab.schemas import Event


def test_emit_appends_with_monotonic_seq(bus: EventBus) -> None:
    for i in range(3):
        bus.emit("r_x", "progress", "ingesting", message=f"step {i}")
    events = list(bus.iter_events("r_x"))
    assert [e["seq"] for e in events] == [1, 2, 3]
    assert events[0]["message"] == "step 0"
    Event.model_validate(events[0])  # flat schema with extra keys


def test_seq_continues_after_restart(bus: EventBus) -> None:
    bus.emit("r_x", "progress", "ingesting")
    bus.emit("r_x", "progress", "ingesting")
    fresh = EventBus(bus.runs_dir)
    event = fresh.emit("r_x", "done", "done")
    assert event["seq"] == 3


def test_iter_events_after_seq(bus: EventBus) -> None:
    for _ in range(5):
        bus.emit("r_x", "progress", "s")
    assert [e["seq"] for e in bus.iter_events("r_x", after_seq=3)] == [4, 5]


def test_threaded_emit_is_safe(bus: EventBus) -> None:
    def worker() -> None:
        for _ in range(50):
            bus.emit("r_t", "progress", "s")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seqs = [e["seq"] for e in bus.iter_events("r_t")]
    assert sorted(seqs) == list(range(1, 201))


async def test_subscribe_replays_then_streams_until_done(bus: EventBus) -> None:
    bus.emit("r_s", "stage_started", "ingesting")

    async def produce() -> None:
        await asyncio.sleep(0.05)
        bus.emit("r_s", "progress", "ingesting", message="halfway")
        await asyncio.to_thread(bus.emit, "r_s", "done", "done", verdict="PASS")

    producer = asyncio.create_task(produce())
    received = [e async for e in bus.subscribe("r_s")]
    await producer
    assert [e["type"] for e in received] == ["stage_started", "progress", "done"]
    assert [e["seq"] for e in received] == [1, 2, 3]


async def test_subscribe_with_last_event_id(bus: EventBus) -> None:
    bus.emit("r_l", "progress", "s")
    bus.emit("r_l", "progress", "s")
    bus.emit("r_l", "error", "s", message="boom")
    received = [e async for e in bus.subscribe("r_l", after_seq=2)]
    assert [e["seq"] for e in received] == [3]
