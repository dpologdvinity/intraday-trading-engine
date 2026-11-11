"""Append-only, hash-chained execution journals; hashes detect corruption, not forgery."""

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from pathlib import Path


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class EventLog:
    def __init__(self, path, *, clock=time.time_ns):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", buffering=1)
        self.clock = clock
        self.run_id = uuid.uuid4().hex
        self.sequence = 0
        self.previous = "0" * 64

    def write(self, event, *, event_ns=None, **values):
        if self.sequence % 1000 == 0 and shutil.disk_usage(self.path.parent).free < 5 * 1024**3:
            raise RuntimeError("journal disk guard: preserve at least 5GB free")
        row = json_value(
            dict(
                event=event,
                run_id=self.run_id,
                sequence=self.sequence,
                wall_ns=self.clock(),
                event_ns=event_ns,
                previous=self.previous,
                **values,
            )
        )
        row["hash"] = canonical_hash(row)
        self.file.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        if event in ("start", "order", "fill", "trade", "halt", "session", "finish"):
            self.file.flush()
            os.fsync(self.file.fileno())
        self.previous = row["hash"]
        self.sequence += 1
        return row

    def close(self):
        self.file.close()


def iter_log(path):
    count = 0
    previous = "0" * 64
    run_id = None
    with Path(path).open() as stream:
        for sequence, line in enumerate(stream):
            if not line.endswith("\n"):
                if not line.strip("\0"):
                    break  # zero-filled tail of an unflushed write: the machine stopped
                raise ValueError("truncated execution journal")
            try:
                row = json.loads(line)
                digest = row.pop("hash")
                if (
                    row["sequence"] != sequence
                    or row["previous"] != previous
                    or canonical_hash(row) != digest
                ):
                    raise ValueError("journal integrity mismatch")
                run_id = run_id or row["run_id"]
                if row["run_id"] != run_id:
                    raise ValueError("mixed journal runs")
                row["hash"] = digest
                count += 1
                yield row
                previous = digest
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid execution journal") from exc
    if not count:
        raise ValueError("journal lacks start record")


def read_log(path):
    rows = list(iter_log(path))
    if rows[0]["event"] != "start":
        raise ValueError("journal lacks start record")
    return rows
