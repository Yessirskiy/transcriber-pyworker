import itertools
import json
import os
import sys
import traceback
import urllib.request
import uuid

from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    LogActionConfig,
    BenchmarkConfig,
)

WORKER_VERSION = "2026-09-19-a"  # bump on each change you want to verify on an instance

print(f"===== transcriber-pyworker version {WORKER_VERSION} =====", flush=True)

# --- Model server -------------------------------------------------------------
# `main` (server.py) — the only process supervisor exposes on the host. See
# pyworker-integration.md for the readiness/log-file contract this depends on.

MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 8080
MODEL_LOG_FILE = "/var/log/app/main.log"
MODEL_HEALTHCHECK_ENDPOINT = "/health"

MODEL_LOAD_LOG_MSGS = ["Application startup complete."]
MODEL_ERROR_LOG_MSGS = ["STARTUP FAILED"]

REQUIRED_FIELDS = ("request_id", "get_audio_url", "put_result_url", "duration")


# --- Request handling ---------------------------------------------------------


def request_parser(request: dict) -> dict:
    # get_audio_url / put_result_url are pre-signed upstream (outside this
    # service's or PyWorker's control) — nothing to fetch or mint here, just
    # validate the shape and forward as-is.
    missing = [f for f in REQUIRED_FIELDS if request.get(f) is None]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")
    return request


def compute_workload(request: dict) -> float:
    duration = request.get("duration", 0)
    factor = (
        2 if (request.get("with_alignment") or request.get("with_diarization")) else 1
    )
    return duration * factor


# --- Benchmark data generation -------------------------------------------------
# Benchmark cases (duration, get_audio_url, put_result_url, with_alignment,
# with_diarization) are fetched from an external, pre-populated fixture list —
# each case is real and already reachable, so no local audio/receiver needed.
# Each case is used for exactly one pass, at concurrency 1.

BENCHMARK_API_URL = "https://transvox.ru/api/v1/benchmarks"


def _fetch_benchmark_cases() -> list:
    token = os.environ.get("BENCHMARK_TOKEN")
    if not token:
        raise RuntimeError(
            "BENCHMARK_TOKEN must be set to run the /transcribe benchmark"
        )
    req = urllib.request.Request(
        BENCHMARK_API_URL, headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req) as resp:
        cases = json.load(resp)
    if not cases:
        raise RuntimeError("Benchmark API returned no cases")
    return cases


_benchmark_cases = _fetch_benchmark_cases()
_benchmark_cases_iter = itertools.cycle(_benchmark_cases)


def transcribe_benchmark_generator() -> dict:
    try:
        return _build_benchmark_payload()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise


def _build_benchmark_payload() -> dict:
    case = next(_benchmark_cases_iter)
    return {
        "request_id": str(uuid.uuid4()),
        "get_audio_url": case["get_audio_url"],
        "put_result_url": case["put_result_url"],
        "duration": case["duration"],
        "with_diarization": case.get("with_diarization", False),
        "with_alignment": case.get("with_alignment", False),
        "language": None,
        "priority": 0,
        "num_speakers": None,
        "min_speakers": None,
        "max_speakers": None,
    }


# --- Worker configuration -------------------------------------------------------

worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        HandlerConfig(
            route="/transcribe",
            allow_parallel_requests=True,
            max_queue_time=1800.0,
            workload_calculator=compute_workload,
            request_parser=request_parser,
            benchmark_config=BenchmarkConfig(
                generator=transcribe_benchmark_generator,
                runs=1,
                concurrency=1,
                do_warmup=False,
            ),
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSGS,
        on_error=MODEL_ERROR_LOG_MSGS,
    ),
)

Worker(worker_config).run()
