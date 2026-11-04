"""Background jobs of the camera setup (Phase 2): segmentation and calibration take seconds to minutes on a CPU, so the
API answers at once with a job id and the page polls GET /api/setup/jobs/{id}.

In memory, the last KEEP jobs: a restart of the API forgets them (the Admin starts the step again; a finished
calibration is already a draft in the geometry store). One running job per camera and kind."""
from __future__ import annotations

import asyncio
import secrets
import time

KEEP = 50
STATES = ("queued", "running", "done", "error")


class JobBoard:
    def __init__(self, keep: int = KEEP):
        self.keep, self.jobs = keep, {}

    def get(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    def running(self, kind: str, cam: str) -> dict | None:
        return next((j for j in self.jobs.values() if j["kind"] == kind and j["cam"] == cam
                     and j["state"] in ("queued", "running")), None)

    def start(self, kind: str, cam: str, work) -> dict:
        """Run `work(report)` (an async function; report(stage, progress 0-1) updates the job) in the background ->
        the job: {"id", "kind", "cam", "state", "stage", "progress", "result", "error", "created", "updated"}. A
        failure ends in state "error" with its message (never a traceback with paths or URLs): ValueError messages
        are the Admin's to read, anything else is reported by type."""
        now = time.time()
        job = {"id": secrets.token_hex(8), "kind": kind, "cam": cam, "state": "queued", "stage": "queued",
               "progress": 0.0, "result": None, "error": None, "created": now, "updated": now}
        self.jobs[job["id"]] = job
        self._prune()

        def report(stage: str, progress: float):
            job.update(stage=stage, progress=round(min(1.0, max(0.0, progress)), 3), updated=time.time())

        async def run():
            job.update(state="running", updated=time.time())
            try:
                job["result"] = await work(report)
                job.update(state="done", stage="done", progress=1.0)
            except asyncio.CancelledError:
                job.update(state="error", error="cancelled")
                raise
            except ValueError as e:
                job.update(state="error", error=str(e))
            except Exception as e:  # noqa: BLE001 - the page shows the type; the details stay in the API log
                job.update(state="error", error=f"{type(e).__name__}: {e}"[:300])
            job["updated"] = time.time()
            self._prune()

        task = asyncio.ensure_future(run())
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        job["task"] = task
        return job

    def _prune(self):
        """Forget the oldest finished jobs beyond `keep` (running ones are never dropped)."""
        for old in sorted(self.jobs, key=lambda k: self.jobs[k]["created"])[:max(0, len(self.jobs) - self.keep)]:
            if self.jobs[old]["state"] in ("done", "error"):
                self.jobs.pop(old)

    @staticmethod
    def view(job: dict) -> dict:
        """The job as the API returns it (without the asyncio task)."""
        return {k: v for k, v in job.items() if k != "task"}
