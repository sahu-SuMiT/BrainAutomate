"""
src/automation.py  (hardened)
──────────────────────────────
Full error classification, circuit breaker, jitter, session renewal,
and suspicion-avoidance for the Brain API simulation queue.

Error taxonomy handled
──────────────────────
400 Bad Request          → expression syntax error — log and SKIP (no retry)
401 Unauthorized         → session expired — re-authenticate and retry once
403 Forbidden            → account issue — stop the entire queue and alert
404 Not Found            → wrong sim_id or endpoint — SKIP
409 Conflict             → duplicate submission — SKIP
422 Unprocessable        → invalid alpha expression semantics — SKIP
429 Too Many Requests    → rate limit — respect Retry-After + circuit breaker
500/502/503/504          → server-side transient — exponential backoff + retry

Suspicion avoidance
───────────────────
• Jitter on all delays  (±30% random noise so timings look human)
• Gradual ramp-up: first 5 sims of a session use a wider gap
• Random micro-pauses between poll cycles (2–7s instead of fixed 5s)
• Circuit breaker: if 5 consecutive failures → long pause before resuming
• Exponential backoff (not linear) on server errors: 30 → 60 → 120 → 240s
"""

import time
import queue
import random
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.logger import setup_logger
from src.result_logger import ResultLogger

logger = setup_logger(__name__)

TERMINAL_STATUSES = {"SUCCESS", "ERROR", "WARNING", "COMPLETE", "FAILED"}

# HTTP codes we SKIP without retry (expression-level or account-level problems)
SKIP_CODES = {400, 403, 404, 409, 422}

# HTTP codes we RETRY (transient server or rate-limit issues)
RETRY_CODES = {429, 500, 502, 503, 504}

DEFAULT_THRESHOLDS = {
    "min_sharpe":   1.25,
    "min_fitness":  1.00,
    "max_turnover": 0.70,
    "min_margin":   0.005,
}

# Circuit-breaker thresholds
CIRCUIT_BREAKER_LIMIT  = 5      # consecutive failures before tripping
CIRCUIT_BREAKER_PAUSE  = 300    # seconds to cool down (5 minutes)


def _jitter(base: float, pct: float = 0.30) -> float:
    """Add ±pct random noise to a delay to avoid robotic timing."""
    noise = base * pct * (2 * random.random() - 1)
    return max(1.0, base + noise)


class AlphaAutomation:
    """
    Hardened simulation queue with full error classification,
    circuit breaker, jitter, and suspicion-avoidance behavior.
    """

    def __init__(
        self,
        api_client,
        auto_submit: bool = False,
        thresholds: dict = None,
        optimizer=None,
        learner=None,
        guardian=None,
    ):
        self.api_client    = api_client
        self.auto_submit   = auto_submit
        self.thresholds    = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.optimizer     = optimizer
        self.learner       = learner
        self.guardian      = guardian
        self.task_queue    = queue.Queue()
        self.result_logger = ResultLogger()

        # Session-level state
        self._sims_this_session    = 0     # count for ramp-up logic
        self._ramp_up_limit        = random.randint(3, 8)  # randomized, not fixed at 5
        self._consecutive_failures = 0    # for circuit breaker
        self._circuit_open         = False
        self._circuit_lock         = threading.Lock()   # thread-safe circuit breaker
        self._feedback_lock        = threading.Lock()   # thread-safe learner + queue updates

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def add_task(self, alpha_data: dict):
        self.task_queue.put(alpha_data)
        logger.info(f"Task enqueued: {alpha_data.get('name', 'unnamed')}")

    def add_tasks(self, alphas: list):
        """Bulk-enqueue. Guardian filters first, learner ranks second."""
        if self.guardian is not None:
            alphas = self.guardian.pre_filter_batch(alphas)
        if self.learner is not None and len(alphas) > 1:
            alphas = self.learner.rank_candidates(alphas)
            logger.info("Learner ranked the candidate queue.")
        for alpha in alphas:
            self.add_task(alpha)

    # ------------------------------------------------------------------
    # Core runner
    # ------------------------------------------------------------------

    def run_with_backoff(self, max_retries: int = 5, base_delay: float = 30.0,
                         n_workers: int = 3):
        """
        Drain the queue using up to n_workers parallel threads.
        
        Architecture:
        - Each worker independently polls its own simulation (I/O-bound, no GIL issue)
        - Submissions are serialized by Guardian._submit_lock (one at a time)
        - Learner updates and queue mutations are serialized by _feedback_lock
        - Circuit breaker is protected by _circuit_lock
        """
        def worker():
            """One worker thread: pull alpha, submit, poll, process feedback."""
            while True:
                # Check circuit breaker before taking work
                with self._circuit_lock:
                    if self._circuit_open:
                        logger.warning(
                            f"Circuit breaker OPEN. Cooling down for {CIRCUIT_BREAKER_PAUSE}s…"
                        )
                        # Release lock during sleep so other threads aren't blocked
                        is_open = True
                    else:
                        is_open = False

                if is_open:
                    time.sleep(_jitter(CIRCUIT_BREAKER_PAUSE))
                    with self._circuit_lock:
                        self._circuit_open         = False
                        self._consecutive_failures = 0
                    logger.info("Circuit breaker reset. Resuming.")

                try:
                    alpha = self.task_queue.get(block=False)
                except queue.Empty:
                    break   # Queue drained — this worker is done

                name = alpha.get("name", "unnamed")

                # Human-like random skip: ~3% chance
                if random.random() < 0.03:
                    logger.info(f"[{name}] Randomly skipped (human-like behavior).")
                    self.task_queue.task_done()
                    continue

                success, sim_id, details, skip_reason = self._process_with_retry(
                    alpha, name, max_retries, base_delay
                )

                # ── Extract metrics ───────────────────────────────────
                metrics = {}
                alpha_summary = (details or {}).get("alpha", {}) if success else {}
                if alpha_summary:
                    metrics = {
                        "sharpe":   alpha_summary.get("sharpe"),
                        "fitness":  alpha_summary.get("fitness"),
                        "turnover": alpha_summary.get("turnover"),
                        "margin":   alpha_summary.get("margin"),
                    }
                elif details and "error" in details:
                    metrics = {"error_message": details.get("error")}

                tests      = (details or {}).get("checks", [])
                tests_pass = self._all_checks_pass(name, tests)
                quality_ok = self._meets_thresholds(name, metrics)

                submitted = False
                if success and tests_pass and quality_ok and self.auto_submit:
                    alpha_id = (details or {}).get("alpha", {}).get("id")
                    if alpha_id:
                        submitted = self._try_submit(name, alpha_id)

                if skip_reason:
                    status_label = f"SKIPPED:{skip_reason}"
                elif not success:
                    status_label = "FAILED"
                elif success and tests_pass and quality_ok:
                    status_label = "SUBMITTED" if submitted else "QUALIFIES"
                elif success and not tests_pass:
                    status_label = "TEST_FAIL"
                else:
                    status_label = "BELOW_THRESHOLD"

                logger.info(f"[{'\u2713' if success else '\u2717'}] {name} → {status_label}")

                # ── Circuit breaker ───────────────────────────────────
                with self._circuit_lock:
                    if status_label.startswith("FAILED") or status_label.startswith("SKIPPED"):
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= CIRCUIT_BREAKER_LIMIT:
                            self._circuit_open = True
                    else:
                        self._consecutive_failures = 0

                # ── Feedback: learner + optimizer (serialized) ────────
                sharpe  = metrics.get("sharpe")
                fitness = metrics.get("fitness")

                with self._feedback_lock:
                    if self.optimizer is not None:
                        op = alpha.get("_meta", {}).get("op")
                        if op:
                            self.optimizer.record_result(op, sharpe)

                        if status_label == "BELOW_THRESHOLD" and sharpe is not None:
                            for r in self.optimizer.refine_near_miss(alpha, sharpe):
                                self.task_queue.put(r)
                            for c in self.optimizer.hill_climb_settings(alpha, sharpe):
                                self.task_queue.put(c)

                        if sharpe is not None and sharpe <= -1.0 and fitness is not None and fitness < 0:
                            for neg in self.optimizer.flip_negative_alpha(alpha, sharpe):
                                self.task_queue.put(neg)

                        if success and tests:
                            passed_tests = [c for c in tests if c.get("result") == "PASS"]
                            if len(passed_tests) >= 3 and not (tests_pass and quality_ok):
                                self.optimizer.add_to_ensemble(alpha)
                                if len(self.optimizer._ensemble_pool) >= 5:
                                    for e in self.optimizer.generate_ensembles():
                                        self.task_queue.put(e)

                        self.optimizer.save_visited()

                    if self.learner is not None:
                        self.learner.record(alpha, sharpe, metrics.get("fitness"), metrics.get("turnover"))

                        q_size = self.task_queue.qsize()
                        n_done = self.learner._model.n_samples
                        if q_size > 1 and n_done > 0 and n_done % 20 == 0:
                            logger.info(f"Re-ranking {q_size} remaining queue items…")
                            remaining = []
                            while not self.task_queue.empty():
                                try:
                                    remaining.append(self.task_queue.get_nowait())
                                except queue.Empty:
                                    break
                            for item in self.learner.rank_candidates(remaining):
                                self.task_queue.put(item)

                    if self.guardian is not None:
                        self.guardian.post_result(alpha, status_label)

                metrics["tests_pass"]  = tests_pass
                metrics["auto_submit"] = submitted
                self.result_logger.record(
                    alpha_name=name,
                    sim_id=sim_id or "N/A",
                    status=status_label,
                    details=metrics,
                )

                self.task_queue.task_done()

        logger.info(f"Starting parallel run with {n_workers} workers, {self.task_queue.qsize()} queued alphas.")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(worker) for _ in range(n_workers)]
            for f in as_completed(futures):
                exc = f.exception()
                if exc:
                    logger.error(f"Worker thread crashed: {exc}", exc_info=exc)
        logger.info("All workers finished. Queue drained.")

    # ------------------------------------------------------------------
    # Test-case + threshold checks (unchanged from previous version)
    # ------------------------------------------------------------------

    def _all_checks_pass(self, name: str, checks: list) -> bool:
        if not checks:
            logger.warning(f"[{name}] No test cases returned by API.")
            return True
        failed   = [c for c in checks if c.get("result") == "FAIL"]
        warnings = [c for c in checks if c.get("result") == "WARN"]
        if failed:
            names = ", ".join(c.get("name", "?") for c in failed)
            logger.warning(f"[{name}] ✗ Failed checks: {names}")
            return False
        if warnings:
            names = ", ".join(c.get("name", "?") for c in warnings)
            logger.info(f"[{name}] ⚠ Warning checks: {names}")
        logger.info(f"[{name}] ✓ All {len(checks)} test cases passed.")
        return True

    def _meets_thresholds(self, name: str, metrics: dict) -> bool:
        s, f, t, m = (metrics.get(k) for k in ("sharpe", "fitness", "turnover", "margin"))
        if not all(v is not None for v in [s, f, t, m]):
            return False
        ok = True
        if s < self.thresholds["min_sharpe"]:
            logger.info(f"[{name}] Sharpe {s:.3f} below threshold"); ok = False
        if f < self.thresholds["min_fitness"]:
            logger.info(f"[{name}] Fitness {f:.3f} below threshold"); ok = False
        if t > self.thresholds["max_turnover"]:
            logger.info(f"[{name}] Turnover {t:.3f} too high"); ok = False
        if m < self.thresholds["min_margin"]:
            logger.info(f"[{name}] Margin {m:.5f} too low"); ok = False
        if ok:
            logger.info(f"[{name}] ✓ All thresholds met  sharpe={s:.3f}  fitness={f:.3f}")
        return ok

    def _try_submit(self, name: str, alpha_id: str) -> bool:
        try:
            self.api_client.submit_alpha(alpha_id)
            logger.info(f"[{name}] 🚀 Auto-submitted alpha {alpha_id}!")
            return True
        except Exception as e:
            logger.error(f"[{name}] Auto-submit failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Hardened submission + retry
    # ------------------------------------------------------------------

    def _process_with_retry(
        self,
        alpha: dict,
        name: str,
        max_retries: int,
        base_delay: float,
    ) -> tuple[bool, str | None, dict | None, str | None]:
        """
        Submit & poll one alpha with full error classification.

        Returns
        -------
        (success, sim_id, details, skip_reason)
        skip_reason is set for errors that should not be retried.
        """
        sim_id = None

        for attempt in range(1, max_retries + 1):
            try:
                # ── Pre-submit: rate limit + cooldown + jitter ─────────
                if self.guardian is not None:
                    if not self.guardian.pre_submit():
                        return False, sim_id, None, "QUOTA_EXHAUSTED"

                # Suspicion avoidance: ramp-up gap for first few sims
                # Use a randomized threshold (not always 5) to vary session signature
                if self._sims_this_session < self._ramp_up_limit:
                    ramp_sleep = _jitter(6.0)
                    logger.info(f"[{name}] Ramp-up pause: {ramp_sleep:.1f}s")
                    time.sleep(ramp_sleep)

                # ── Submit ─────────────────────────────────────────────
                sim_id = self.api_client.simulate_alpha(alpha)
                self._sims_this_session += 1

                if self.guardian is not None:
                    self.guardian.post_submit_success(alpha.get("regular", ""))

                # ── Poll ───────────────────────────────────────────────
                success, details = self._poll_until_done(sim_id, name)
                return success, sim_id, details, None

            except requests.exceptions.HTTPError as e:
                code     = e.response.status_code if e.response is not None else None
                body     = ""
                try:
                    body = e.response.text[:300] if e.response is not None else ""
                except Exception:
                    pass

                # ── Hard skips (expression/account problems) ───────────
                if code == 400:
                    logger.error(f"[{name}] 400 Bad Request — invalid expression syntax. "
                                 f"Skipping. Body: {body}")
                    return False, sim_id, None, "BAD_EXPRESSION"

                if code == 401:
                    logger.warning(f"[{name}] 401 Unauthorized — session expired. Re-authenticating…")
                    try:
                        self.api_client._authenticate()
                        logger.info(f"[{name}] Re-authenticated successfully. Retrying…")
                        continue   # retry the same attempt
                    except Exception as auth_err:
                        logger.error(f"[{name}] Re-auth failed: {auth_err}. Stopping queue.")
                        return False, sim_id, None, "AUTH_FAILED"

                if code == 403:
                    logger.error(f"[{name}] 403 Forbidden — account suspended or quota banned. "
                                 f"STOPPING QUEUE. Body: {body}")
                    # Drain the queue to stop all future submissions
                    while not self.task_queue.empty():
                        self.task_queue.get_nowait()
                    return False, sim_id, None, "ACCOUNT_FORBIDDEN"

                if code == 404:
                    logger.error(f"[{name}] 404 Not Found — endpoint or resource missing. Skipping.")
                    return False, sim_id, None, "NOT_FOUND"

                if code == 409:
                    logger.warning(f"[{name}] 409 Conflict — already submitted. Skipping.")
                    return False, sim_id, None, "DUPLICATE"

                if code == 422:
                    logger.error(f"[{name}] 422 Unprocessable — expression semantically invalid. "
                                 f"Skipping. Body: {body}")
                    return False, sim_id, None, "INVALID_EXPRESSION"

                # ── Retriable: rate limits and server errors ───────────
                if code == 429:
                    if self.guardian is not None:
                        self.guardian.post_submit_429()
                    self._exponential_backoff(name, attempt, max_retries, base_delay, e)

                elif code in RETRY_CODES:
                    logger.warning(f"[{name}] Server error {code}. Will retry.")
                    self._exponential_backoff(name, attempt, max_retries, base_delay, e)

                else:
                    logger.error(f"[{name}] Unexpected HTTP {code}: {body}. Skipping.")
                    return False, sim_id, None, f"HTTP_{code}"

            except requests.exceptions.Timeout as e:
                logger.warning(f"[{name}] Request timeout (attempt {attempt}/{max_retries}).")
                self._exponential_backoff(name, attempt, max_retries, base_delay, e)

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"[{name}] Connection error (attempt {attempt}/{max_retries}): {e}")
                self._exponential_backoff(name, attempt, max_retries, base_delay, e)

            except requests.exceptions.RequestException as e:
                logger.warning(f"[{name}] Request error (attempt {attempt}/{max_retries}): {e}")
                self._exponential_backoff(name, attempt, max_retries, base_delay, e)

        logger.error(f"[{name}] Exhausted {max_retries} retries.")
        return False, sim_id, None, "MAX_RETRIES"

    def _poll_until_done(self, sim_id: str, name: str) -> tuple[bool, dict]:
        """
        Poll with jittered intervals to avoid robotic fixed-5s pattern.
        """
        logger.info(f"[{name}] Polling simulation {sim_id}…")
        poll_errors = 0

        while True:
            try:
                data     = self.api_client.get_simulation_status(sim_id)
                progress = data.get("progress", "")
                status   = data.get("status", "UNKNOWN")
                poll_errors = 0  # reset on success

                logger.info(
                    f"[{name}] {status}"
                    f"{f' ({progress * 100:.1f}%)' if isinstance(progress, float) else ''}"
                )

                if status in TERMINAL_STATUSES:
                    # If finished successfully, Brain returns "COMPLETE" and an alpha ID string
                    if status == "COMPLETE":
                        alpha_id = data.get("alpha")
                        if isinstance(alpha_id, str):
                            try:
                                alpha_data = self.api_client.session.get(
                                    f"{self.api_client.base_url}/alphas/{alpha_id}",
                                    timeout=20,
                                ).json()
                                # The stats are under the "is" (In-Sample) key
                                data["alpha"] = alpha_data.get("is", {})
                                data["alpha"]["id"] = alpha_id
                                data["checks"] = data["alpha"].get("checks", [])
                            except Exception as e:
                                logger.error(f"[{name}] Failed to fetch alpha details for {alpha_id}: {e}")
                    
                    return status in {"SUCCESS", "WARNING", "COMPLETE"}, data

            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response is not None else None
                poll_errors += 1
                if code == 401:
                    logger.warning(f"[{name}] Poll 401 — re-authenticating…")
                    try:
                        self.api_client._authenticate()
                    except Exception:
                        pass
                elif poll_errors >= 3:
                    logger.error(f"[{name}] Poll failed {poll_errors} times — abandoning sim {sim_id}.")
                    return False, {}

            except requests.exceptions.RequestException as e:
                poll_errors += 1
                logger.warning(f"[{name}] Poll error #{poll_errors}: {e}")
                if poll_errors >= 3:
                    logger.error(f"[{name}] Abandoning poll after 3 network failures.")
                    return False, {}

            # Jittered poll interval (3–9s instead of fixed 5s)
            sleep_t = _jitter(5.0, pct=0.40)
            time.sleep(sleep_t)

    def _exponential_backoff(
        self,
        name: str,
        attempt: int,
        max_retries: int,
        base_delay: float,
        error,
    ):
        """
        True exponential backoff with jitter: 30 → 60 → 120 → 240s
        Respects Retry-After header. Caps at 300s.
        """
        if attempt >= max_retries:
            logger.error(f"[{name}] Max retries ({max_retries}) reached.")
            return

        # Exponential: 30 * 2^(attempt-1)
        delay = min(base_delay * (2 ** (attempt - 1)), 300.0)

        # Honour Retry-After header if present
        if isinstance(error, requests.exceptions.HTTPError) and error.response is not None:
            retry_after = error.response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                delay = max(delay, float(retry_after))

        # Add jitter so all concurrent sessions don't retry simultaneously
        delay = _jitter(delay, pct=0.20)

        logger.warning(
            f"[{name}] Attempt {attempt}/{max_retries} failed — "
            f"backing off {delay:.0f}s (exp). Error: {error}"
        )
        time.sleep(delay)
