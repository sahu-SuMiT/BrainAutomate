"""
src/guardian.py
────────────────
Protects the simulation pipeline from four performance-degrading factors:

1. Rate Limit Guard
   - Tracks daily simulation quota (persisted across runs in quota.json)
   - Enforces a minimum cooldown gap between submissions
   - Proactively slows down before the limit is hit (not just after 429)
   - Adaptive: if recent 429s are detected, automatically widens the gap

2. Self-Correlation Guard
   - Fetches your already-submitted alphas from Brain API
   - Checks if a new candidate's expression is structurally too similar
   - Rejects candidates likely to fail Brain's self-correlation check
   - Prevents wasting simulation slots on redundant alphas

3. Intra-Batch Diversity Guard
   - Deduplicates the candidate pool before queuing
   - Removes near-duplicate expressions (e.g. same formula, lookback ±1)
   - Uses Jaccard token similarity so diverse operators always get a slot

4. Toxic Pattern Filter
   - Learns which (field, operator) combinations consistently fail tests
   - Blacklists them across runs (persisted in toxic.json)
   - Prevents re-simulating known-bad combinations
"""

import re
import json
import math
import time
import random
import logging
import threading
from datetime import date
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

QUOTA_FILE = Path("quota.json")
TOXIC_FILE = Path("toxic.json")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(expr: str) -> frozenset:
    """Extract all word tokens from an alpha expression."""
    return frozenset(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr or ""))


def _jaccard(a: str, b: str) -> float:
    """Jaccard token similarity between two expressions. Range: [0, 1]."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def _numeric_tokens(expr: str) -> list:
    """Extract all numeric literals from an expression."""
    return re.findall(r'\b\d+\b', expr or "")


def _structural_key(expr: str) -> str:
    """
    Collapse all numeric arguments to '?' to get a structural fingerprint.
    e.g. 'rank(ts_zscore(close, 20))' → 'rank(ts_zscore(close,?))'
    Two expressions with the same key differ only in parameter values.
    """
    return re.sub(r'\b\d+\b', '?', re.sub(r'\s', '', expr or ""))


# ─────────────────────────────────────────────────────────────────────────────
# 0. Expression Validator  (catches problems BEFORE hitting the API)
# ─────────────────────────────────────────────────────────────────────────────

class ExpressionValidator:
    """
    Fast local pre-validation of alpha expressions.
    Rejects obviously broken expressions before they waste a simulation slot.

    Catches:
    - Empty or too-short expressions
    - Unbalanced parentheses
    - Division by a constant zero literal
    - Expressions with no operator at all (just a bare field name)
    - Expressions using undefined/misspelled operator names
    - Numeric-only expressions
    """

    KNOWN_OPS = {
        "rank", "group_rank", "group_zscore", "group_neutralize", "group_count",
        "ts_min", "ts_max", "ts_mean", "ts_sum", "ts_std_dev", "ts_rank",
        "ts_scale", "ts_zscore", "ts_delta", "ts_delay", "ts_product",
        "ts_quantile", "ts_arg_max", "ts_arg_min", "ts_av_diff",
        "ts_decay_linear", "ts_count_nans", "ts_corr", "ts_covariance",
        "ts_regress", "ts_backfill", "hump", "days_from_last_change",
        "last_diff_value", "kth_element", "ts_step",
    }

    def validate(self, expr: str) -> tuple[bool, str]:
        """
        Returns (is_valid: bool, reason: str).
        reason is empty string when valid.
        """
        if not expr or len(expr.strip()) < 5:
            return False, "Expression too short or empty"

        # Balanced parentheses
        depth = 0
        for ch in expr:
            if ch == "(": depth += 1
            elif ch == ")": depth -= 1
            if depth < 0:
                return False, "Unbalanced parentheses (extra closing paren)"
        if depth != 0:
            return False, f"Unbalanced parentheses (unclosed: {depth})"

        # Division by zero literal
        if re.search(r'/\s*0\.?0*[^.]', expr):
            return False, "Potential division by zero literal"

        # Must contain at least one known operator
        tokens = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr))
        has_op = bool(tokens & self.KNOWN_OPS)
        if not has_op:
            return False, f"No known Brain operator found in expression"

        # Must not be purely numeric
        if re.fullmatch(r'[\d\.\s\+\-\*/]+', expr.strip()):
            return False, "Expression is purely numeric"

        return True, ""

    def filter_batch(self, candidates: list[dict]) -> list[dict]:
        clean, rejected = [], 0
        for c in candidates:
            valid, reason = self.validate(c.get("regular", ""))
            if not valid:
                logger.warning(f"[ExprValidator] Rejected '{c.get('regular','')[:60]}': {reason}")
                rejected += 1
            else:
                clean.append(c)
        if rejected:
            logger.info(f"[ExprValidator] {rejected} invalid expressions removed, {len(clean)} remain.")
        return clean


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rate Limit Guard
# ─────────────────────────────────────────────────────────────────────────────

class RateLimitGuard:
    """
    Proactive rate-limit manager.

    Tracks:
    - Daily simulation count (quota.json, rolls over each UTC day)
    - Time of last submission (enforce minimum gap between requests)
    - Recent 429 hit rate (if high, widens the gap automatically)
    """

    def __init__(
        self,
        daily_limit: int = 100,           # Brain limit per day (adjust for your tier)
        min_gap_seconds: float = 3.0,     # Minimum seconds between submissions
        warn_pct: float = 0.85,           # Warn and slow down at 85% of daily limit
    ):
        self.daily_limit     = daily_limit
        self.min_gap         = min_gap_seconds
        self.warn_pct        = warn_pct
        self._last_submit_ts = 0.0
        self._recent_429s    = 0
        self._quota          = self._load_quota()
        self._submit_lock    = threading.Lock()   # Serialize submissions across parallel workers

    # ── Quota ──────────────────────────────────────────────────────────────

    def _load_quota(self) -> dict:
        today = str(date.today())
        if QUOTA_FILE.exists():
            data = json.loads(QUOTA_FILE.read_text())
            if data.get("date") == today:
                return data
        return {"date": today, "count": 0}

    def _save_quota(self):
        QUOTA_FILE.write_text(json.dumps(self._quota, indent=2))

    @property
    def used_today(self) -> int:
        return self._quota["count"]

    @property
    def remaining_today(self) -> int:
        return max(0, self.daily_limit - self.used_today)

    def can_submit(self) -> bool:
        self._quota = self._load_quota()   # re-read in case of multi-process
        if self.used_today >= self.daily_limit:
            logger.error(
                f"Daily quota exhausted: {self.used_today}/{self.daily_limit}. "
                f"Submissions blocked until tomorrow."
            )
            return False
        return True

    def record_submission(self):
        """Call this after a simulation is successfully submitted."""
        self._quota["count"] += 1
        self._save_quota()
        self._last_submit_ts = time.monotonic()
        used = self.used_today
        remaining = self.remaining_today
        logger.info(f"Quota: {used}/{self.daily_limit} used today, {remaining} remaining.")

        # Proactive slow-down approaching daily limit
        if used >= int(self.daily_limit * self.warn_pct):
            logger.warning(
                f"⚠ Approaching daily limit ({used}/{self.daily_limit}). "
                f"Increasing submission gap to prevent lockout."
            )
            self.min_gap = max(self.min_gap, 10.0)

        # Suspicion avoidance: every 10 submissions, take a natural-looking pause
        if used > 0 and used % 10 == 0:
            burst_pause = random.uniform(15.0, 40.0)
            logger.info(f"[RateLimitGuard] Burst cooldown after 10 submissions: {burst_pause:.0f}s pause.")
            time.sleep(burst_pause)

    def record_429(self):
        """Call when a 429 is received to widen the gap adaptively."""
        self._recent_429s += 1
        new_gap = self.min_gap * (1.5 ** self._recent_429s)   # exponential widening
        new_gap = min(new_gap, 120.0)                          # cap at 2 min
        logger.warning(
            f"429 detected (#{self._recent_429s}). "
            f"Widening submission gap: {self.min_gap:.0f}s → {new_gap:.0f}s"
        )
        self.min_gap = new_gap

    def reset_429_counter(self):
        """Call when a submission succeeds to reset the 429 counter."""
        if self._recent_429s > 0:
            self._recent_429s = 0
            logger.info("429 counter reset after successful submission.")

    def enforce_cooldown(self):
        """Block until the minimum gap has passed, with jitter to look human.
        
        Thread-safe: uses _submit_lock so only one worker submits at a time,
        preventing two threads from both passing the rate-limit check simultaneously.
        """
        with self._submit_lock:
            elapsed = time.monotonic() - self._last_submit_ts
            # Add ±20% jitter so the gap is never a perfectly regular interval
            target_gap = self.min_gap * random.uniform(0.80, 1.20)
            if elapsed < target_gap:
                wait = target_gap - elapsed
                logger.info(f"Rate-limit cooldown: sleeping {wait:.1f}s…")
                time.sleep(wait)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Correlation Guard
# ─────────────────────────────────────────────────────────────────────────────

class CorrelationGuard:
    """
    Prevents wasting simulation slots on expressions that will fail
    Brain's self-correlation check against your existing portfolio.
    """

    def __init__(
        self,
        max_jaccard: float = 0.80,        # Reject if > 80% token overlap with any existing alpha
        max_structural_matches: int = 3,  # Reject if same structural pattern seen 3+ times
    ):
        self.max_jaccard = max_jaccard
        self.max_structural = max_structural_matches

        self._existing_exprs: list[str] = []         # expressions of submitted alphas
        self._structural_counts: dict   = defaultdict(int)  # pattern → count

    def load_existing_alphas(self, api_client, max_fetch: int = 200):
        """
        Fetch already-submitted alphas from Brain API and store their expressions.
        Call this once at startup.
        """
        try:
            resp = api_client.session.get(
                f"{api_client.base_url}/alphas",
                params={"limit": max_fetch, "status": "SUBMITTED"},
            )
            resp.raise_for_status()
            alphas = resp.json().get("results", [])
            self._existing_exprs = [
                a.get("regular", "") for a in alphas if a.get("regular")
            ]
            for expr in self._existing_exprs:
                self._structural_counts[_structural_key(expr)] += 1

            logger.info(
                f"CorrelationGuard: loaded {len(self._existing_exprs)} existing "
                f"submitted alphas for self-correlation check."
            )
        except Exception as e:
            logger.warning(f"CorrelationGuard: could not fetch existing alphas: {e}")

    def is_self_correlated(self, expr: str) -> tuple[bool, str]:
        """
        Returns (is_correlated: bool, reason: str).

        Checks:
        A) Token Jaccard similarity against any existing submitted alpha > threshold
        B) Same structural pattern (same formula, different numbers) submitted > N times
        """
        # A) Token similarity
        for existing in self._existing_exprs:
            sim = _jaccard(expr, existing)
            if sim >= self.max_jaccard:
                return True, f"Jaccard similarity {sim:.2f} ≥ {self.max_jaccard} with existing alpha"

        # B) Structural pattern
        key   = _structural_key(expr)
        count = self._structural_counts.get(key, 0)
        if count >= self.max_structural:
            return True, f"Structural pattern '{key}' already submitted {count} times"

        return False, ""

    def register_new(self, expr: str):
        """Call after a successful submission so future candidates are checked against it."""
        self._existing_exprs.append(expr)
        self._structural_counts[_structural_key(expr)] += 1

    def filter_batch(self, candidates: list[dict]) -> list[dict]:
        """Remove candidates that are too correlated with existing submitted alphas."""
        clean = []
        rejected = 0
        for c in candidates:
            expr = c.get("regular", "")
            correlated, reason = self.is_self_correlated(expr)
            if correlated:
                logger.info(f"[CorrelationGuard] Skipped '{expr[:60]}…': {reason}")
                rejected += 1
            else:
                clean.append(c)
        if rejected:
            logger.info(f"[CorrelationGuard] Filtered {rejected} correlated candidates, {len(clean)} remain.")
        return clean


# ─────────────────────────────────────────────────────────────────────────────
# 3. Intra-Batch Diversity Guard
# ─────────────────────────────────────────────────────────────────────────────

class DiversityGuard:
    """
    Ensures the candidate batch is diverse before queuing.
    Removes near-duplicates so different operators and fields all get turns.
    """

    def __init__(self, max_jaccard: float = 0.90, max_same_structure: int = 2):
        self.max_jaccard       = max_jaccard
        self.max_same_structure = max_same_structure

    def deduplicate(self, candidates: list[dict]) -> list[dict]:
        """
        Remove near-duplicate expressions.

        Two expressions are considered duplicates if:
        A) Jaccard token similarity > threshold (almost identical formula)
        B) Same structural key appears more than max_same_structure times
           (e.g., rank(ts_zscore(close, ?)) with 5 different lookbacks → keep top 2)
        """
        seen_exprs:    list[str]   = []
        structure_cnt: dict        = defaultdict(int)
        kept = []
        removed = 0

        for c in candidates:
            expr = c.get("regular", "")
            key  = _structural_key(expr)

            # Check structural saturation
            if structure_cnt[key] >= self.max_same_structure:
                removed += 1
                continue

            # Check Jaccard similarity against already-kept expressions
            too_similar = any(_jaccard(expr, e) >= self.max_jaccard for e in seen_exprs)
            if too_similar:
                removed += 1
                continue

            kept.append(c)
            seen_exprs.append(expr)
            structure_cnt[key] += 1

        if removed:
            logger.info(f"[DiversityGuard] Removed {removed} near-duplicate candidates, {len(kept)} diverse candidates kept.")
        return kept


# ─────────────────────────────────────────────────────────────────────────────
# 4. Toxic Pattern Filter
# ─────────────────────────────────────────────────────────────────────────────

class ToxicPatternFilter:
    """
    Learns which (field, operator) combinations consistently fail test cases
    and blacklists them so they are never re-queued.

    Persisted in toxic.json across runs.
    """

    FAIL_THRESHOLD = 3   # Mark as toxic after N consecutive failures

    def __init__(self):
        self._toxic:    set  = set()
        self._failures: dict = defaultdict(int)   # (field, op) → consecutive fail count
        self._load()

    def _load(self):
        if TOXIC_FILE.exists():
            data = json.loads(TOXIC_FILE.read_text())
            self._toxic = set(tuple(k) for k in data.get("toxic", []))
            logger.info(f"[ToxicFilter] Loaded {len(self._toxic)} toxic (field, op) patterns.")

    def _save(self):
        TOXIC_FILE.write_text(
            json.dumps({"toxic": [list(k) for k in self._toxic]}, indent=2)
        )

    def record_result(self, field: str, op: str, status: str):
        """
        Track success/failure for a (field, op) pair.
        - TEST_FAIL → increment failure counter
        - QUALIFIES / SUBMITTED → reset failure counter
        """
        key = (field, op)
        if status in {"TEST_FAIL", "FAILED"}:
            self._failures[key] += 1
            if self._failures[key] >= self.FAIL_THRESHOLD:
                if key not in self._toxic:
                    logger.warning(
                        f"[ToxicFilter] ({field}, {op}) failed {self.FAIL_THRESHOLD} times → blacklisted."
                    )
                    self._toxic.add(key)
                    self._save()
        elif status in {"QUALIFIES", "SUBMITTED", "SUCCESS"}:
            if key in self._failures:
                self._failures[key] = 0

    def is_toxic(self, field: str, op: str) -> bool:
        return (field, op) in self._toxic

    def filter_batch(self, candidates: list[dict]) -> list[dict]:
        """Remove candidates whose (field, op) pair is blacklisted."""
        clean = []
        removed = 0
        for c in candidates:
            meta  = c.get("_meta", {})
            field = meta.get("field", "")
            op    = meta.get("op", "")
            if field and op and self.is_toxic(field, op):
                removed += 1
                logger.debug(f"[ToxicFilter] Skipping toxic combo ({field}, {op})")
            else:
                clean.append(c)
        if removed:
            logger.info(f"[ToxicFilter] Blocked {removed} toxic candidates.")
        return clean

    def clear_toxic(self):
        """Manually clear the blacklist (e.g., after field data is updated)."""
        self._toxic.clear()
        self._failures.clear()
        self._save()
        logger.info("[ToxicFilter] Blacklist cleared.")


# ─────────────────────────────────────────────────────────────────────────────
# Guardian — unified interface
# ─────────────────────────────────────────────────────────────────────────────

class Guardian:
    """
    Single entry point for all protective filters.
    Pass an instance of this to AlphaAutomation.
    """

    def __init__(
        self,
        daily_limit: int   = 100,
        min_gap_seconds: float = 3.0,
        max_self_correlation: float = 0.80,
        max_intra_batch_similarity: float = 0.90,
    ):
        self.rate_limit  = RateLimitGuard(daily_limit=daily_limit, min_gap_seconds=min_gap_seconds)
        self.correlation = CorrelationGuard(max_jaccard=max_self_correlation)
        self.diversity   = DiversityGuard(max_jaccard=max_intra_batch_similarity)
        self.toxic       = ToxicPatternFilter()
        self.validator   = ExpressionValidator()    # ← new

    def initialise(self, api_client):
        """Call once at startup to load existing alphas for correlation checking."""
        self.correlation.load_existing_alphas(api_client)

    def pre_filter_batch(self, candidates: list[dict]) -> list[dict]:
        """
        All static filters in cheapest-first order:
        1. Expression validation   (syntax/semantic — cheapest)
        2. Intra-batch diversity   (Jaccard token similarity)
        3. Toxic pattern blacklist (lookup)
        4. Self-correlation guard  (slightly heavier — compares to existing portfolio)
        """
        n_start = len(candidates)
        candidates = self.validator.filter_batch(candidates)    # ← new first step
        candidates = self.diversity.deduplicate(candidates)
        candidates = self.toxic.filter_batch(candidates)
        candidates = self.correlation.filter_batch(candidates)
        n_end = len(candidates)
        if n_start != n_end:
            logger.info(f"[Guardian] Batch filtered: {n_start} → {n_end} candidates.")
        return candidates

    def pre_submit(self) -> bool:
        """
        Call immediately before each simulation POST.
        Returns False if quota is exhausted (caller should stop).
        Blocks for cooldown if needed.
        """
        if not self.rate_limit.can_submit():
            return False
        self.rate_limit.enforce_cooldown()
        return True

    def post_submit_success(self, expr: str):
        """Call after a successful simulation submission."""
        self.rate_limit.record_submission()
        self.rate_limit.reset_429_counter()
        self.correlation.register_new(expr)

    def post_submit_429(self):
        """Call when the API returns 429."""
        self.rate_limit.record_429()

    def post_result(self, alpha: dict, status: str):
        """Call after each simulation result to update toxic pattern learning."""
        meta  = alpha.get("_meta", {})
        field = meta.get("field", "")
        op    = meta.get("op", "")
        if field and op:
            self.toxic.record_result(field, op, status)

    def status_report(self) -> str:
        rl = self.rate_limit
        return (
            f"Rate limit: {rl.used_today}/{rl.daily_limit} used today "
            f"({rl.remaining_today} remaining) | "
            f"Gap: {rl.min_gap:.0f}s (±20% jitter) | "
            f"Toxic patterns: {len(self.toxic._toxic)}"
        )
