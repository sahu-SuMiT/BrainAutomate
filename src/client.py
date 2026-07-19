import random
import requests
import logging
from config import BASE_URL

logger = logging.getLogger(__name__)

# Rotate between realistic browser User-Agents to avoid bot fingerprinting
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
]


class BrainAPIClient:
    """
    Thin wrapper around the WorldQuant Brain REST API.

    Authentication uses HTTP Basic Auth (username + password).
    Every method raises requests.exceptions.RequestException on
    HTTP errors (including 429 Too Many Requests) so the caller's
    retry/back-off logic can handle them uniformly.
    """

    def __init__(self, credentials: dict):
        self.session = requests.Session()
        self.session.auth = (credentials["username"], credentials["password"])
        self.base_url = BASE_URL

        # Masquerade as a real browser — pick a random UA on each session start
        ua = random.choice(_USER_AGENTS)
        self.session.headers.update({
            "User-Agent": ua,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Origin": "https://platform.worldquantbrain.com",
            "Referer": "https://platform.worldquantbrain.com/",
        })
        logger.info(f"Session initialized with User-Agent: {ua[:50]}…")
        self._authenticate()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _authenticate(self):
        """Verify credentials by hitting the /authentication endpoint."""
        resp = self.session.post(f"{self.base_url}/authentication", timeout=20)
        resp.raise_for_status()
        logger.info("Successfully authenticated with WorldQuant Brain API.")

    # ------------------------------------------------------------------
    # Alpha submission
    # ------------------------------------------------------------------

    def simulate_alpha(self, alpha_data: dict) -> str:
        """
        Submit a simulation request and return the simulation ID.

        Parameters
        ----------
        alpha_data : dict
            Payload expected by the Brain API, e.g.
            {
                "type":       "REGULAR",
                "settings":   { "instrumentType": "EQUITY", ... },
                "regular":    "rank(close)"
            }

        Returns
        -------
        str
            The simulation ID returned by the platform.
        """
        # Strip metadata fields like "name" from payload before sending
        payload = {k: v for k, v in alpha_data.items() if k in {"type", "settings", "regular"}}
        resp = self.session.post(f"{self.base_url}/simulations", json=payload, timeout=20)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error details: {resp.text}")
            raise e
        sim_id = resp.headers.get("Location", "").split("/")[-1]
        logger.info(f"Simulation submitted, ID: {sim_id}")
        return sim_id

    # ------------------------------------------------------------------
    # Status polling
    # ------------------------------------------------------------------

    def get_simulation_status(self, sim_id: str) -> dict:
        """
        Return the full status JSON for a simulation.

        The 'status' field will be one of: 'PENDING', 'RUNNING',
        'SUCCESS', 'ERROR', 'WARNING'.
        """
        resp = self.session.get(f"{self.base_url}/simulations/{sim_id}", timeout=20)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Alpha submission (final)
    # ------------------------------------------------------------------

    def submit_alpha(self, alpha_id: str) -> dict:
        """Submit a successfully simulated alpha for review."""
        resp = self.session.post(f"{self.base_url}/alphas/{alpha_id}/submit", timeout=20)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Data field discovery  (composite scoring)
    # ------------------------------------------------------------------

    def fetch_top_fields(
        self,
        instrument_type: str = "EQUITY",
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        category: str = None,
        min_coverage: float = 0.5,
        top_n: int = 50,
        learner_stats: dict = None,   # optional: {field_id -> avg_sharpe} from Learner
    ) -> list[dict]:
        """
        Fetch available data fields and rank them using a COMPOSITE SCORE
        that considers three signals:

        1. Coverage quality  (Brain data — refreshed on every call)
           Higher coverage = more stocks have this data = more reliable signal.
           Weight: 0.3

        2. Market novelty   (Brain data — refreshed on every call)
           Brain's `alphas` count tells you how crowded a field is.
           Too high (>5000) → oversaturated, hard to avoid self-correlation.
           Too low (<10)    → untested, risky.
           Sweet spot: moderate alpha count gets the highest score.
           Weight: 0.3

        3. Personal success rate  (our local Learner data)
           Fields where YOUR past simulations got higher Sharpe are ranked up.
           This makes the selection improve with every run.
           Weight: 0.4 (highest weight — your data beats global averages)

        Brain updates alphas count and coverage live, so every fetch gives
        you the current state of the platform.

        Parameters
        ----------
        learner_stats : dict or None
            Pass learner.field_sharpe_map() — {field_id: avg_sharpe}
            If None, only Brain signals are used (first run).
        """
        PAGE_SIZE = 20     # Brain rejects any limit > 20 on /data-fields
        params = {
            "instrumentType": instrument_type,
            "region":         region,
            "universe":       universe,
            "delay":          delay,
            "limit":          PAGE_SIZE,
            "offset":         0,
        }
        # The API doesn't filter perfectly by category name but we can search for category types
        # like 'fundamental', 'model', etc., or just get all fields and rank them
        if category:
            params["search"] = category # search works better than category param in Brain API

        # Paginate until we have enough candidates to rank from
        raw_fields: list = []
        max_pages = max(1, (top_n * 3) // PAGE_SIZE)   # fetch 3x needed, then rank down

        for page in range(max_pages):
            params["offset"] = page * PAGE_SIZE
            try:
                resp = self.session.get(f"{self.base_url}/data-fields", params=params)
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                logger.error(f"Failed to fetch fields (page {page}): {resp.text}")
                raise e

            page_data = resp.json()
            page_results = page_data.get("results", [])
            raw_fields.extend(page_results)

            # Stop early if Brain returned fewer results than page size (last page)
            if len(page_results) < PAGE_SIZE:
                break

        logger.info(f"Fetched {len(raw_fields)} raw fields from Brain API ({max_pages} pages).")

        # ── Step 1: Filter by type and minimum coverage ──────────────────
        filtered = [
            f for f in raw_fields
            if f.get("type") == "MATRIX"
            and (f.get("coverage") or 0) >= min_coverage
        ]

        # ── Step 2: Compute composite score for each field ───────────────
        alpha_counts = [f.get("alphaCount", 0) for f in filtered]
        max_alphas   = max(alpha_counts, default=1)

        # Novelty score: peaks at moderate alpha counts, drops off at extremes.
        def _novelty(n: int) -> float:
            if n <= 0:
                return 0.1 # heavily penalize untested fields
            ratio = n / max(max_alphas, 1)
            if ratio > 0.70:
                return max(0.0, 1.0 - ratio)
            return min(1.0, ratio * 2.0) if ratio < 0.5 else 1.0 - (ratio - 0.5)

        # Local success score from Learner
        def _local_success(field_id: str) -> float:
            if not learner_stats:
                return 0.5
            sharpe = learner_stats.get(field_id)
            if sharpe is None:
                return 0.5
            return min(1.0, max(0.0, sharpe / 1.5))

        scored = []
        for f in filtered:
            cov     = min(1.0, f.get("coverage", 0))
            novelty = _novelty(f.get("alphaCount", 0))
            local   = _local_success(f["id"])
            composite = 0.3 * cov + 0.3 * novelty + 0.4 * local
            f["_score"] = round(composite, 4)
            scored.append(f)

        # ── Step 3: Sort by composite score descending ───────────────────
        scored.sort(key=lambda f: f["_score"], reverse=True)
        top_fields = scored[:top_n]

        logger.info(
            f"Top {len(top_fields)} fields selected "
            f"(coverage×0.3 + novelty×0.3 + local_success×0.4):"
        )
        for f in top_fields[:5]:
            logger.info(
                f"  {f['id']:<45}  score={f['_score']}  "
                f"alphas={f.get('alphaCount', '?')}  coverage={f.get('coverage', 0):.0%}"
            )

        return top_fields
