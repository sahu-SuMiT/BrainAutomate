import csv
import json
import logging
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


class ResultLogger:
    """
    Persists simulation results to both a JSON lines file and a CSV file
    under ./results/ so you can review outcomes after a run.

    Files created:
        results/run_<timestamp>.jsonl
        results/run_<timestamp>.csv
        results/qualifying_alphas.csv   ← cumulative; all QUALIFIES across runs
    """

    def __init__(self):
        RESULTS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.jsonl_path = RESULTS_DIR / f"run_{ts}.jsonl"
        self.csv_path = RESULTS_DIR / f"run_{ts}.csv"
        self.qualifying_path = RESULTS_DIR / "qualifying_alphas.csv"
        self._csv_initialized = False
        self._lock = threading.Lock()   # Thread-safe file writes for parallel workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, alpha_name: str, sim_id: str, status: str, details: dict = None,
               meta: dict = None):
        """
        Persist one simulation result.

        Parameters
        ----------
        alpha_name : str
        sim_id : str
        status : str   One of SUCCESS | WARNING | ERROR | FAILED
        details : dict Extra fields returned by the API (sharpe, fitness, etc.)
        meta : dict    _meta dict from the alpha (op, lookback, field, etc.) for Learner reload
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "alpha_name": alpha_name,
            "simulation_id": sim_id,
            "status": status,
            **(details or {}),
        }
        # Store meta fields so Learner.load_history() can reconstruct feature vectors
        if meta:
            entry["operator"]       = meta.get("op", "rank")
            entry["lookback"]       = meta.get("lookback")
            entry["decay"]          = meta.get("decay", 0)
            entry["neutralization"] = meta.get("neutralization", "MARKET")
            entry["field"]          = meta.get("field", "")
            entry["field_category"] = meta.get("field_category", "price_volume")
            # Store expression string so GeneticEvolver can mutate it on future runs
            if meta.get("expression"):
                entry["expression"] = meta["expression"]
        self._write_jsonl(entry)
        self._write_csv(entry)
        logger.info(f"Result recorded for {alpha_name}: {status}")

    def record_qualifying(self, alpha_name: str, sim_id: str, details: dict = None):
        """
        Append a qualifying alpha to the persistent qualifying_alphas.csv.

        This file accumulates across all runs so you can review and manually
        submit these alphas on your main account later.
        """
        entry = {
            "timestamp":      datetime.now().isoformat(),
            "alpha_name":     alpha_name,
            "simulation_id":  sim_id,
            "sharpe":         (details or {}).get("sharpe"),
            "fitness":        (details or {}).get("fitness"),
            "turnover":       (details or {}).get("turnover"),
            "margin":         (details or {}).get("margin"),
        }
        with self._lock:
            file_exists = self.qualifying_path.exists()
            with open(self.qualifying_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=entry.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(entry)
        logger.info(
            f"[QUALIFIES] {alpha_name} logged to qualifying_alphas.csv "
            f"(sharpe={entry['sharpe']}, fitness={entry['fitness']})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_jsonl(self, entry: dict):
        with self._lock:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def _write_csv(self, entry: dict):
        with self._lock:
            write_header = not self._csv_initialized
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=entry.keys())
                if write_header:
                    writer.writeheader()
                    self._csv_initialized = True
                writer.writerow(entry)
