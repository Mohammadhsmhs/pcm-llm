import csv
import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List


class RunInfoLogger:
    """
    Dedicated logger for tracking run information and real-time task completion.
    Provides immediate feedback and detailed run metadata.
    """

    def __init__(self, run_id: str = None, log_dir: str = "logs"):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = log_dir
        self.start_time = datetime.now()

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # File paths
        self.run_info_file = os.path.join(log_dir, f"run_info_{self.run_id}.json")
        self.task_log_file = os.path.join(log_dir, f"task_log_{self.run_id}.csv")
        self.real_time_log_file = os.path.join(log_dir, f"realtime_{self.run_id}.log")

        # Initialize files
        self._init_run_info()
        self._init_task_log()
        self._init_real_time_log()

        # Thread safety
        self.lock = threading.Lock()

        print(f"📋 Run Info Logger initialized: {self.run_id}")
        print(
            f"📁 Run files: run_info_{self.run_id}.json, task_log_{self.run_id}.csv, realtime_{self.run_id}.log"
        )

    def _init_run_info(self):
        """Initialize run information file."""
        run_info = {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "status": "running",
            "config": {},
            "tasks_completed": 0,
            "total_tasks": 0,
            "memory_usage": [],
            "performance_metrics": {},
            "errors": [],
        }

        with open(self.run_info_file, "w") as f:
            json.dump(run_info, f, indent=2)

    def _init_task_log(self):
        """Initialize task completion log CSV."""
        with open(self.task_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "task_id",
                    "task_type",
                    "compression_method",
                    "status",
                    "latency",
                    "score",
                    "tokens_input",
                    "tokens_output",
                    "memory_usage",
                    "prompt_preview",
                    "output_preview",
                ]
            )

    def _init_real_time_log(self):
        """Initialize real-time log file."""
        with open(self.real_time_log_file, "w") as f:
            f.write(f"=== Real-time Log for Run {self.run_id} ===\n")
            f.write(f"Started: {self.start_time}\n\n")

    def update_run_config(self, config: Dict[str, Any]):
        """Update run configuration information."""
        with self.lock:
            try:
                with open(self.run_info_file, "r") as f:
                    run_info = json.load(f)

                run_info["config"] = config

                with open(self.run_info_file, "w") as f:
                    json.dump(run_info, f, indent=2)
            except Exception as e:
                self.log_error(f"Failed to update run config: {e}")

    def log_task_completion(self, task_data: Dict[str, Any]):
        """Log individual task completion with full details."""
        with self.lock:
            try:
                timestamp = datetime.now().isoformat()

                # Write to CSV task log
                with open(self.task_log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            timestamp,
                            task_data.get("task_id", ""),
                            task_data.get("task_type", ""),
                            task_data.get("compression_method", ""),
                            task_data.get("status", "completed"),
                            task_data.get("latency", 0),
                            task_data.get("score", 0),
                            task_data.get("tokens_input", 0),
                            task_data.get("tokens_output", 0),
                            task_data.get("memory_usage", 0),
                            task_data.get("prompt_preview", "")[
                                :100
                            ],  # Truncate for CSV
                            task_data.get("output_preview", "")[
                                :100
                            ],  # Truncate for CSV
                        ]
                    )

                # Write detailed info to real-time log
                self._write_real_time_log(task_data)

                # Update run info
                self._update_run_progress()

            except Exception as e:
                self.log_error(f"Failed to log task completion: {e}")

    def _write_real_time_log(self, task_data: Dict[str, Any]):
        """Write detailed task information to real-time log."""
        try:
            with open(self.real_time_log_file, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"TASK COMPLETED: {task_data.get('task_id', 'unknown')}\n")
                f.write(f"Time: {datetime.now()}\n")
                f.write(f"Type: {task_data.get('task_type', 'unknown')}\n")
                f.write(f"Method: {task_data.get('compression_method', 'none')}\n")
                f.write(f"Status: {task_data.get('status', 'completed')}\n")
                f.write(f"Latency: {task_data.get('latency', 0):.2f}s\n")
                f.write(f"Score: {task_data.get('score', 0):.3f}\n")
                f.write(f"Memory: {task_data.get('memory_usage', 0):.1f}MB\n")

                if "prompt_preview" in task_data:
                    f.write(f"\nInput Prompt:\n{task_data['prompt_preview']}\n")

                if "output_preview" in task_data:
                    f.write(f"\nOutput Response:\n{task_data['output_preview']}\n")

                f.write(f"{'='*60}\n")

        except Exception as e:
            print(f"⚠️  Failed to write real-time log: {e}")

    def _update_run_progress(self):
        """Update run progress in run info file."""
        try:
            with open(self.run_info_file, "r") as f:
                run_info = json.load(f)

            run_info["tasks_completed"] += 1
            run_info["last_update"] = datetime.now().isoformat()

            with open(self.run_info_file, "w") as f:
                json.dump(run_info, f, indent=2)

        except Exception as e:
            self.log_error(f"Failed to update run progress: {e}")

    def log_memory_usage(self, memory_mb: float):
        """Log memory usage for monitoring."""
        with self.lock:
            try:
                with open(self.run_info_file, "r") as f:
                    run_info = json.load(f)

                run_info["memory_usage"].append(
                    {"timestamp": datetime.now().isoformat(), "memory_mb": memory_mb}
                )

                with open(self.run_info_file, "w") as f:
                    json.dump(run_info, f, indent=2)

            except Exception as e:
                self.log_error(f"Failed to log memory usage: {e}")

    def log_error(self, error_message: str):
        """Log errors to run info file."""
        with self.lock:
            try:
                with open(self.run_info_file, "r") as f:
                    run_info = json.load(f)

                run_info["errors"].append(
                    {"timestamp": datetime.now().isoformat(), "message": error_message}
                )

                with open(self.run_info_file, "w") as f:
                    json.dump(run_info, f, indent=2)

                # Also write to real-time log
                with open(self.real_time_log_file, "a") as f:
                    f.write(f"\nERROR: {datetime.now()} - {error_message}\n")

            except Exception as e:
                print(f"⚠️  Failed to log error: {e}")

    def finalize_run(self, final_stats: Dict[str, Any] = None):
        """Finalize the run with final statistics."""
        with self.lock:
            try:
                with open(self.run_info_file, "r") as f:
                    run_info = json.load(f)

                run_info["status"] = "completed"
                run_info["end_time"] = datetime.now().isoformat()
                run_info["duration_seconds"] = (
                    datetime.now() - self.start_time
                ).total_seconds()

                if final_stats:
                    run_info["final_stats"] = final_stats

                with open(self.run_info_file, "w") as f:
                    json.dump(run_info, f, indent=2)

                # Final entry in real-time log
                with open(self.real_time_log_file, "a") as f:
                    f.write(f"\n{'='*60}\n")
                    f.write("RUN COMPLETED\n")
                    f.write(f"End Time: {datetime.now()}\n")
                    f.write(f"Duration: {run_info['duration_seconds']:.1f} seconds\n")
                    f.write(f"Tasks Completed: {run_info['tasks_completed']}\n")
                    f.write(f"{'='*60}\n")

                print(f"✅ Run {self.run_id} finalized")
                print(f"📊 Final stats saved to {self.run_info_file}")

            except Exception as e:
                self.log_error(f"Failed to finalize run: {e}")

    def get_run_summary(self) -> Dict[str, Any]:
        """Get current run summary."""
        try:
            with open(self.run_info_file, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to read run summary: {e}"}
