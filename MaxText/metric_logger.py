"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=bare-except, consider-using-generator
# pytype: disable=attribute-error
"""Logger that saves metrics to a local file, GCS and TensorBoard."""

import json
import os
import queue
from typing import List, Dict, Any

import numpy as np

import jax

from MaxText import max_logging
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText.utils import gcs_utils
from MaxText.gcp_workload_monitor import GCPWorkloadMonitor
from MaxText.globals import EPS

from collections import defaultdict


def _prepare_metrics_for_json(metrics, step, run_name):
  """Converts metric dictionary into json supported types (e.g. float)"""
  metrics_dict = {val: float(metrics["scalar"][val]) for val in metrics["scalar"]}
  metrics_dict["step"] = float(step)
  metrics_dict["run_name"] = run_name
  return metrics_dict


class MetricLogger:
  """
  Logger for saving metrics to a local file, GCS and TensorBoard.
  """

  def __init__(self, config, learning_rate_schedule):
    self.writer = max_utils.initialize_summary_writer(config.tensorboard_dir, config.run_name)
    self.config = config
    self.metadata = {}
    self.running_gcs_metrics = [] if config.gcs_metrics else None
    self.performance_metric_queue = self.get_performance_metric_queue(config)
    self.learning_rate_schedule = learning_rate_schedule
    self.cumulative_eval_metrics = {"scalar": defaultdict(float)}
    self.buffered_train_metrics = None
    
    # Enable Weights & Biases on process 0 only if requested.
    self.use_wandb = getattr(config, "use_wandb", False) and jax.process_index() == 0
    # Internal flags for optional W&B relog/backfill behavior
    self._wandb_relog_requested = False
    self._wandb_relog_done = False
    self._wandb_relog_source = None  # 'gcs', 'tensorboard', or 'auto'

    if self.use_wandb:
      try:
        import wandb  # type: ignore
        # Convert config to a plain dict for wandb; fallback to None if unavailable
        cfg_dict = None
        try:
          # Prefer an explicit dict of keys if available
          if hasattr(config, "get_keys"):
            cfg_dict = dict(config.get_keys())
          elif hasattr(config, "to_dict"):
            cfg_dict = dict(config.to_dict())  # type: ignore
        except Exception:
          cfg_dict = None

        # Support env var overrides so we don't require config edits.
        # Project/name can come from config or env (WANDB_PROJECT, WANDB_NAME).
        project = getattr(config, "wandb_project", "") or os.environ.get("WANDB_PROJECT", "")
        name = getattr(config, "wandb_run_name", "") or os.environ.get("WANDB_NAME", "")
        # Support resuming into the same run to keep a single continuous curve.
        # If provided, use a fixed run id and a resume policy (e.g. "allow" or "must").
        run_id = getattr(config, "wandb_run_id", "") or os.environ.get("WANDB_RUN_ID")
        resume_policy = getattr(config, "wandb_resume", None) or os.environ.get("WANDB_RESUME")

        # Special relog mode: backfill from history before continuing.
        # We convert the policy to 'allow' for wandb.init and set a flag to run backfill once.
        if resume_policy == "relog":
          self._wandb_relog_requested = True
          # default relog source is gcs; can override via CLI or env
          self._wandb_relog_source = (
              getattr(config, "wandb_relog_source", None)
              or os.environ.get("WANDB_RELOG_SOURCE", "gcs")
          )
          resume_policy = "allow"

        init_kwargs = dict(
            project=project,
            name=name,
            config=cfg_dict,
        )
        if run_id:
            init_kwargs["id"] = run_id
        if resume_policy:
            init_kwargs["resume"] = resume_policy

        wandb.init(**init_kwargs)
        # Stash module for later usage without re-importing
        self._wandb = wandb
      except Exception:  # pylint: disable=broad-except
        # If wandb is not available or initialization fails, disable it gracefully.
        self.use_wandb = False

  def write_metrics(self, metrics, step, is_training=True):
    """Entry point for all metrics writing in Train's Main."""
    if metrics:
      # Perform an optional one-time backfill to W&B if requested.
      if self.use_wandb and self._wandb_relog_requested and not self._wandb_relog_done and jax.process_index() == 0:
        try:
          self._maybe_wandb_relog(int(step))
        except Exception:
          # Never break training due to relog failures; just continue.
          pass

      self.log_metrics(metrics, step, is_training)

      if self.config.enable_tensorboard:
        self.write_metrics_to_tensorboard(metrics, step, is_training)

      if self.config.metrics_file:
        self.write_metrics_locally(metrics, step)

      if self.config.gcs_metrics and jax.process_index() == 0:
        self.write_metrics_for_gcs(metrics, step, is_training)
        
      if self.use_wandb:
        self.write_metrics_to_wandb(metrics, step)

  def log_metrics(self, metrics, step, is_training):
    """Logs metrics via max_logging."""

    if is_training:
      loss = metrics['scalar']['learning/loss']
      log_message = (
          f"completed step: {step}, seconds: {metrics['scalar']['perf/step_time_seconds']:.3f}, "
          f"TFLOP/s/device: {metrics['scalar']['perf/per_device_tflops_per_sec']:.3f}, "
          f"Tokens/s/device: {metrics['scalar']['perf/per_device_tokens_per_sec']:.3f}, "
          f"total_weights: {metrics['scalar']['learning/total_weights']}, "
          f"loss: {loss:.3f}, "
          f"lr: {self.learning_rate_schedule(step):.3e}, "
          f"grad_norm: {metrics['scalar']['learning/grad_norm']}, "
          f"raw_grad_norm: {metrics['scalar']['learning/raw_grad_norm']}"
      )
      if "learning/zeros" in metrics['scalar']:
        log_message += f", sparsity: {metrics['scalar']['learning/zeros'] / metrics['scalar']['learning/params']}"

      if self.config.mtp_num_layers > 0:
        mtp_loss = metrics["scalar"].get("learning/mtp_loss", 0.0)
        main_model_loss = loss - mtp_loss
        log_message += f", main_model_loss: {main_model_loss:.3f}, mtp_loss: {mtp_loss:.3f}"

      # Optional CE/KD breakdown if KD is enabled
      if getattr(self.config, 'use_kd', False):
        ce = metrics['scalar'].get('learning/ce_loss')
        kd = metrics['scalar'].get('learning/kd_loss')
        if ce is not None:
          log_message += f", ce_loss: {ce:.3f}"
        if kd is not None:
          log_message += f", kd_loss: {kd:.3f}"

      max_logging.log(log_message)

    else:
      log_message = (
          f"eval metrics after step: {step},"
          f" loss={self.cumulative_eval_metrics['scalar']['eval/avg_loss']:.3f},"
          f" total_weights={self.cumulative_eval_metrics['scalar']['eval/total_weights']},"
          f" step_time_seconds={self.cumulative_eval_metrics['scalar']['eval/step_time_seconds']:.3f}"
      )

      if self.config.mtp_num_layers > 0:
        log_message += (
            f", avg_mtp_loss={self.cumulative_eval_metrics['scalar']['eval/avg_mtp_loss']:.3f},"
            f" avg_mtp_acceptance_rate={self.cumulative_eval_metrics['scalar']['eval/avg_mtp_acceptance_rate_percent']:.2f}%"
        )

      # Optional CE/KD breakdown if KD is enabled
      if getattr(self.config, 'use_kd', False):
        avg_ce = self.cumulative_eval_metrics['scalar'].get('eval/avg_ce_loss')
        avg_kd = self.cumulative_eval_metrics['scalar'].get('eval/avg_kd_loss')
        if avg_ce is not None:
          log_message += f", avg_ce_loss={avg_ce:.3f}"
        if avg_kd is not None:
          log_message += f", avg_kd_loss={avg_kd:.3f}"

      max_logging.log(log_message)

  def write_metrics_locally(self, metrics, step):
    """Writes metrics locally for testing."""
    with open(self.config.metrics_file, "a", encoding="utf8") as local_metrics_file:
      if step == 0:
        local_metrics_file.truncate(0)

      metrics_dict = _prepare_metrics_for_json(metrics, step, self.config.run_name)
      local_metrics_file.write(str(json.dumps(metrics_dict)) + "\n")

  def write_metrics_for_gcs(self, metrics, step, is_training):
    """Writes metrics to GCS."""
    metrics_dict_step = _prepare_metrics_for_json(metrics, step, self.config.run_name)
    self.running_gcs_metrics.append(metrics_dict_step)
    if is_training and (step + 1) % self.config.log_period == 0 or step == self.config.steps - 1:
      start_step = (step // self.config.log_period) * self.config.log_period
      metrics_filename = f"metrics_step_{start_step:06}_to_step_{step:06}.txt"
      with open(metrics_filename, "wt", encoding="utf8") as metrics_for_gcs:
        for metrics_step in self.running_gcs_metrics:
          metrics_for_gcs.write(str(json.dumps(metrics_step)) + "\n")

      gcs_filename = os.path.join(self.config.metrics_dir, metrics_filename)
      max_logging.log(f"Moving file {metrics_filename} to GCS...")
      gcs_utils.upload_blob(gcs_filename, metrics_filename)
      max_logging.log(f"File {metrics_filename} moved successfully!")
      self.running_gcs_metrics = []  # reset running_metrics to empty list

  def write_metrics_to_tensorboard(self, metrics, step, is_training):
    """Writes metrics to TensorBoard."""
    if jax.process_index() == 0:
      for metric_name in metrics.get("scalar", []):
        self.writer.add_scalar(metric_name, np.array(metrics["scalar"][metric_name]), step)
      for metric_name in metrics.get("scalars", []):
        self.writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)

    if is_training:
      full_log = step % self.config.log_period == 0

      if full_log and jax.process_index() == 0:
        max_logging.log(f"To see full metrics 'tensorboard --logdir={self.config.tensorboard_dir}'")
        self.writer.flush()
        
  def write_metrics_to_wandb(self, metrics, step):
    flat_metrics = {}
    for key, val in metrics.get("scalar", {}).items():
      flat_metrics[key] = float(val)
    for key, val in metrics.get("scalars", {}).items():
      for subkey, subval in val.items():
        flat_metrics[f"{key}/{subkey}"] = float(subval)
    # Use cached module reference to avoid global import
    self._wandb.log(flat_metrics, step=step)

  # -------------------
  # W&B relog helpers
  # -------------------
  def _maybe_wandb_relog(self, current_step: int):
    """Backfill historical metrics into W&B once, before continuing logging.

    Only runs on process 0. Safe to call multiple times; executes once.
    """
    if self._wandb_relog_done:
      return
    source = (self._wandb_relog_source or "gcs").lower()
    if source == "auto":
      did = self._wandb_try_gcs(current_step)
      if not did:
        self._wandb_try_tensorboard(current_step)
    elif source == "gcs":
      self._wandb_try_gcs(current_step)
    elif source == "tensorboard":
      self._wandb_try_tensorboard(current_step)
    self._wandb_relog_done = True

  def _wandb_try_gcs(self, current_step: int) -> bool:
    """Reads metric JSONL files from GCS metrics_dir and logs them to W&B.

    Only logs entries with step < current_step to avoid duplication.
    """
    try:
      from google.cloud import storage  # Lazy import
    except Exception:
      return False

    metrics_dir = getattr(self.config, "metrics_dir", "")
    if not metrics_dir or not metrics_dir.startswith("gs://"):
      return False

    try:
      storage_client = storage.Client()
      # Parse bucket and prefix
      from MaxText.utils.gcs_utils import parse_gcs_bucket_and_prefix, add_trailing_slash
      bucket_name, prefix = parse_gcs_bucket_and_prefix(metrics_dir)
      prefix = add_trailing_slash(prefix)
      bucket = storage_client.bucket(bucket_name)

      # List all metric files under metrics_dir
      blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
      entries: List[Dict[str, Any]] = []
      for blob in blobs:
        name = blob.name
        if not name.endswith('.txt'):
          continue
        try:
          content = blob.download_as_text()
          for line in content.splitlines():
            line = line.strip()
            if not line:
              continue
            try:
              rec = json.loads(line)
              step_val = int(rec.get("step", -1))
              if step_val >= 0 and step_val < int(current_step):
                # Remove non-metric fields
                rec = {k: v for k, v in rec.items() if k not in ("step", "run_name")}
                entries.append({"step": step_val, "metrics": rec})
            except Exception:
              continue
        except Exception:
          continue

      if not entries:
        return False
      # Sort by step and de-dup by step keeping last occurrence
      entries.sort(key=lambda x: x["step"]) 
      dedup: Dict[int, Dict[str, float]] = {}
      for e in entries:
        dedup[e["step"]] = e["metrics"]

      for step_val in sorted(dedup.keys()):
        self._wandb.log(dedup[step_val], step=int(step_val))
      return True
    except Exception:
      # Silent failure, training should proceed
      return False

  def _wandb_try_tensorboard(self, current_step: int) -> bool:
    """Parses TensorBoard event files and logs scalars to W&B.

    Looks under {tensorboard_dir}/{run_name} unless overridden by WANDB_RELOG_TB_DIR.
    Only logs entries with step < current_step.
    """
    try:
      from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore
    except Exception:
      return False

    # Determine TB directory
    tb_dir_override = os.environ.get("WANDB_RELOG_TB_DIR")
    if tb_dir_override:
      tb_dir = tb_dir_override
    else:
      base = getattr(self.config, "tensorboard_dir", "")
      run_name = getattr(self.config, "run_name", "")
      if not base or not run_name:
        return False
      tb_dir = os.path.join(base, run_name)

    try:
      ea = EventAccumulator(tb_dir)
      ea.Reload()
    except Exception:
      return False

    try:
      scalar_tags = list(ea.Tags().get('scalars', []))
      if not scalar_tags:
        return False

      # Aggregate by step
      by_step: Dict[int, Dict[str, float]] = {}
      for tag in scalar_tags:
        for ev in ea.Scalars(tag):
          step_val = int(ev.step)
          if step_val >= int(current_step):
            continue
          d = by_step.setdefault(step_val, {})
          # tensorboard tags are already hierarchical; keep as-is
          d[tag] = float(ev.value)

      if not by_step:
        return False

      for step_val in sorted(by_step.keys()):
        self._wandb.log(by_step[step_val], step=int(step_val))
      return True
    except Exception:
      return False

  def write_setup_info_to_tensorboard(self, params):
    """Writes setup information like train config params, num model params, and XLA flags to TensorBoard."""
    num_model_parameters = max_utils.calculate_num_params_from_pytree(params)
    self.metadata["per_device_tflops"], _, _ = maxtext_utils.calculate_tflops_training_per_device(self.config)
    self.metadata["per_device_tokens"] = maxtext_utils.calculate_tokens_training_per_device(self.config)
    max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")
    max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), self.writer)
    max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], self.writer)
    maxtext_utils.add_config_to_summary_writer(self.config, self.writer)

  def get_performance_metric_queue(self, config):
    """Records heartbeat metrics and performance metrics to GCP."""
    performance_metric_queue = None
    if config.report_heartbeat_metric_for_gcp_monitoring or config.report_performance_metric_for_gcp_monitoring:
      gcp_workload_monitor = GCPWorkloadMonitor(config.run_name)
      if config.report_heartbeat_metric_for_gcp_monitoring:
        gcp_workload_monitor.start_heartbeat_reporting_thread(config.heartbeat_reporting_interval_in_seconds)
      if config.report_performance_metric_for_gcp_monitoring:
        performance_metric_queue = queue.Queue()
        gcp_workload_monitor.start_performance_reporting_thread(performance_metric_queue)
    return performance_metric_queue

  def buffer_and_write_train_metrics(self, metrics, step, step_time_delta):
    """
    Buffers metrics for the current training step and simultaneously writes the training metrics
    for the previous step to GCS and/or TensorBoard. This buffering strategy allows for back-to-back
    execution of training steps, by overlapping data loading for step n with the execution of step n−1.
    This significantly boosts training efficiency.
    """
    if self.buffered_train_metrics is not None:
      (step_to_write, metrics_to_write) = self.buffered_train_metrics
      self.write_metrics(metrics_to_write, step_to_write)

    self.record_train_metrics(metrics, step, step_time_delta)
    self.buffered_train_metrics = (step, metrics)

  def record_train_metrics(self, metrics, step, step_time_delta):
    """Records training metrics for the current step."""
    metrics["scalar"].update({"perf/step_time_seconds": step_time_delta.total_seconds()})
    metrics["scalar"].update({"perf/per_device_tflops": self.metadata["per_device_tflops"]})
    metrics["scalar"].update(
        {"perf/per_device_tflops_per_sec": self.metadata["per_device_tflops"] / step_time_delta.total_seconds()}
    )
    metrics["scalar"].update({"perf/per_device_tokens": self.metadata["per_device_tokens"]})
    metrics["scalar"].update(
        {"perf/per_device_tokens_per_sec": self.metadata["per_device_tokens"] / step_time_delta.total_seconds()}
    )
    metrics["scalar"].update({"learning/current_learning_rate": self.learning_rate_schedule(step)})
    if self.performance_metric_queue:
      self.performance_metric_queue.put(step_time_delta.total_seconds())

  def record_eval_metrics(self, step, metrics=None, eval_step_count=None):
    """Records eval metrics and writes the metrics to GCS and/or to TensorBoard."""
    if metrics:
      self.cumulative_eval_metrics["scalar"]["eval/total_loss"] += float(metrics["scalar"].get("evaluation/total_loss", 0.0))
      self.cumulative_eval_metrics["scalar"]["eval/total_weights"] += float(
          metrics["scalar"].get("evaluation/total_weights", 0.0)
      )
      self.cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] += float(
          metrics["scalar"].get("evaluation/moe_lb_loss", 0.0)
      )
      self.cumulative_eval_metrics["scalar"]["eval/mtp_loss"] += float(metrics["scalar"].get("evaluation/mtp_loss", 0.0))
      self.cumulative_eval_metrics["scalar"]["eval/mtp_acceptance_rate_percent"] += float(
          metrics["scalar"].get("evaluation/mtp_acceptance_rate_percent", 0.0)
      )
      if self.config.use_dpo:
        self.cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] += float(
            metrics["scalar"].get("evaluation/dpo_reward_accuracy", 0.0)
        )

    if eval_step_count:
      eval_loss = self.cumulative_eval_metrics["scalar"]["eval/total_loss"] / (
          self.cumulative_eval_metrics["scalar"]["eval/total_weights"] + EPS
      )
      self.cumulative_eval_metrics["scalar"]["eval/avg_loss"] = eval_loss
      # Derive avg CE and KD if their per-step scalars are present
      total_weights = self.cumulative_eval_metrics["scalar"]["eval/total_weights"]
      if total_weights > 0:
        ce_total = float(metrics["scalar"].get("evaluation/ce_loss", 0.0)) * float(metrics["scalar"].get("evaluation/total_weights", 0.0))
        kd_val = float(metrics["scalar"].get("evaluation/kd_loss", 0.0))
        # Maintain running averages by simple mean over steps if present
        prev_avg_ce = self.cumulative_eval_metrics["scalar"].get("eval/avg_ce_loss", 0.0)
        prev_avg_kd = self.cumulative_eval_metrics["scalar"].get("eval/avg_kd_loss", 0.0)
        if metrics["scalar"].get("evaluation/ce_loss") is not None:
          self.cumulative_eval_metrics["scalar"]["eval/avg_ce_loss"] = (
              (prev_avg_ce * (eval_step_count - 1) + float(metrics["scalar"]["evaluation/ce_loss"])) / eval_step_count
          )
        if metrics["scalar"].get("evaluation/kd_loss") is not None:
          self.cumulative_eval_metrics["scalar"]["eval/avg_kd_loss"] = (
              (prev_avg_kd * (eval_step_count - 1) + kd_val) / eval_step_count
          )
      self.cumulative_eval_metrics["scalar"]["eval/avg_moe_lb_loss"] = (
          self.cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] / eval_step_count
      )
      self.cumulative_eval_metrics["scalar"]["eval/avg_mtp_loss"] = (
          self.cumulative_eval_metrics["scalar"]["eval/mtp_loss"] / eval_step_count
      )
      self.cumulative_eval_metrics["scalar"]["eval/avg_mtp_acceptance_rate_percent"] = (
          self.cumulative_eval_metrics["scalar"]["eval/mtp_acceptance_rate_percent"] / eval_step_count
      )
      if self.config.use_dpo:
        self.cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] = (
            self.cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] / eval_step_count
        )

      self.write_metrics(self.cumulative_eval_metrics, step, is_training=False)

  def flush_metrics_and_cleanup(self):
    """
    This is a terminal operation that uploads any buffered metrics to GCS
    and/or TensorBoard before closing the writer objects. Once called, the
    logger instance should not be used to add or write more metrics as the
    underlying writer objects (e.g., TensorBoard SummaryWriter) will be closed.
    """
    if self.buffered_train_metrics is not None:
      (step_to_write, metrics_to_write) = self.buffered_train_metrics
      self.write_metrics(metrics_to_write, step_to_write)

    max_utils.close_summary_writer(self.writer)
    
    if self.use_wandb:
      self._wandb.finish()
