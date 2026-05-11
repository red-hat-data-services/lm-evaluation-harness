#!/usr/bin/env python3
"""LMEval adapter main entry point for evalhub.

Configuration is provided via a job spec JSON file (see `meta/job.json` for an example).
The callback_url in the job spec defines where status updates and results are sent.

Required environment variables:
- REGISTRY_URL: OCI registry URL

Optional environment variables:
- EVALHUB_JOB_SPEC_PATH: path to the job spec JSON (defaults to ``/meta/job.json``); must resolve under ``/meta``
- REGISTRY_USERNAME: Registry username (optional)
- REGISTRY_PASSWORD: Registry password/token (optional)

Offline / air-gapped clusters: The job file is read before ``lm_eval`` loads (see
``_seed_hf_offline_before_lm_eval_import``). The adapter only checks top-level
``parameters.tokenizer``—the same tokenizer path used everywhere else, not anything under nested
``parameters.parameters``. If that path exists on disk, sits under ``/test_data`` but is not the
``/test_data`` mount path by itself, and another folder next to it under ``/test_data`` contains
``dataset_dict.json`` (offline dataset files, e.g. next to ``tokenizer/``), the adapter turns on
Hugging Face offline mode: it sets ``HF_HOME`` and related env vars so those libraries use local
files and do not call the Hub. You do not need ``parameters.offline``.
"""

import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TEST_DATA_DIR = "/test_data"
# EvalHub mounts the job spec JSON under this directory only; reject other paths (CWE-22).
_JOB_SPEC_ALLOWED_ROOT = Path("/meta")


def _resolve_job_spec_path_for_read(path: str) -> Path | None:
    """Return a resolved path to open, or None if ``path`` is invalid or escapes ``/meta``."""
    if not isinstance(path, str) or not path.strip():
        print("WARNING: job spec path is empty; refusing to open", file=sys.stderr)
        return None
    try:
        resolved = Path(path.strip()).resolve()
    except (OSError, ValueError) as exc:
        print(f"WARNING: invalid job spec path {path!r}: {exc}", file=sys.stderr)
        return None
    try:
        allowed = _JOB_SPEC_ALLOWED_ROOT.resolve()
    except OSError:
        allowed = _JOB_SPEC_ALLOWED_ROOT
    if not resolved.is_relative_to(allowed):
        print(
            f"WARNING: job spec path {path!r} resolves to {resolved} "
            f"which is not under {allowed}; refusing to open",
            file=sys.stderr,
        )
        return None
    return resolved


def _read_job_spec_parameters_from_path(path: str) -> dict[str, Any]:
    """Load top-level ``parameters`` object from the job spec JSON file.

    Only files whose resolved path stays under ``/meta`` are opened (see ``EVALHUB_JOB_SPEC_PATH``).
    """
    resolved = _resolve_job_spec_path_for_read(path)
    if resolved is None:
        return {}
    try:
        with open(resolved, encoding="utf-8") as f:
            spec = json.load(f)
    except FileNotFoundError:
        return {}
    except PermissionError as exc:
        print(
            f"WARNING: permission denied reading job spec {resolved!r}: {exc}",
            file=sys.stderr,
        )
        return {}
    except OSError as exc:
        print(f"WARNING: I/O error reading job spec {resolved!r}: {exc}", file=sys.stderr)
        return {}
    except json.JSONDecodeError as exc:
        print(f"WARNING: invalid JSON in job spec {resolved!r}: {exc}", file=sys.stderr)
        return {}
    except TypeError as exc:
        print(
            f"WARNING: unexpected type while parsing job spec {resolved!r}: {exc}",
            file=sys.stderr,
        )
        return {}
    if not isinstance(spec, dict):
        return {}
    parameters = spec.get("parameters")
    if not isinstance(parameters, dict):
        return {}
    return parameters


def _extract_tokenizer_parameter(parameters: dict[str, Any]) -> str | None:
    """Tokenizer path or id from benchmark ``parameters.tokenizer`` (must match ``build_lmeval_config``).

    Only the top-level key is used—the same source as ``model_args["tokenizer"]``—so HF offline
    auto-detection never diverges from the tokenizer the adapter actually loads.
    """
    raw = parameters.get("tokenizer")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _dataset_material_present_under_test_data(
    root_resolved: Path, tokenizer_resolved: Path
) -> bool:
    """True when a DatasetDict bundle (``dataset_dict.json``) exists beside the tokenizer path.

    Both paths must already be ``Path.resolve()`` results. Matches EvalHub ``/test_data`` sync:
    e.g. ``tokenizer/`` and ``allenai--ai2_arc--ARC-Easy/`` as direct children of the mount.
    """
    try:
        for child in root_resolved.iterdir():
            if not child.is_dir():
                continue
            try:
                cres = child.resolve()
            except OSError:
                continue
            if cres == tokenizer_resolved:
                continue
            if tokenizer_resolved.is_relative_to(cres):
                continue
            if (child / "dataset_dict.json").is_file():
                return True
    except OSError:
        return False

    return False


def _infer_auto_offline_from_local_test_data(
    parameters: dict[str, Any],
    *,
    test_data_root: str | Path = _TEST_DATA_DIR,
) -> bool:
    """True when ``parameters.tokenizer`` points into ``test_data_root`` and datasets are co-located.

    Tokenizer: absolute path under ``test_data_root``, not the mount root alone, path exists.
    Datasets: see ``_dataset_material_present_under_test_data`` (same ``/test_data`` directory).
    """
    tokenizer_str = _extract_tokenizer_parameter(parameters)
    if not tokenizer_str or not tokenizer_str.startswith("/"):
        return False

    root = Path(test_data_root)
    try:
        root_res = root.resolve()
    except OSError:
        return False
    if not root_res.is_dir():
        return False

    tokenizer_path = Path(tokenizer_str)
    try:
        tok_res = tokenizer_path.resolve()
    except OSError:
        return False

    if tok_res == root_res or not tok_res.is_relative_to(root_res):
        return False
    try:
        if not tok_res.exists():
            return False
    except OSError:
        return False

    return _dataset_material_present_under_test_data(root_res, tok_res)


def configure_hf_offline_environment(hf_home: str) -> None:
    """Use local Hugging Face caches only (disconnected / no huggingface.co).

    Pin Hub/datasets cache dirs under hf_home so lookups match /test_data layout after init sync.
    HF_HOME, HF_HUB_CACHE and HF_DATASETS_CACHE are set consistently so they stay aligned.
    """
    root = Path(hf_home)
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(root / "datasets")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_EVALUATE_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _seed_hf_offline_before_lm_eval_import() -> None:
    path = os.environ.get("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
    if not _infer_auto_offline_from_local_test_data(_read_job_spec_parameters_from_path(path)):
        return
    configure_hf_offline_environment(_TEST_DATA_DIR)


from evalhub.adapter import (
    DefaultCallbacks,
    EvaluationResult,
    ErrorInfo,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    MessageInfo,
    OCIArtifactSpec,
)
from evalhub.adapter.auth import read_model_auth_key, resolve_model_credentials


_seed_hf_offline_before_lm_eval_import()

# NOTE: keep these imports after _seed_hf_offline_before_lm_eval_import() so HF offline env vars
# are set before lm_eval (and Hugging Face libraries) are imported.
from lm_eval import simple_evaluate  # noqa: E402
from lm_eval.tasks import TaskManager  # noqa: E402


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    """Best-effort conversion to JSON-serializable types."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _status_message(text: str, code: str = "status_update") -> MessageInfo:
    return MessageInfo(message=text, message_code=code)


def build_lmeval_config(job_spec: JobSpec) -> tuple[str, dict, str | None]:
    """Derive lm-evaluation-harness model backend + args from job spec.

    Always uses OpenAI-compatible endpoint configuration.
    Adapter-specific params (batch_size, tokenizer, parameters) come from job_spec.parameters.

    Returns:
        (model_backend, model_args, gen_kwargs)
    """
    model_spec = job_spec.model
    model_name = model_spec.name
    benchmark_params = job_spec.parameters

    # Adapter-specific settings from parameters
    _MAX_CONCURRENT = 128
    batch_size = int(benchmark_params.get("batch_size", 1))
    _raw_concurrent = int(benchmark_params.get("num_concurrent", 1))
    if _raw_concurrent <= 0:
        raise ValueError(
            f"num_concurrent must be a positive integer, got {_raw_concurrent}"
        )
    num_concurrent = min(_raw_concurrent, _MAX_CONCURRENT)
    if num_concurrent < _raw_concurrent:
        logger.warning(
            "num_concurrent clamped from %d to maximum %d",
            _raw_concurrent,
            _MAX_CONCURRENT,
        )
    timeout_seconds = int(benchmark_params.get("timeout_seconds", 300))

    # Optional generation parameters for generate_until tasks.
    parameters = benchmark_params.get("parameters", {})
    gen_kwargs = ",".join(f"{k}={v}" for k, v in parameters.items()) or None

    # Build completions URL from model.url
    base = str(model_spec.url or "").rstrip("/")
    if not base:
        raise ValueError(
            "Job spec.model.url is required for OpenAI-compatible endpoints"
        )

    if base.endswith("/completions"):
        completions_url = base
    elif base.endswith("/v1"):
        completions_url = f"{base}/completions"
    else:
        # best-effort: if the user gave a base URL, assume /v1/completions underneath
        completions_url = f"{base}/v1/completions"

    # For OpenAI-compatible endpoints, we need a HuggingFace tokenizer.
    # The tokenizer can be specified in parameters, otherwise use model.name
    tokenizer = str(benchmark_params.get("tokenizer", model_name))

    # Helpful error message if tokenizer is not a valid HF model
    if tokenizer == model_name and "/" not in tokenizer:
        logger.warning(
            f"Model name '{model_name}' may not be a valid HuggingFace tokenizer. "
            f"Specify the actual model in parameters.tokenizer "
            f"(e.g., 'google/flan-t5-small' or 'meta-llama/Llama-3.1-8B-Instruct')"
        )

    if num_concurrent <= 1:
        logger.info(
            "Concurrent requests are disabled (num_concurrent=1). "
            "Add num_concurrent to benchmark parameters to enable."
        )
    else:
        logger.info("Concurrent requests enabled: num_concurrent=%d", num_concurrent)

    # Use local-completions backend for OpenAI-compatible endpoints.
    # tokenized_requests=False ensures we send string prompts, not token IDs.
    return (
        "local-completions",
        {
            "model": model_name,
            "base_url": completions_url,
            "tokenizer_backend": "huggingface",
            "tokenizer": tokenizer,
            "tokenized_requests": False,
            "num_concurrent": num_concurrent,
            "batch_size": batch_size,
            "timeout": timeout_seconds,
        },
        gen_kwargs,
    )


class LMEvalAdapter(FrameworkAdapter):
    """LM Evaluation Harness adapter for EvalHub.

    This adapter integrates the lm-evaluation-harness framework with EvalHub,
    allowing benchmarks to be executed as EvalHub jobs.
    """

    def __init__(self, job_spec_path: str | None = None):
        """Initialize the LMEval adapter.

        Args:
            job_spec_path: Optional path to job specification file.
                          If not provided, uses EVALHUB_JOB_SPEC_PATH env var or default.
        """
        super().__init__(job_spec_path=job_spec_path)
        logger.info("LMEval adapter initialized")

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """Run LMEval benchmark with evalhub callbacks.

        Args:
            config: Job specification to execute
            callbacks: Callback handler for status and results

        Returns:
            JobResults: Evaluation results

        Raises:
            RuntimeError: If evaluation fails
        """
        start_time = time.time()

        try:
            # Auto-detect disconnected layout (see module docstring); re-apply HF env in case
            # the import-time seed skipped (e.g. job file unreadable at process start).
            benchmark_params = (
                config.parameters if isinstance(config.parameters, dict) else {}
            )
            hf_offline = _infer_auto_offline_from_local_test_data(benchmark_params)
            if hf_offline:
                test_data_dir = _TEST_DATA_DIR
                if not os.path.isdir(test_data_dir):
                    raise RuntimeError(
                        f"Local /test_data layout detected from parameters.tokenizer but "
                        f"{test_data_dir} does not exist. Ensure test_data_ref is configured so "
                        "the init container populates the directory before the adapter starts."
                    )
                configure_hf_offline_environment(test_data_dir)
                logger.info(
                    "HF offline mode (auto-detected from parameters.tokenizer + /test_data): "
                    "HF_HOME=%s, downloads disabled",
                    test_data_dir,
                )

            creds = resolve_model_credentials()
            if creds.api_key:
                os.environ["OPENAI_API_KEY"] = creds.api_key
            else:
                auth_value = creds.auth_headers.get("Authorization", "")
                if auth_value.startswith("Bearer "):
                    token = auth_value.replace("Bearer ", "").strip()
                    os.environ["OPENAI_API_KEY"] = token

            # Set HF_TOKEN for gated dataset access (e.g. leaderboard_gpqa).
            # Priority: HF_TOKEN env var > hf-token in model auth secret.
            if not os.environ.get("HF_TOKEN"):
                hf_token = read_model_auth_key("hf-token")
                if hf_token:
                    os.environ["HF_TOKEN"] = hf_token
                    logger.info("HF_TOKEN set from model auth secret (hf-token)")

            job_id = config.id
            benchmark_id = config.benchmark_id
            model_name = config.model.name

            # Number of examples from top-level JobSpec field
            # (extracted from parameters by the service)
            num_examples = config.num_examples

            # Adapter-specific params from parameters
            num_fewshot = int(benchmark_params.get("num_few_shot", 0))
            random_seed = int(benchmark_params.get("random_seed", 42))

            model_backend, model_args, gen_kwargs = build_lmeval_config(config)
            if creds.ca_cert_path:
                # Resolve to a real path so lm_eval accepts it (K8s secret volume mounts expose keys as symlinks).
                model_args["verify_certificate"] = os.path.realpath(creds.ca_cert_path)

            # Phase 1: Initialization
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=_status_message(
                        f"Initializing evaluation for {benchmark_id}"
                    ),
                )
            )

            logger.info(f"Job ID: {job_id}")
            logger.info(f"Model: {model_name}")
            logger.info(f"Benchmark: {benchmark_id}")
            logger.info(f"Examples limit: {num_examples}")
            logger.info(f"Few-shot: {num_fewshot}")
            logger.info("Device: cpu (forced)")
            logger.info(f"Model backend: {model_backend}")

            # Phase 2: Loading data
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.LOADING_DATA,
                    progress=0.1,
                    message=_status_message(
                        f"Loading benchmark data for {benchmark_id}"
                    ),
                )
            )

            # Initialize task manager
            task_manager = TaskManager()

            # Phase 3: Running evaluation
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.2,
                    message=_status_message(f"Running evaluation on {model_name}"),
                )
            )

            # Run evaluation based on job spec
            # Note: batch_size is passed in model_args for local-completions backend
            results = simple_evaluate(
                model=model_backend,
                model_args=model_args,
                tasks=[benchmark_id],
                num_fewshot=int(num_fewshot),
                device="cpu",
                limit=num_examples,
                random_seed=random_seed,
                numpy_random_seed=random_seed,
                torch_random_seed=random_seed,
                task_manager=task_manager,
                log_samples=True,
                gen_kwargs=gen_kwargs,
            )

            # Phase 4: Post-processing
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.8,
                    message=_status_message("Processing evaluation results"),
                )
            )

            # Extract results
            task_results = results.get("results", {}).get(benchmark_id, {})

            # For group tasks (e.g. leaderboard_bbh), lm-eval stores metrics under
            # subtask names, not the group name. Fall back to averaging subtask results.
            if not any(k.endswith(",none") for k in task_results):
                group_subtasks = results.get("group_subtasks", {}).get(benchmark_id, [])
                if group_subtasks:
                    logger.info(
                        "Benchmark %s is a group task, aggregating %d subtask results",
                        benchmark_id,
                        len(group_subtasks),
                    )
                    all_results = results.get("results", {})
                    subtask_metrics: dict[str, float] = {}
                    subtask_count: dict[str, int] = {}
                    for subtask in group_subtasks:
                        for metric_name, metric_value in all_results.get(subtask, {}).items():
                            if not metric_name.endswith(",none"):
                                continue
                            if metric_value == "N/A" or metric_value is None:
                                continue
                            clean = metric_name.replace(",none", "")
                            subtask_metrics[clean] = subtask_metrics.get(clean, 0) + float(metric_value)
                            subtask_count[clean] = subtask_count.get(clean, 0) + 1
                    task_results = {
                        f"{k},none": subtask_metrics[k] / subtask_count[k]
                        for k in subtask_metrics
                    }

            # Build evaluation results
            evaluation_results = []
            overall_score = None

            for metric_name, metric_value in task_results.items():
                if metric_name.endswith(",none"):
                    # Primary metric (usually accuracy or similar)
                    clean_metric = metric_name.replace(",none", "")
                    # lm-eval returns 'N/A' for metrics it cannot compute
                    # (e.g. when the model produces unparseable outputs).
                    if metric_value == "N/A" or metric_value is None:
                        logger.warning(
                            "Metric %s has value N/A, skipping",
                            clean_metric,
                        )
                        continue
                    evaluation_results.append(
                        EvaluationResult(
                            metric_name=clean_metric,
                            metric_value=float(metric_value),
                        )
                    )
                    # Use first primary metric as overall score
                    if overall_score is None:
                        overall_score = float(metric_value)

            # Get number of examples evaluated
            samples = results.get("samples", {}).get(benchmark_id, [])
            num_examples_evaluated = (
                len(samples) if isinstance(samples, list) else num_examples
            )

            duration = time.time() - start_time

            # Prepare metadata (convert non-serializable objects to strings)
            lmeval_config = results.get("config", {})
            serializable_config = _jsonable(lmeval_config)

            # Create job results
            job_results = JobResults(
                id=job_id,
                benchmark_id=benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=model_name,
                results=evaluation_results,
                overall_score=overall_score,
                num_examples_evaluated=int(num_examples_evaluated)
                if num_examples_evaluated is not None
                else 0,
                duration_seconds=duration,
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "lmeval_version": results.get("lm_eval_version", "unknown"),
                    "framework": "lm-evaluation-harness",
                    "config": serializable_config,
                    "job_spec": _jsonable(config.model_dump()),
                },
            )

            # Phase 5: Persist artifacts
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.PERSISTING_ARTIFACTS,
                    progress=0.9,
                    message=_status_message("Persisting evaluation artifacts"),
                )
            )

            # Save results to file
            output_dir = Path(__file__).parent / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            results_file = output_dir / f"results_{job_id}.json"
            with open(results_file, "w") as f:
                json.dump(
                    job_results.model_dump(mode="json"),
                    f,
                    indent=2,
                    default=str,
                )

            # Create OCI artifact (only when exports are configured)
            oci_exports = config.exports.oci if config.exports else None
            if oci_exports is not None:
                coords = oci_exports.coordinates.model_copy(deep=True)
                coords.annotations.update(
                    {
                        "org.opencontainers.image.created": datetime.now(UTC).isoformat(),
                        "io.github.eval-hub.benchmark": benchmark_id,
                        "io.github.eval-hub.model": model_name,
                        "io.github.eval-hub.job_id": job_id,
                    }
                )
                oci_spec = OCIArtifactSpec(
                    files_path=output_dir,
                    coordinates=coords,
                )
                oci_result = callbacks.create_oci_artifact(oci_spec)
                job_results.oci_artifact = oci_result
                logger.info(f"OCI artifact created: {oci_result.reference}")
            else:
                logger.info("No OCI exports configured; skipping artifact persistence")

            # Return results (will be reported by entrypoint)
            return job_results

        except Exception as e:
            logger.error("Evaluation failed", exc_info=True)

            error_str = str(e)
            error_lower = error_str.lower()
            is_gated = (
                "gated repo" in error_lower
                or "gated dataset" in error_lower
                or ("403" in error_lower and "huggingface" in error_lower)
            )

            if is_gated:
                error_message = (
                    "Gated HuggingFace dataset error; authentication required. "
                    "Set HF_TOKEN by adding an 'hf-token' key to your "
                    "model auth secret (model.auth.secret_ref)."
                )
                error_code = "gated_dataset_auth_required"
            else:
                error_message = f"Evaluation failed: {type(e).__name__}"
                error_code = "evaluation_failed"

            # Report failure
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    phase=JobPhase.COMPLETED,
                    progress=0.0,
                    message=_status_message(
                        "Evaluation failed", code=error_code
                    ),
                    error=ErrorInfo(
                        message=error_message,
                        message_code=error_code,
                    ),
                    error_details={"exception_type": type(e).__name__},
                )
            )

            raise RuntimeError(f"Evaluation failed: {e}") from e


def main() -> int:
    """Main entry point.

    The adapter automatically loads:
    - Settings from environment variables (REGISTRY_URL, etc.)
    - JobSpec from /meta/job.json (mounted via ConfigMap in Kubernetes)

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    import os

    try:
        # Create adapter with job spec path from environment or default
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = LMEvalAdapter(job_spec_path=job_spec_path)

        logger.info("=" * 80)
        logger.info("LMEval EvalHub Adapter")
        logger.info("=" * 80)
        logger.info(f"Loaded job spec from: {job_spec_path}")
        logger.info("Job spec configuration:")
        logger.info(f"  Job ID: {adapter.job_spec.id}")
        logger.info(f"  Benchmark: {adapter.job_spec.benchmark_id}")
        logger.info(f"  Model: {adapter.job_spec.model.name}")
        logger.info(f"  Examples: {adapter.job_spec.num_examples}")
        few_shot = adapter.job_spec.parameters.get("num_few_shot")
        logger.info(f"  Few-shot: {few_shot}")
        logger.info("=" * 80)
        logger.info(f"Callback URL: {adapter.job_spec.callback_url}")
        logger.info(f"Provider ID: {adapter.job_spec.provider_id}")
        logger.info(
            "OCI registry auth config present: %s",
            bool(adapter.settings.oci_auth_config_path),
        )
        logger.info("OCI insecure: %s", adapter.settings.oci_insecure)
        logger.info("EvalHub insecure: %s", adapter.settings.evalhub_insecure)
        logger.info("=" * 80)

        # Initialize callbacks using job spec callback_url and adapter settings
        callbacks = DefaultCallbacks.from_adapter(adapter)

        # Run evaluation
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

        logger.info("=" * 80)
        logger.info("Evaluation completed successfully")
        logger.info(f"Overall score: {results.overall_score}")
        logger.info(f"Examples evaluated: {results.num_examples_evaluated}")
        logger.info(f"Duration: {results.duration_seconds:.2f}s")
        logger.info("=" * 80)

        # MLflow first; run id from save() is sent on report_results when SDK returns it.
        mlflow_run_id = callbacks.mlflow.save(results, adapter.job_spec)
        if mlflow_run_id:
            results.mlflow_run_id = mlflow_run_id

        # Report final results to EvalHub (status/results API)
        callbacks.report_results(results)

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
