#!/usr/bin/env python3
"""LMEval adapter main entry point for evalhub.

Configuration is provided via a job spec JSON file (see `meta/job.json` for an example).
The callback_url in the job spec defines where status updates and results are sent.

Required environment variables:
- REGISTRY_URL: OCI registry URL

Optional environment variables:
- EVALHUB_JOB_SPEC_PATH: path to the job spec JSON (defaults to `meta/job.json`)
- REGISTRY_USERNAME: Registry username (optional)
- REGISTRY_PASSWORD: Registry password/token (optional)
"""

import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager


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
    Adapter-specific params (batch_size, tokenizer, parameters) come from benchmark_config.

    Returns:
        (model_backend, model_args, gen_kwargs)
    """
    model_spec = job_spec.model
    model_name = model_spec.name
    benchmark_config = job_spec.benchmark_config

    # Adapter-specific settings from benchmark_config
    batch_size = int(benchmark_config.get("batch_size", 1))
    timeout_seconds = int(job_spec.timeout_seconds or 300)

    # Optional generation parameters for generate_until tasks.
    parameters = benchmark_config.get("parameters", {})
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
    # The tokenizer can be specified in benchmark_config, otherwise use model.name
    tokenizer = str(benchmark_config.get("tokenizer", model_name))

    # Helpful error message if tokenizer is not a valid HF model
    if tokenizer == model_name and "/" not in tokenizer:
        logger.warning(
            f"Model name '{model_name}' may not be a valid HuggingFace tokenizer. "
            f"Specify the actual model in benchmark_config.tokenizer "
            f"(e.g., 'google/flan-t5-small' or 'meta-llama/Llama-3.1-8B-Instruct')"
        )

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
            job_id = config.id
            benchmark_id = config.benchmark_id
            model_name = config.model.name

            # Number of examples from top-level JobSpec field
            # (extracted from benchmark_config by the service)
            num_examples = config.num_examples

            # Adapter-specific params from benchmark_config
            benchmark_cfg = config.benchmark_config
            num_fewshot = int(benchmark_cfg.get("num_few_shot", 0))
            random_seed = int(benchmark_cfg.get("random_seed", 42))

            model_backend, model_args, gen_kwargs = build_lmeval_config(config)

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

            # Build evaluation results
            evaluation_results = []
            overall_score = None

            for metric_name, metric_value in task_results.items():
                if metric_name.endswith(",none"):
                    # Primary metric (usually accuracy or similar)
                    clean_metric = metric_name.replace(",none", "")
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

            # Create OCI artifact
            oci_spec = OCIArtifactSpec(
                files=[results_file],
                base_path=output_dir,
                title=f"LMEval Results - {benchmark_id}",
                description=f"Evaluation results for {model_name} on {benchmark_id}",
                annotations={
                    "org.opencontainers.image.created": datetime.now(UTC).isoformat(),
                    "org.evalhub.benchmark": benchmark_id,
                    "org.evalhub.model": model_name,
                    "org.evalhub.job_id": job_id,
                },
                id=job_id,
                benchmark_id=benchmark_id,
                model_name=model_name,
            )

            oci_result = callbacks.create_oci_artifact(oci_spec)
            job_results.oci_artifact = oci_result
            logger.info(f"OCI artifact created: {oci_result.reference}")

            # Return results (will be reported by entrypoint)
            return job_results

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)

            # Report failure
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    phase=JobPhase.COMPLETED,
                    progress=0.0,
                    message=_status_message(
                        "Evaluation failed", code="evaluation_failed"
                    ),
                    error=ErrorInfo(
                        message=str(e),
                        message_code="evaluation_failed",
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
        few_shot = adapter.job_spec.benchmark_config.get("num_few_shot")
        logger.info(f"  Few-shot: {few_shot}")
        logger.info("=" * 80)
        logger.info(f"Callback URL: {adapter.job_spec.callback_url}")
        logger.info(f"Provider ID: lm_evaluation_harness")
        logger.info(f"OCI registry configured: {adapter.settings.registry_url}")
        logger.info("=" * 80)

        # Initialize callbacks using job spec callback_url and adapter settings
        callbacks = DefaultCallbacks(
            job_id=adapter.job_spec.id,
            benchmark_id=adapter.job_spec.benchmark_id,
            sidecar_url=adapter.job_spec.callback_url,
            registry_url=adapter.settings.registry_url,
            registry_username=adapter.settings.registry_username,
            registry_password=adapter.settings.registry_password,
            insecure=bool(adapter.settings.registry_insecure),
        )

        # Run evaluation
        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

        logger.info("=" * 80)
        logger.info("Evaluation completed successfully")
        logger.info(f"Overall score: {results.overall_score}")
        logger.info(f"Examples evaluated: {results.num_examples_evaluated}")
        logger.info(f"Duration: {results.duration_seconds:.2f}s")
        logger.info("=" * 80)

        # Report final results
        callbacks.report_results(results)

        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
