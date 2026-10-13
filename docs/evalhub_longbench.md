# EvalHub LongBench v2 context configuration

The harness task/group ID is `longbench2`, and its primary metric is `acc`.
This implementation uses multiple-choice loglikelihood scoring. It is not the
official LongBench runner's answer-generation protocol; scores should not be
assumed comparable to its leaderboard.

Set `max_length` in the benchmark's job parameters to the context limit actually
served by the endpoint. The adapter forwards this value to the API backend.
The backend reserves one output token, so prompt plus answer-choice tokens must
fit within `max_length - 1`. Rendered chat-template tokens are part of the prompt.
The model's advertised context capacity can exceed the server's configured limit.

```json
{
  "id": "longbench2",
  "provider_id": "lm_evaluation_harness",
  "parameters": {
    "max_length": 37568,
    "loglikelihood_context_overflow": "error",
    "num_fewshot": 0,
    "apply_chat_template": true,
    "tokenizer": "meta-llama/Llama-3.3-70B-Instruct"
  }
}
```

This is an evaluation-request benchmark fragment. The context value is an
example server configuration, not a general Llama or LongBench default. Use the
tokenizer matching the served model. This requires an adapter image containing
the context and chat-template changes; older images can ignore these settings.

For `longbench2` and its leaf tasks, `max_length` is required and
`loglikelihood_context_overflow` defaults to `error`. An oversized likelihood
input raises an error before that input is sent to the model. Previously
processed inputs may already have reached the endpoint; no final successful
benchmark score is returned for the failed run. Increasing the client limit
does not increase the server's capacity.

Explicitly selecting `loglikelihood_context_overflow: "truncate"` retains the
backend's existing left-truncation behavior. This evaluates shortened inputs;
document that protocol when reporting scores. It is not full-context validation.
The setting controls likelihood requests, not generation requests. Other tasks
retain their existing default context/truncation behavior unless overridden.
The selected limit and policy are retained in `additional_info`.

Group results use the harness's aggregate, preserving `weight_by_size`.
`num_examples_evaluated` counts samples across the selected leaf tasks, including
nested groups. When sample logs are unavailable, the harness's effective sample
counts are used. A request limit applies per leaf task, not to the group as a
whole.

Tests in `tests/test_adapter_longbench.py` exercise budgets above the old 2K
default, boundary and overflow handling, synchronous/asynchronous rejection,
weighted group results, and sample coverage. These are local regression tests;
they do not establish that a deployed image or a full dataset run is validated.

Sources: [harness task definition](../lm_eval/tasks/longbench2/_longbench_common_yaml),
[group aggregation](../lm_eval/tasks/longbench2/_longbench2.yaml), and the
[official LongBench v2 evaluation instructions](https://github.com/THUDM/LongBench#evaluation).
