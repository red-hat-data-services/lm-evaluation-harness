# Partial tinyTruthfulQA runs in EvalHub

The standard `tinyTruthfulQA` gpirt estimate requires exactly 100 predefined
anchor outcomes in their corresponding order. A smaller evaluation can still
execute successfully without producing that estimate.

The EvalHub adapter handles this automatically for `tinyTruthfulQA`. It checks
the actual number of per-document scores passed to aggregation, rather than the
requested `num_examples` limit:

- Exactly 100 outcomes: call the existing standard aggregation without changing
  its inputs or calculation.
- Any other positive count: skip the standard aggregation and return completed
  sample execution without official metrics or an overall score.
- No completed outcomes, dataset/model errors, or errors from the standard
  estimator on a complete input: fail the job normally.

No smoke flag or special EvalHub fixture is required. A request such as
`parameters.num_examples: 1` retains its existing meaning as an execution limit;
the adapter separately reports whether the standard estimate could be computed.
For example, requesting 100 items from a dataset that yields 99 completed outcomes
reports the actual 99 and skips the estimate.

For an incomplete input, `additional_info` includes:

```json
{
  "num_examples_evaluated": 1,
  "aggregation": {
    "status": "skipped",
    "required_examples": 100,
    "reason": "The standard tinyTruthfulQA gpirt estimate requires exactly 100 predefined anchor outcomes in order. Sample execution completed; no official score was calculated."
  }
}
```

`completed` confirms sample execution and result delivery; the skipped-aggregation
information makes clear that an official benchmark estimate was not produced.
No missing outcomes are padded and no zero or other placeholder score is emitted.

The adapter wraps only the aggregation on its own tinyTruthfulQA task instance.
It does not change other benchmarks, the shared task YAML, the standalone
lm-evaluation-harness CLI, or the tinyBenchmarks estimator. Existing EvalHub smoke
completion/result-presence checks can accept this scoreless result. A rebuilt
adapter image containing this implementation is needed to use the behavior in a
cluster; a new fixture or test-code checkout alone does not update that image.
