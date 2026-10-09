# Chat templates in the EvalHub adapter

For an instruction-tuned model, opt in to the model tokenizer's chat template
with `apply_chat_template` in the adapter's top-level job parameters:

```json
{
  "parameters": {
    "apply_chat_template": true,
    "tokenizer": "meta-llama/Llama-3.3-70B-Instruct",
    "parameters": {
      "temperature": 0,
      "max_gen_toks": 2048
    }
  }
}
```

The outer `parameters` object is `JobSpec.parameters`. The nested `parameters`
object holds generation settings. `apply_chat_template` must be a JSON boolean;
it defaults to `false`, preserving existing plain-prompt evaluations.

The adapter passes this option to `simple_evaluate`. The `local-completions`
backend renders the tokenizer's role and turn delimiters as a string, including
the assistant generation prompt. Requests continue to use `/v1/completions`
with `tokenized_requests=false`. The tokenizer must provide a suitable chat
template; use the matching tokenizer and satisfy any gated-repository access
requirements.

This supports both generation tasks such as IFEval and multiple-choice
loglikelihood tasks such as ToxiGen and TruthfulQA MC1. Candidate continuations
are scored after the rendered context; the option does not change the task's
scorer or metric names. The separate chat-completions backend retains its
role/content message payloads. The adapter records the requested option in its
additional run metadata.

Generation limits, few-shot counts, and benchmark scope remain explicit job
settings. Enabling the template does not make an evaluation an exact
reproduction of a published result. Retest generation and multiple-choice tasks
with the rebuilt adapter image before starting a full run.
