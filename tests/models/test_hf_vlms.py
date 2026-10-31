"""Regression tests for Hugging Face multimodal model registration."""

import transformers

from lm_eval.models.hf_vlms import HFMultimodalLM


def test_hf_multimodal_uses_transformers_image_text_auto_class() -> None:
    assert HFMultimodalLM.AUTO_MODEL_CLASS is transformers.AutoModelForImageTextToText
