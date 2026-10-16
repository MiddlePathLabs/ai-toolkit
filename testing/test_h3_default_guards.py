"""prepare_h3_default_guards: MiniMax-H3 LoRAs skip the token refiner and
blocks 0-5 by default (opt back in with train_token_refiner / train_low_blocks),
and train_blocks + train_token_refiner fails closed."""

from types import SimpleNamespace

import pytest

from toolkit.h3_train_blocks import prepare_h3_default_guards, prepare_train_blocks


def _sd(arch="minimax_h3"):
    return SimpleNamespace(arch=arch)


def _nc(**kw):
    base = dict(train_blocks=None, train_token_refiner=False,
                train_low_blocks=False, type="lora")
    base.update(kw)
    return SimpleNamespace(**base)


def test_h3_defaults_exclude_refiner_and_low_blocks():
    kwargs = prepare_h3_default_guards(_nc(), _sd(), {})
    ignore = kwargs["ignore_if_contains"]
    assert "token_refiner" in ignore
    for i in range(6):
        assert f"blocks.{i}." in ignore
    # guard must not touch anything outside the window
    assert "blocks.6." not in ignore
    assert "blocks.20." not in ignore


def test_opt_outs_leave_the_run_unscoped():
    nc = _nc(train_token_refiner=True, train_low_blocks=True)
    kwargs = prepare_h3_default_guards(nc, _sd(), {})
    assert "ignore_if_contains" not in kwargs


def test_non_h3_arch_untouched():
    for arch in ("krea2", "qwen_image_2", "flux3", None):
        kwargs = prepare_h3_default_guards(_nc(), _sd(arch), {})
        assert kwargs == {}


def test_train_blocks_window_bypasses_guards():
    nc = _nc(train_blocks="20-49")
    kwargs = prepare_h3_default_guards(nc, _sd(), {})
    assert kwargs == {}


def test_existing_ignore_entries_are_deduped():
    kwargs = prepare_h3_default_guards(_nc(), _sd(), {"ignore_if_contains": ["token_refiner", "blocks.0."]})
    ignore = kwargs["ignore_if_contains"]
    assert ignore.count("token_refiner") == 1
    assert ignore.count("blocks.0.") == 1
    assert "blocks.1." in ignore  # the rest still added


def test_guard_words_match_dotted_module_paths_but_not_neighbors():
    # the words must hit the real clean_name form and nothing else
    assert "token_refiner" in "lora_transformer.token_refiner.blocks.0.ff"
    assert "blocks.0." in "lora_transformer.blocks.0.attn.qkv"
    assert "blocks.0." not in "lora_transformer.blocks.10.attn.qkv"
    assert "blocks.0." not in "lora_transformer.blocks.20.mlp.fc1"
    assert "blocks.5." in "lora_transformer.blocks.5.mlp.fc2"
    assert "blocks.5." not in "lora_transformer.blocks.50.mlp.fc2"


def test_train_blocks_with_refiner_opt_in_fails_closed():
    nc = SimpleNamespace(train_blocks="20-49", train_token_refiner=True,
                         type="lora", only_if_contains=None)
    with pytest.raises(ValueError, match="token refiner"):
        prepare_train_blocks(nc, _sd(), {}, train_text_encoder=False,
                             train_embedding=False)
