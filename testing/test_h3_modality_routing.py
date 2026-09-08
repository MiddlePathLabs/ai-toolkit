import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from toolkit.config_modules import TrainConfig
from toolkit.h3_modality_routing import (
    ModalityBlockRouter,
    bind_modality_router,
    build_trunk_param_index,
    classify_file_item,
    format_block_spec,
    parse_block_spec,
    uses_modality_block_routing,
    window_kind,
)


def test_train_config_routing_defaults_off():
    cfg = TrainConfig()
    assert cfg.modality_block_routing.enabled is False
    assert cfg.modality_block_routing.photo_blocks is None
    assert uses_modality_block_routing(cfg) is False
    cfg = TrainConfig(modality_block_routing={"photo_blocks": "20-49"})
    assert uses_modality_block_routing(cfg) is True
    assert cfg.modality_block_routing.photo_blocks == "20-49"
    assert cfg.modality_block_routing.clip_blocks is None


def test_parse_block_spec_ranges_and_singles():
    assert parse_block_spec("20-49") == list(range(20, 50))
    assert parse_block_spec("3-12, 14-15, 22,27,31-33") == (
        list(range(3, 13)) + [14, 15, 22, 27, 31, 32, 33]
    )
    assert format_block_spec(parse_block_spec("20-22,24")) == "20-22,24"


def test_parse_block_spec_rejects_typos():
    with pytest.raises(ValueError, match="no blocks"):
        parse_block_spec("")
    with pytest.raises(ValueError, match="backwards"):
        parse_block_spec("49-20")
    with pytest.raises(ValueError, match="duplicate"):
        parse_block_spec("20-22,21")
    with pytest.raises(ValueError, match="duplicate"):
        parse_block_spec("20-49, 20-49")
    with pytest.raises(ValueError, match="cannot read"):
        parse_block_spec("20-foo")
    with pytest.raises(ValueError, match="do not exist"):
        parse_block_spec("20-49", num_blocks=40)
    with pytest.raises(ValueError, match="do not exist"):
        parse_block_spec("50", num_blocks=50)


def test_classify_file_item_photo_clip_voice():
    assert classify_file_item(SimpleNamespace(is_video=False)) == "photo"
    assert classify_file_item(SimpleNamespace(is_video=True)) == "clip"
    assert classify_file_item(SimpleNamespace(is_video=True, is_audio_only=True)) == "voice"
    assert window_kind({"photo"}) == "photo"
    assert window_kind({"photo", "clip"}) == "mixed"
    assert window_kind(set()) == "mixed"


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)


class _Transformer(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.token_refiner = nn.Linear(4, 4, bias=False)
        self.blocks = nn.ModuleList([_Block() for _ in range(n)])


class _Adapter(nn.Module):
    def __init__(self, org, lora_name):
        super().__init__()
        self.lora_name = lora_name
        self.orig_module_ref = weakref.ref(org)
        self.down = nn.Linear(4, 2, bias=False)


class _Network:
    def __init__(self, transformer: _Transformer):
        self.unet_loras = [
            _Adapter(transformer.blocks[0].linear, "lycoris_lora_blocks$$0$$linear"),
            _Adapter(transformer.blocks[1].linear, "lycoris_lora_blocks$$1$$linear"),
            _Adapter(transformer.blocks[2].linear, "lycoris_lora_blocks$$2$$linear"),
            _Adapter(transformer.token_refiner, "lycoris_lora_token_refiner_blocks_0"),
        ]
        self.text_encoder_loras = []


class _H3:
    arch = "minimax_h3"

    def __init__(self, transformer):
        self.model = transformer


def test_trunk_index_uses_orig_module_ref_not_lora_name():
    transformer = _Transformer(4)
    network = _Network(transformer)
    by_block, always_active = build_trunk_param_index(transformer, network)
    assert set(by_block) == {0, 1, 2}
    # PEFT $$ names would have been unparsable; ownership still follows the live module.
    assert list(network.unet_loras[0].parameters()) == by_block[0]
    refiner_params = list(network.unet_loras[3].parameters())
    assert refiner_params == always_active


def test_photo_only_window_keeps_refiners_and_allowed_blocks():
    transformer = _Transformer(4)
    network = _Network(transformer)
    by_block, always_active = build_trunk_param_index(transformer, network)
    all_params = always_active + [p for ps in by_block.values() for p in ps]
    router = ModalityBlockRouter(
        photo_blocks={2, 3},
        clip_blocks=None,
        voice_blocks=None,
        by_block=by_block,
        always_active=always_active,
        all_params=all_params,
    )
    photo_batch = SimpleNamespace(file_items=[SimpleNamespace(is_video=False)])
    clip_batch = SimpleNamespace(file_items=[SimpleNamespace(is_video=True)])
    mixed_batch = SimpleNamespace(
        file_items=[SimpleNamespace(is_video=False), SimpleNamespace(is_video=True)]
    )
    photo_active = router.active_params_for_batch(photo_batch)
    assert photo_active is not None
    photo_ids = {id(p) for p in photo_active}
    assert {id(p) for p in by_block[2]}.issubset(photo_ids)
    assert {id(p) for p in always_active}.issubset(photo_ids)
    assert {id(p) for p in by_block[0]}.isdisjoint(photo_ids)
    assert {id(p) for p in by_block[1]}.isdisjoint(photo_ids)
    assert router.active_params_for_batch(clip_batch) is None
    assert router.active_params_for_batch(mixed_batch) is None


def test_clip_blocks_null_leaves_clips_unrestricted():
    transformer = _Transformer(4)
    network = _Network(transformer)
    router = bind_modality_router(
        TrainConfig(modality_block_routing={"photo_blocks": "2-3"}),
        _H3(transformer),
        network,
    )
    assert router is not None
    assert router.allowed["photo"] == {2, 3}
    assert router.allowed["clip"] is None
    clip_batch = SimpleNamespace(file_items=[SimpleNamespace(is_video=True)])
    assert router.active_params_for_batch(clip_batch) is None


def test_bind_fails_on_non_h3_arch():
    transformer = _Transformer(4)
    network = _Network(transformer)

    class Other:
        arch = "flux"

        def __init__(self):
            self.model = transformer

    with pytest.raises(ValueError, match="H3-only"):
        bind_modality_router(
            TrainConfig(modality_block_routing={"photo_blocks": "0-1"}),
            Other(),
            network,
        )


def test_bind_off_returns_none():
    assert bind_modality_router(TrainConfig(), _H3(_Transformer()), _Network(_Transformer())) is None


def test_observe_and_consume_mixed_window_is_unrestricted():
    transformer = _Transformer(4)
    network = _Network(transformer)
    router = bind_modality_router(
        TrainConfig(modality_block_routing={"photo_blocks": "2"}),
        _H3(transformer),
        network,
    )
    router.observe_batch(SimpleNamespace(file_items=[SimpleNamespace(is_video=False)]))
    router.observe_batch(SimpleNamespace(file_items=[SimpleNamespace(is_video=True)]))
    assert router.consume_window() is None


def test_weight_noise_skips_inactive_ids():
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

    torch.manual_seed(0)
    tagged_active = torch.nn.Parameter(torch.ones(8))
    tagged_active._is_lora = True
    tagged_inactive = torch.nn.Parameter(torch.ones(8))
    tagged_inactive._is_lora = True
    trainer = SDTrainer.__new__(SDTrainer)
    trainer.params = [tagged_active, tagged_inactive]
    trainer.train_config = TrainConfig(
        weight_noise={"enabled": True, "mode": "absolute", "sigma": 0.5, "log_every": 0}
    )
    trainer.step_num = 0
    trainer.optimizer = None
    trainer._post_step_active_ids = frozenset({id(tagged_active)})
    before_inactive = tagged_inactive.detach().clone()
    trainer._inject_weight_noise()
    assert not torch.equal(tagged_active, torch.ones_like(tagged_active))
    assert torch.equal(tagged_inactive, before_inactive)
