"""Empty sample prompts are skipped, not sampled.

`SampleConfig` used to pass `""`/whitespace prompts straight through, so every
preview cycle burned a generation slot on a blank prompt. Worse, the sample
loop pairs `prompts[i]` with `samples[i]` by index while the `prompts` property
dropped `None` entries — any dropped or empty item desynced that pairing and
silently misassigned seeds/settings to the wrong prompt. Both are fixed by
filtering at construction time.
"""
from toolkit.config_modules import SampleConfig


def test_legacy_empty_prompts_are_skipped():
    config = SampleConfig(prompts=["a photo of sks", "", "   ", "a dog"])
    assert config.prompts == ["a photo of sks", "a dog"]
    # samples stay index-aligned with prompts
    assert len(config.samples) == len(config.prompts)


def test_sample_items_with_empty_prompts_are_dropped():
    config = SampleConfig(
        samples=[
            {"prompt": "a cat", "seed": 11},
            {"prompt": "", "seed": 22},
            {"prompt": None, "seed": 33},
            {"prompt": "  ", "seed": 44},
            {"prompt": "a dog", "seed": 55},
        ]
    )
    assert config.prompts == ["a cat", "a dog"]
    # the surviving items keep their own settings, proving no index shift
    assert [(s.prompt, s.seed) for s in config.samples] == [("a cat", 11), ("a dog", 55)]


def test_promptless_sample_items_no_longer_desync_indices():
    # a None-prompt item used to occupy a samples slot while being invisible
    # to the prompts property, shifting samples[i] off prompts[i]
    config = SampleConfig(
        samples=[
            {"prompt": None},
            {"prompt": "the real prompt", "seed": 9},
        ]
    )
    assert config.prompts == ["the real prompt"]
    assert config.samples[0].prompt == "the real prompt"
    assert config.samples[0].seed == 9


def test_all_prompts_empty_means_no_samples():
    config = SampleConfig(prompts=["", "   "])
    assert config.prompts == []
    assert config.samples == []


def test_no_prompt_key_in_sample_item_is_dropped():
    # SampleItem expects a prompt in kwargs; items without one can never be
    # sampled, so they must not consume an index slot
    config = SampleConfig(samples=[{"seed": 1}, {"prompt": "real", "seed": 2}])
    assert config.prompts == ["real"]
    assert [s.seed for s in config.samples] == [2]


def test_control_image_item_keeps_empty_prompt():
    # the control image carries the conditioning, so "" is a valid prompt
    # for these; dropping them removed previously supported previews
    config = SampleConfig(
        samples=[
            {"prompt": "a cat"},
            {"prompt": "", "ctrl_img": "input.png", "seed": 7},
            {"prompt": "   ", "ctrl_img_2": "depth.png", "seed": 8},
        ]
    )
    assert config.prompts == ["a cat", "", ""]
    assert len(config.samples) == 3
    assert config.samples[1].ctrl_img == "input.png"
    assert config.samples[1].seed == 7
    assert config.samples[2].ctrl_img_2 == "depth.png"


def test_control_image_item_without_prompt_key_is_kept_and_normalized():
    # prompt None + ctrl image also stays; the prompt normalizes to "" so
    # prompt injection downstream never sees None ("" was the value the old
    # pass-through behavior fed the pipeline)
    config = SampleConfig(samples=[{"ctrl_img": "input.png"}, {"prompt": "real"}])
    assert config.prompts == ["", "real"]
    assert config.samples[0].prompt == ""
    assert config.samples[0].ctrl_img == "input.png"


def test_empty_prompt_without_control_image_still_dropped():
    config = SampleConfig(
        samples=[
            {"prompt": "", "seed": 1},
            {"prompt": "   "},
            {"prompt": "real", "seed": 3},
        ]
    )
    assert config.prompts == ["real"]
    assert [s.seed for s in config.samples] == [3]
