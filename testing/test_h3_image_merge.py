"""H3-Image must retain its audio-free contract through the fork's forward path."""
from types import SimpleNamespace

import torch

from extensions_built_in.diffusion_models.minimax_h3.minimax_h3_image import (
    H3_IMAGE_SAMPLE_SIGMAS,
    MinimaxH3ImageModel,
)
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds


def test_image_training_forward_packs_no_audio_rows():
    class RecordingTransformer:
        device = torch.device("cpu")

        def to(self, device):
            self.device = device

        def __call__(self, **kwargs):
            assert kwargs["audio_hidden_states"].shape == (1, 0, 32)
            assert kwargs["audio_indices"].numel() == 0
            assert not (kwargs["token_tags"] == 2).any()
            assert kwargs["pdd_weights"][0].shape == (1, 6)
            return torch.zeros_like(kwargs["hidden_states"]), torch.zeros(1, 0, 32)

    model = object.__new__(MinimaxH3ImageModel)
    model.device_torch = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.model_config = SimpleNamespace(model_kwargs={})
    model.vae = SimpleNamespace(audio_vae=None)
    model.model = RecordingTransformer()
    model.max_text_length = 4096
    embeds = AdvancedPromptEmbeds(
        text_embeds=[torch.zeros(3, 8)],
        text_token_tags=[torch.ones(3, dtype=torch.long)],
    )
    latents = torch.randn(1, 24, 1, 4, 4)
    prediction = model.get_noise_prediction(latents, torch.tensor([750.0]), embeds)
    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape == latents.shape
    assert torch.isfinite(prediction).all()


def test_image_training_scheduler_keeps_the_six_sampling_timesteps():
    scheduler = MinimaxH3ImageModel.get_train_scheduler()
    scheduler.set_train_timesteps(1000, device="cpu", timestep_type="sigmoid")
    expected = torch.tensor(H3_IMAGE_SAMPLE_SIGMAS)
    assert torch.allclose(scheduler.sigmas, expected)
    assert torch.allclose(scheduler.timesteps, expected[:-1] * 1000)
