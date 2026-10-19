import math
from typing import Union
from torch.distributions import LogNormal
from diffusers import FlowMatchEulerDiscreteScheduler
import torch
import numpy as np
from toolkit.timestep_weighing.default_weighing_scheme import default_weighing_scheme, x0_weighing_scheme


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class CustomFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_noise_sigma = 1.0
        self.timestep_type = "linear"

        with torch.no_grad():
            # create weights for timesteps
            num_timesteps = 1000
            # Bell-Shaped Mean-Normalized Timestep Weighting
            # bsmntw? need a better name

            x = torch.arange(num_timesteps, dtype=torch.float32)
            y = torch.exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)

            # Shift minimum to 0
            y_shifted = y - y.min()

            # Scale to make mean 1
            bsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

            # only do half bell
            hbsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

            # flatten second half to max
            hbsmntw_weighing[num_timesteps //
                             2:] = hbsmntw_weighing[num_timesteps // 2:].max()

            # Create linear timesteps from 1000 to 1
            timesteps = torch.linspace(1000, 1, num_timesteps, device='cpu')

            self.linear_timesteps = timesteps
            self.linear_timesteps_weights = bsmntw_weighing
            self.linear_timesteps_weights2 = hbsmntw_weighing
            pass

    def get_weights_for_timesteps(self, timesteps: torch.Tensor, v2=False, timestep_type="linear", x0_pred=False) -> torch.Tensor:
        # Get the indices of the timesteps. Stochastic train grids (sigmoid,
        # trunc_logit_normal) can repeat a timestep; take the first match so a
        # duplicate resolves deterministically instead of raising.
        step_indices = [(self.timesteps == t).nonzero().view(-1)[0].item()
                        for t in timesteps]

        # Get the weights for the timesteps
        if timestep_type == "weighted":
            weighing_scheme = x0_weighing_scheme if x0_pred else default_weighing_scheme
            weights = torch.tensor(
                [weighing_scheme[i] for i in step_indices],
                device=timesteps.device,
                dtype=timesteps.dtype
            )
        elif v2:
            weights = self.linear_timesteps_weights2[step_indices].flatten()
        else:
            weights = self.linear_timesteps_weights[step_indices].flatten()

        return weights

    def get_sigmas(self, timesteps: torch.Tensor, n_dim, dtype, device) -> torch.Tensor:
        # flow-match sigma is the timestep on the training grid by construction
        # (add_noise uses timesteps / num_train_timesteps). A lookup into
        # self.sigmas by exact timestep match cannot resolve the repeated
        # values stochastic train grids (sigmoid, trunc_logit_normal) contain.
        sigma = (timesteps / self.config.num_train_timesteps).to(device=device, dtype=dtype).reshape(-1)
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma

    def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> torch.Tensor:
        t_01 = (timesteps / 1000).to(original_samples.device)
        # forward ODE
        noisy_model_input = (1.0 - t_01) * original_samples + t_01 * noise
        # reverse ODE
        # noisy_model_input = (1 - t_01) * noise + t_01 * original_samples
        return noisy_model_input

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        return sample

    def set_train_timesteps(
        self,
        num_timesteps,
        device,
        timestep_type='linear',
        latents=None,
        patch_size=1,
        shift_override=None,
        logit_loc=-1.09,
        logit_scale=0.95,
        sigma_cutoff=0.95,
        uniform_frac=0.01,
    ):
        # shift_override replaces self.shift for this training grid only (see
        # toolkit/low_noise_share.py). The scheduler's own shift is untouched.
        if shift_override is not None and timestep_type not in ['flux_shift', 'lumina2_shift', 'shift']:
            raise ValueError(
                f"shift_override needs a shift timestep type, got {timestep_type!r}"
            )
        if timestep_type == 'trunc_logit_normal':
            # the trunc_logit_normal params only apply to that timestep type
            if not 0.0 < float(sigma_cutoff) < 1.0:
                raise ValueError(f"sigma_cutoff must be in (0, 1), got {sigma_cutoff!r}")
            if float(logit_scale) <= 0.0:
                raise ValueError(f"logit_scale must be positive, got {logit_scale!r}")
            if not 0.0 <= float(uniform_frac) < 1.0:
                raise ValueError(f"uniform_frac must be in [0, 1), got {uniform_frac!r}")
        self.timestep_type = timestep_type
        if timestep_type == 'linear' or timestep_type == 'weighted':
            timesteps = torch.linspace(1000, 1, num_timesteps, device=device)
            self.timesteps = timesteps
            return timesteps
        elif timestep_type == 'sigmoid':
            # distribute them closer to center. Inference distributes them as a bias toward first
            # Generate values from 0 to 1
            t = torch.sigmoid(torch.randn((num_timesteps,), device=device))

            # Scale and reverse the values to go from 1000 to 0
            timesteps = ((1 - t) * 1000)

            # Sort the timesteps in descending order
            timesteps, _ = torch.sort(timesteps, descending=True)

            self.timesteps = timesteps.to(device=device)

            return timesteps
        elif timestep_type in ['flux_shift', 'lumina2_shift', 'shift']:
            # matches inference dynamic shifting
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(
                    self.sigma_min), num_timesteps
            )

            sigmas = timesteps / self.config.num_train_timesteps

            if self.config.use_dynamic_shifting:
                if shift_override is not None:
                    raise ValueError(
                        "shift_override is for static-shift schedulers; this one "
                        "uses dynamic shifting"
                    )
                if latents is None:
                    raise ValueError('latents is None')

                # for flux we double up the patch size before sending her to simulate the latent reduction
                h = latents.shape[2]
                w = latents.shape[3]
                image_seq_len = h * w // (patch_size**2)

                mu = calculate_shift(
                    image_seq_len,
                    self.config.get("base_image_seq_len", 256),
                    self.config.get("max_image_seq_len", 4096),
                    self.config.get("base_shift", 0.5),
                    self.config.get("max_shift", 1.16),
                )
                sigmas = self.time_shift(mu, 1.0, sigmas)
            else:
                shift = self.shift if shift_override is None else float(shift_override)
                sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

            if self.config.shift_terminal:
                sigmas = self.stretch_shift_to_terminal(sigmas)

            if self.config.use_karras_sigmas:
                sigmas = self._convert_to_karras(
                    in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)
            elif self.config.use_exponential_sigmas:
                sigmas = self._convert_to_exponential(
                    in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)
            elif self.config.use_beta_sigmas:
                sigmas = self._convert_to_beta(
                    in_sigmas=sigmas, num_inference_steps=self.config.num_train_timesteps)

            sigmas = torch.from_numpy(sigmas).to(
                dtype=torch.float32, device=device)
            timesteps = sigmas * self.config.num_train_timesteps

            if self.config.invert_sigmas:
                sigmas = 1.0 - sigmas
                timesteps = sigmas * self.config.num_train_timesteps
                sigmas = torch.cat(
                    [sigmas, torch.ones(1, device=sigmas.device)])
            else:
                sigmas = torch.cat(
                    [sigmas, torch.zeros(1, device=sigmas.device)])

            self.timesteps = timesteps.to(device=device)
            self.sigmas = sigmas

            self.timesteps = timesteps.to(device=device)
            return timesteps

        elif timestep_type == 'lognorm_blend':
            # disgtribute timestepd to the center/early and blend in linear
            alpha = 0.75

            lognormal = LogNormal(loc=0, scale=0.333)

            # Sample from the distribution
            t1 = lognormal.sample((int(num_timesteps * alpha),)).to(device)

            # Scale and reverse the values to go from 1000 to 0
            t1 = ((1 - t1/t1.max()) * 1000)

            # add half of linear
            t2 = torch.linspace(1000, 1, int(
                num_timesteps * (1 - alpha)), device=device)
            timesteps = torch.cat((t1, t2))

            # Sort the timesteps in descending order
            timesteps, _ = torch.sort(timesteps, descending=True)

            timesteps = timesteps.to(torch.int)
            self.timesteps = timesteps.to(device)
            return timesteps
        elif timestep_type == 'trunc_logit_normal':
            # MOSS-Image's pretraining sigma draw (technical report Appendix A.2, Table 6):
            # sigma = sigmoid(-Z) with Z ~ N(logit_loc, logit_scale^2), truncated to
            # sigma <= sigma_cutoff by inverse CDF, rescaled by / sigma_cutoff onto the
            # full grid, snapped to the nearest-above j/num_train_timesteps cell, with a
            # uniform_frac share of uniform draws mixed in. Defaults are the report's
            # final pretraining stage (loc -1.09, scale 0.95, cutoff 0.95, 1% uniform);
            # the distribution is noise-tilted (median sigma ~0.8), unlike 'sigmoid'
            # (loc 0, scale 1, untruncated, unsnapped, no uniform tail).
            total = int(self.config.num_train_timesteps)
            # drawn and computed on the CPU in float64: MPS has no float64 tensors,
            # and the truncated tail needs float64's range. The result moves to the
            # requested device at the end.
            u = torch.rand((num_timesteps, 2))
            mix, pos = u[:, 0], u[:, 1].double()

            def cdf(x):
                # P(S <= x) for the untruncated sigmoid(-N(logit_loc, logit_scale^2)),
                # in survival-function form. 1 - Phi(...) cancels to 0 in the far
                # tail (extreme loc/cutoff put F(cutoff) below float64 only near
                # 1e-300); erfc keeps it representable down to double precision.
                return 0.5 * math.erfc(
                    (math.log((1.0 - x) / x) - logit_loc) / (logit_scale * math.sqrt(2.0))
                )

            f_cutoff = cdf(float(sigma_cutoff))
            p = (pos * f_cutoff).clamp_min(1e-300)
            # inverse through the same survival side: F(s) = p with s = sigmoid(-Z)
            # gives logit(s) = logit_scale * ndtri(p) - logit_loc. ndtri(p), not
            # ndtri(1 - p): 1 - p rounds to 1 for small p and returns inf.
            s = torch.sigmoid(logit_scale * torch.special.ndtri(p) - logit_loc)
            # rescale [0, cutoff] -> [0, 1], then snap up to the grid cell j/total
            snapped = (s / float(sigma_cutoff) * total).ceil().clamp(1, total)
            uniform = (pos * total).floor() + 1.0
            steps = torch.where(mix < float(uniform_frac), uniform, snapped)
            timesteps, _ = torch.sort(steps, descending=True)
            timesteps = timesteps.to(torch.int)
            self.timesteps = timesteps.to(device)
            return timesteps
        else:
            raise ValueError(f"Invalid timestep type: {timestep_type}")
