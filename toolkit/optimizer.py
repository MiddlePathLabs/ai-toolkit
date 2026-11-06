import torch


def get_optimizer(
        params,
        optimizer_type='adam',
        learning_rate=1e-6,
        optimizer_params=None
):
    # Merge factory defaults before forwarding user overrides.  Passing a
    # hard-coded keyword together with optimizer_params used to raise
    # ``multiple values for keyword`` and made epsilon impossible to tune.
    optimizer_params = dict(optimizer_params or {})

    def _defaults(**defaults):
        merged = dict(defaults)
        merged.update(optimizer_params)
        return merged

    lower_type = optimizer_type.lower()
    if lower_type.startswith("dadaptation"):
        # dadaptation uses a larger learning-rate range than Adam.
        import dadaptation
        print("Using DAdapt optimizer")
        use_lr = learning_rate if learning_rate >= 0.1 else 1.0
        if lower_type.endswith("lion"):
            optimizer = dadaptation.DAdaptLion(
                params, lr=use_lr, **_defaults(eps=1e-6)
            )
        elif lower_type.endswith("adam"):
            optimizer = dadaptation.DAdaptAdam(
                params, lr=use_lr, **_defaults(eps=1e-6)
            )
        elif lower_type == "dadaptation":
            optimizer = dadaptation.DAdaptAdam(
                params, lr=use_lr, **_defaults(eps=1e-6)
            )
            print("WARNING: Dadaptation optimizer type has been changed to DadaptationAdam. Please update your config.")
        else:
            raise ValueError(f'Unknown optimizer type {optimizer_type}')
    elif lower_type.startswith("prodigy8bit"):
        from toolkit.optimizers.prodigy_8bit import Prodigy8bit
        print("Using Prodigy optimizer")
        use_lr = learning_rate if learning_rate >= 0.1 else 1.0
        optimizer = Prodigy8bit(
            params, lr=use_lr, **_defaults(eps=1e-6)
        )
    elif lower_type.startswith("prodigy"):
        from prodigyopt import Prodigy
        print("Using Prodigy optimizer")
        use_lr = learning_rate if learning_rate >= 0.1 else 1.0
        optimizer = Prodigy(
            params, lr=use_lr, **_defaults(eps=1e-6)
        )
    elif lower_type == "adam8":
        from toolkit.optimizers.adam8bit import Adam8bit
        optimizer = Adam8bit(
            params, lr=learning_rate, **_defaults(eps=1e-6)
        )
    elif lower_type == "adamw8":
        from toolkit.optimizers.adam8bit import Adam8bit
        optimizer = Adam8bit(
            params, lr=learning_rate, **_defaults(eps=1e-6, decouple=True)
        )
    elif lower_type.endswith("8bit"):
        import bitsandbytes
        if lower_type == "adam8bit":
            optimizer = bitsandbytes.optim.Adam8bit(
                params, lr=learning_rate, **_defaults(eps=1e-6)
            )
        elif lower_type == "ademamix8bit":
            optimizer = bitsandbytes.optim.AdEMAMix8bit(
                params, lr=learning_rate, **_defaults(eps=1e-6)
            )
        elif lower_type == "adamw8bit":
            optimizer = bitsandbytes.optim.AdamW8bit(
                params, lr=learning_rate, **_defaults(eps=1e-6)
            )
        elif lower_type == "lion8bit":
            optimizer = bitsandbytes.optim.Lion8bit(
                params, lr=learning_rate, **optimizer_params
            )
        else:
            raise ValueError(f'Unknown optimizer type {optimizer_type}')
    elif lower_type == 'adam':
        optimizer = torch.optim.Adam(
            params, lr=float(learning_rate), **_defaults(eps=1e-6)
        )
    elif lower_type == 'adamw':
        optimizer = torch.optim.AdamW(
            params, lr=float(learning_rate), **_defaults(eps=1e-6)
        )
    elif lower_type == 'lion':
        try:
            from lion_pytorch import Lion
            optimizer = Lion(params, lr=learning_rate, **optimizer_params)
        except ImportError:
            raise ImportError("Please install lion_pytorch to use Lion optimizer -> pip install lion-pytorch")
    elif lower_type == 'adagrad':
        optimizer = torch.optim.Adagrad(
            params, lr=float(learning_rate), **optimizer_params
        )
    elif lower_type == 'adafactor':
        from toolkit.optimizers.adafactor import Adafactor
        optimizer_params.setdefault('relative_step', False)
        optimizer_params.setdefault('scale_parameter', False)
        optimizer_params.setdefault('warmup_init', False)
        optimizer = Adafactor(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagic':
        from toolkit.optimizers.automagic import Automagic
        optimizer = Automagic(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagic2':
        from toolkit.optimizers.automagic2 import Automagic2
        optimizer = Automagic2(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagic3':
        from toolkit.optimizers.automagic3 import Automagic3
        optimizer = Automagic3(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagic4':
        from toolkit.optimizers.automagic4 import Automagic4
        optimizer = Automagic4(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagicexperiment':
        from toolkit.optimizers.automagicEXPERIMENT import AutomagicEXPERIMENT
        optimizer = AutomagicEXPERIMENT(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'rose':
        # Range-Of-Slice Equilibration (stateless; LR is independently tuned).
        from toolkit.optimizers.rose import Rose
        optimizer = Rose(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'adamconvrot':
        from toolkit.optimizers.adamconvrot import AdamConvRot
        optimizer_params.setdefault('eps', 1e-6)
        optimizer = AdamConvRot(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'muon':
        # Muon — Newton-Schulz orthogonalized momentum.  Defaults are
        # RMS-matched to AdamW (adjust_lr_fn=match_rms_adamw), so existing
        # AdamW learning rates carry over unchanged (the pairing Fizgig
        # measured across their Krea 2 / Qwen 2.1 LoRA presets, v9.0.8).
        # torch.optim.Muon directly when every parameter is 2D (PyTorch
        # 2.10+; this repo pins 2.13); the vendored copy covers non-2D
        # tensors (conv factors, biases, norms).
        params = list(params)
        if params and isinstance(params[0], dict):
            # Materialize each group's params exactly once.  A group may carry
            # a generator (or a bare tensor); the dimension check below would
            # otherwise exhaust the generator and hand the optimizer empty
            # groups — a silent no-update run.  Mirrors the replacement
            # torch's add_param_group performs on these same dicts anyway.
            # Groups without a "params" key stay untouched so torch raises
            # the same KeyError every other factory branch produces.
            for group in params:
                group_params = group.get('params')
                if isinstance(group_params, torch.Tensor):
                    group['params'] = [group_params]
                elif group_params is not None:
                    group['params'] = list(group_params)
            flat = [p for group in params for p in group.get('params', ())]
        else:
            flat = params
        optimizer_params.setdefault('adjust_lr_fn', 'match_rms_adamw')
        optimizer_params.setdefault('weight_decay', 0.01)
        optimizer_params.setdefault('momentum', 0.95)
        optimizer_params.setdefault('nesterov', True)
        optimizer_params.setdefault('ns_steps', 5)
        if hasattr(torch.optim, 'Muon') and all(getattr(p, 'ndim', 0) == 2 for p in flat):
            optimizer = torch.optim.Muon(params, lr=float(learning_rate), **optimizer_params)
        else:
            from toolkit.optimizers.muon import Muon
            optimizer = Muon(params, lr=float(learning_rate), **optimizer_params)
    else:
        raise ValueError(f'Unknown optimizer type {optimizer_type}')
    # New factory branches must be classified in toolkit/optimizer_runtime.py
    # (supports_step_scale / supports_active_param_mask / update_phase)
    # before per_image_adaptive_lr_mode: lr or modality_block_routing.

    return optimizer
