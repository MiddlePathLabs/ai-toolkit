import React from 'react';
import { ConfigDoc } from '@/types';
import { IoFlaskSharp } from 'react-icons/io5';

const docs: { [key: string]: ConfigDoc } = {
  'config.name': {
    title: 'Training Name',
    description: (
      <>
        The name of the training job. This name will be used to identify the job in the system and will the the filename
        of the final model. It must be unique and can only contain alphanumeric characters, underscores, and dashes. No
        spaces or special characters are allowed.
      </>
    ),
  },
  gpuids: {
    title: 'GPU ID',
    description: (
      <>
        This is the GPU that will be used for training. Only one GPU can be used per job at a time via the UI currently.
        However, you can start multiple jobs in parallel, each using a different GPU.
      </>
    ),
  },
  'config.process[0].trigger_word': {
    title: 'Trigger Word',
    description: (
      <>
        Optional: This will be the word or token used to trigger your concept or character.
        <br />
        <br />
        When using a trigger word, If your captions do not contain the trigger word, it will be added automatically the
        beginning of the caption. If you do not have captions, the caption will become just the trigger word. If you
        want to have variable trigger words in your captions to put it in different spots, you can use the{' '}
        <code>{'[trigger]'}</code> placeholder in your captions. This will be automatically replaced with your trigger
        word.
        <br />
        <br />
        Trigger words will not automatically be added to your test prompts, so you will need to either add your trigger
        word manually or use the
        <code>{'[trigger]'}</code> placeholder in your test prompts as well.
      </>
    ),
  },
  'config.process[0].model.name_or_path': {
    title: 'Name or Path',
    description: (
      <>
        The name of a diffusers repo on Huggingface or the local path to the base model you want to train from. The
        folder needs to be in diffusers format for most models. For some models, such as SDXL and SD1, you can put the
        path to an all in one safetensors checkpoint here.
      </>
    ),
  },
  'datasets.control_path': {
    title: 'Control Dataset',
    description: (
      <>
        The control dataset needs to have files that match the filenames of your training dataset. They should be
        matching file pairs. These images are fed as control/input images during training. The control images will be
        resized to match the training images.
      </>
    ),
  },
  'datasets.multi_control_paths': {
    title: 'Multi Control Dataset',
    description: (
      <>
        The control dataset needs to have files that match the filenames of your training dataset. They should be
        matching file pairs. These images are fed as control/input images during training.
        <br />
        <br />
        For multi control datasets, the controls will all be applied in the order they are listed. If the model does not
        require the images to be the same aspect ratios, such as with Qwen/Qwen-Image-Edit-2509, then the control images
        do not need to match the aspect size or aspect ratio of the target image and they will be automatically resized
        to the ideal resolutions for the model / target images.
      </>
    ),
  },
  'datasets.num_frames': {
    title: 'Number of Frames',
    description: (
      <>
        This sets the number of frames to shrink videos to for a video dataset. If this dataset is images, set this to 1
        for one frame. If your dataset is only videos, frames will be extracted evenly spaced from the videos in the
        dataset.
        <br />
        <br />
        It is best to trim your videos to the proper length before training. Wan is 16 frames a second. Doing 81 frames
        will result in a 5 second video. So you would want all of your videos trimmed to around 5 seconds for best
        results.
        <br />
        <br />
        Example: Setting this to 81 and having 2 videos in your dataset, one is 2 seconds and one is 90 seconds long,
        will result in 81 evenly spaced frames for each video making the 2 second video appear slow and the 90second
        video appear very fast.
      </>
    ),
  },
  'datasets.do_i2v': {
    title: 'Do I2V',
    description: (
      <>
        For video models that can handle both I2V (Image to Video) and T2V (Text to Video), this option sets this
        dataset to be trained as an I2V dataset. This means that the first frame will be extracted from the video and
        used as the start image for the video. If this option is not set, the dataset will be treated as a T2V dataset.
      </>
    ),
  },
  'datasets.do_audio': {
    title: 'Do Audio',
    description: (
      <>
        For models that support audio with video, this option will load the audio from the video and resize it to match
        the video sequence. Since the video is automatically resized, the audio may drop or raise in pitch to match the
        new speed of the video. It is important to prep your dataset to have the proper length before training.
      </>
    ),
  },
  'datasets.audio_normalize': {
    title: 'Audio Normalize',
    description: (
      <>
        When loading audio, this will normalize the audio volume to the max peaks. Useful if your dataset has varying
        audio volumes. Warning, do not use if you have clips with full silence you want to keep, as it will raise the
        volume of those clips.
      </>
    ),
  },
  'datasets.audio_preserve_pitch': {
    title: 'Audio Preserve Pitch',
    description: (
      <>
        When loading audio to match the number of frames requested, this option will preserve the pitch of the audio if
        the length does not match training target. It is recommended to have a dataset that matches your target length,
        as this option can add sound distortions.
      </>
    ),
  },
  'datasets.flip': {
    title: 'Flip X and Flip Y',
    description: (
      <>
        You can augment your dataset on the fly by flipping the x (horizontal) and/or y (vertical) axis. Flipping a
        single axis will effectively double your dataset. It will result it training on normal images, and the flipped
        versions of the images. This can be very helpful, but keep in mind it can also be destructive. There is no
        reason to train people upside down, and flipping a face can confuse the model as a person's right side does not
        look identical to their left side. For text, obviously flipping text is not a good idea.
        <br />
        <br />
        Control images for a dataset will also be flipped to match the images, so they will always match on the pixel
        level.
      </>
    ),
  },
  'train.per_image_adaptive_lr': {
    title: 'Per-Image Adaptive LR',
    description: (
      <>
        Tracks each dataset image's loss trend across training. Images that stay hard without
        improving (often a bad or mismatched caption) get their learning rate throttled, escalating
        the longer they stay stuck, so one bad image can't keep yanking the weights all run.
        Consistently healthy images get a small boost. Works for every model architecture and both
        LoKr and LoRA. Needs a few evaluation windows of history before it starts acting. The window
        auto-sizes to your unique image count regardless of resolution list or repeats, so a large
        multi-resolution dataset won't inflate it — but on a short run (e.g. Krea at ~2000 steps) it
        may still be worth lowering the warmup below so it has time to actually do something.
      </>
    ),
  },
  'train.per_image_adaptive_lr_warmup_windows': {
    title: 'Adaptive LR Warmup',
    description: (
      <>
        How many evaluation windows of loss history to collect before throttling/boosting starts.
        Each window is auto-sized to roughly one pass over your unique images (resolution copies
        and repeats don't inflate it). Lower this (e.g. to 1) if the run is short enough that the
        default warmup would eat most of your step budget.
      </>
    ),
  },
  'train.per_image_adaptive_lr_stats_only': {
    title: 'Adaptive LR Stats Only',
    description: (
      <>
        Runs the per-image loss watcher in observation-only mode. The stuck / suspect / exhausted
        / healthy verdicts and per-resolution average loss lines still print each window, but NO
        loss multiplier is ever applied — training is byte-for-byte identical to leaving adaptive
        LR off. Useful for inspecting what the classifier would do on your dataset before
        committing to it. Also applies (stats-only wins) if both this and the live toggle are on.
      </>
    ),
  },
  'train.unload_text_encoder': {
    title: 'Unload Text Encoder',
    description: (
      <>
        Unloading text encoder will cache the trigger word and the sample prompts and unload the text encoder from the
        GPU. Captions in for the dataset will be ignored
      </>
    ),
  },
  'train.cache_text_embeddings': {
    title: 'Cache Text Embeddings',
    description: (
      <>
        <small>(experimental)</small>
        <br />
        Caching text embeddings will process and cache all the text embeddings from the text encoder to the disk. The
        text encoder will be unloaded from the GPU. This does not work with things that dynamically change the prompt
        such as trigger words, caption dropout, etc.
      </>
    ),
  },
  'model.multistage': {
    title: 'Stages to Train',
    description: (
      <>
        Some models have multi stage networks that are trained and used separately in the denoising process. Most
        common, is to have 2 stages. One for high noise and one for low noise. You can choose to train both stages at
        once or train them separately. If trained at the same time, The trainer will alternate between training each
        model every so many steps and will output 2 different LoRAs. If you choose to train only one stage, the trainer
        will only train that stage and output a single LoRA.
      </>
    ),
  },
  'train.switch_boundary_every': {
    title: 'Switch Boundary Every',
    description: (
      <>
        When training a model with multiple stages, this setting controls how often the trainer will switch between
        training each stage.
        <br />
        <br />
        For low vram settings, the model not being trained will be unloaded from the gpu to save memory. This takes some
        time to do, so it is recommended to alternate less often when using low vram. A setting like 10 or 20 is
        recommended for low vram settings.
        <br />
        <br />
        The swap happens at the batch level, meaning it will swap between a gradient accumulation steps. To train both
        stages in a single step, set them to switch every 1 step and set gradient accumulation to 2.
      </>
    ),
  },
  'train.force_first_sample': {
    title: 'Force First Sample',
    description: (
      <>
        This option will force the trainer to generate samples when it starts. The trainer will normally only generate a
        first sample when nothing has been trained yet, but will not do a first sample when resuming from an existing
        checkpoint. This option forces a first sample every time the trainer is started. This can be useful if you have
        changed sample prompts and want to see the new prompts right away.
      </>
    ),
  },
  'model.layer_offloading': {
    title: (
      <>
        Layer Offloading{' '}
        <span className="text-yellow-500">
          ( <IoFlaskSharp className="inline text-yellow-500" name="Experimental" /> Experimental)
        </span>
      </>
    ),
    description: (
      <>
        This is an experimental feature based on{' '}
        <a className="text-blue-500" href="https://github.com/lodestone-rock/RamTorch" target="_blank">
          RamTorch
        </a>
        . This feature is early and will have many updates and changes, so be aware it may not work consistently from
        one update to the next. It will also only work with certain models.
        <br />
        <br />
        Layer Offloading uses the CPU RAM instead of the GPU ram to hold most of the model weights. This allows training
        a much larger model on a smaller GPU, assuming you have enough CPU RAM. This is slower than training on pure GPU
        RAM, but CPU RAM is cheaper and upgradeable. You will still need GPU RAM to hold the optimizer states and LoRA
        weights, so a larger card is usually still needed.
        <br />
        <br />
        You can also select the percentage of the layers to offload. It is generally best to offload as few as possible
        (close to 0%) for best performance, but you can offload more if you need the memory.
      </>
    ),
  },
  'model.qie.match_target_res': {
    title: 'Match Target Res',
    description: (
      <>
        This setting will make the control images match the resolution of the target image. The official inference
        example for Qwen-Image-Edit-2509 feeds the control image is at 1MP resolution, no matter what size you are
        generating. Doing this makes training at lower res difficult because 1MP control images are fed in despite how
        large your target image is. Match Target Res will match the resolution of your target to feed in the control
        images allowing you to use less VRAM when training with smaller resolutions. You can still use different aspect
        ratios, the image will just be resizes to match the amount of pixels in the target image.
      </>
    ),
  },
  'train.diff_output_preservation': {
    title: 'Differential Output Preservation',
    description: (
      <>
        Differential Output Preservation (DOP) is a technique to help preserve class of the trained concept during
        training. For this, you must have a trigger word set to differentiate your concept from its class. For instance,
        You may be training a woman named Alice. Your trigger word may be "Alice". The class is "woman", since Alice is
        a woman. We want to teach the model to remember what it knows about the class "woman" while teaching it what is
        different about Alice. During training, the trainer will make a prediction with your LoRA bypassed and your
        trigger word in the prompt replaced with the class word. Making "photo of Alice" become "photo of woman". This
        prediction is called the prior prediction. Each step, we will do the normal training step, but also do another
        step with this prior prediction and the class prompt in order to teach our LoRA to preserve the knowledge of the
        class. This should not only improve the performance of your trained concept, but also allow you to do things
        like "Alice standing next to a woman" and not make both of the people look like Alice.
      </>
    ),
  },
  'train.blank_prompt_preservation': {
    title: 'Blank Prompt Preservation',
    description: (
      <>
        Blank Prompt Preservation (BPP) is a technique to help preserve the current models knowledge when unprompted.
        This will not only help the model become more flexible, but will also help the quality of your concept during
        inference, especially when a model uses CFG (Classifier Free Guidance) on inference. At each step during
        training, a prior prediction is made with a blank prompt and with the LoRA disabled. This prediction is then
        used as a target on an additional training step with a blank prompt, to preserve the model's knowledge when no
        prompt is given. This helps the model to not overfit to the prompt and retain its generalization capabilities.
      </>
    ),
  },
  'train.do_differential_guidance': {
    title: 'Differential Guidance',
    description: (
      <>
        Differential Guidance will amplify the difference of the model prediction and the target during training to make
        a new target. Differential Guidance Scale will be the multiplier for the difference. This is still experimental,
        but in my tests, it makes the model train faster, and learns details better in every scenario I have tried with
        it.
        <br />
        <br />
        The idea is that normal training inches closer to the target but never actually gets there, because it is
        limited by the learning rate. With differential guidance, we amplify the difference for a new target beyond the
        actual target, this would make the model learn to hit or overshoot the target instead of falling short.
        <br />
        <br />
        <img src="/imgs/diff_guidance.png" alt="Differential Guidance Diagram" className="max-w-full mx-auto" />
      </>
    ),
  },
  'dataset.num_repeats': {
    title: 'Num Repeats',
    description: (
      <>
        Number of Repeats will allow you to repeate the items in a dataset multiple times. This is useful when you are
        using multiple datasets and want to balance the number of samples from each dataset. For instance, if you have a
        small dataset of 10 images and a large dataset of 100 images, you can set the small dataset to have 10 repeats
        to effectively make it 100 images, making the two datasets occour equally during training.
      </>
    ),
  },
  'train.audio_loss_multiplier': {
    title: 'Audio Loss Multiplier',
    description: (
      <>
        When training audio and video, sometimes the video loss is so great that it outweights the audio loss, causing
        the audio to become distorted. If you are noticing this happen, you can increase the audio loss multiplier to
        give more weight to the audio loss. You could try something like 2.0, 10.0 etc. Warning, setting this too high
        could overfit and damage the model.
      </>
    ),
  },
  'datasets.auto_frame_count': {
    title: 'Auto Frame Count',
    description: (
      <>
        This will automatically determine the number of frames to use for each video in your dataset instead of relying
        on a fixed num_frames. This allows you to include videos of different lengths in the dataset, and each video
        will be processed without speeding up or slowing down. Be careful about adding long videos into your dataset, as
        they use up more VRAM. This currently will not work with a batch size greater than 1.
      </>
    ),
  },
  'model.model_kwargs.kv_cache': {
    title: 'KV Cache',
    description: (
      <>
        This will enable KV Cache for control images in a model that supports it. LoRAs trained with this on
        need to also be inferenced with it, and vice versa. This does not speed up or slow down training, but on inference,
        the control images only need to be processed once for the entire generation, vs being processed for every step.
        Which leads to a significant speedup on inference.
      </>
    ),
  },

  'train.weight_noise.enabled': {
    title: 'Weight Noising',
    description: (
      <>
        Adds Gaussian perturbations to LoRA parameter values after each optimizer step. This is disabled by default and
        affects LoRA adapter parameters only.
      </>
    ),
  },
  'train.weight_noise.mode': {
    title: 'Weight Noise Mode',
    description: (
      <>
        Relative mode scales sigma by each parameter's RMS, so zero-initialized LoRA-up weights receive effectively no
        noise until they learn something. Absolute mode applies the same sigma to every parameter.
      </>
    ),
  },
  'train.weight_noise.sigma': {
    title: 'Weight Noise Sigma',
    description: (
      <>
        The noise scale. The conservative starting value is 0.00125; tune it only after comparing the logged noise norm
        with the clipped gradient norm on your model.
      </>
    ),
  },
  'train.weight_noise.bound_norm': {
    title: 'Bound Weight Norm',
    description: (
      <>
        Rescales each tensor back to its pre-noise norm after perturbation. This removes the radial norm growth from a
        long random walk while preserving the tangential change.
      </>
    ),
  },
  'train.weight_noise.log_every': {
    title: 'Weight Noise Metric Cadence',
    description: <>Set to 0 to disable weight-noise metrics, or log them every N optimizer steps.</>,
  },
  'train.gradient_noise.enabled': {
    title: 'Gradient Noising',
    description: (
      <>
        Adds Gaussian perturbations to clipped LoRA gradients immediately before the optimizer step. This is disabled
        by default and leaves untagged parameters unchanged.
      </>
    ),
  },
  'train.gradient_noise.mode': {
    title: 'Gradient Noise Mode',
    description: (
      <>
        Absolute mode uses a fixed sigma. Relative mode scales sigma by each gradient's RMS. Neelakantan mode starts at
        eta and anneals it with gamma as training progresses.
      </>
    ),
  },
  'train.gradient_noise.sigma': {
    title: 'Gradient Noise Sigma',
    description: <>The fixed or relative gradient-noise scale used by the selected mode.</>,
  },
  'train.gradient_noise.eta': {
    title: 'Gradient Noise Eta',
    description: <>The initial noise scale for Neelakantan annealing.</>,
  },
  'train.gradient_noise.gamma': {
    title: 'Gradient Noise Gamma',
    description: <>The decay exponent for Neelakantan gradient noise; larger values anneal more quickly.</>,
  },
  'train.gradient_noise.log_every': {
    title: 'Gradient Noise Metric Cadence',
    description: <>Set to 0 to disable gradient-noise metrics, or log them every N optimizer steps.</>,
  },
  'depth_consistency.loss_weight': {
    title: 'Depth Consistency',
    description: (
      <>
        Enables a Depth Anything V2 perceptor as a perceptual anchor. The anchor decodes the predicted clean latent
        through the VAE, runs DA2 to produce a depth map, and adds an SSI + multi-scale gradient loss against the cached
        ground-truth depth of the training image. Enable sets this weight to 0.001 (a starting value, not a performance
        claim); disable sets it to 0. There is no separate on/off flag. Depth keeps or moves the VAE back onto the GPU,
        so it removes the VAE-offload benefit of latent caching (caching still avoids repeated encodes).
      </>
    ),
  },
  'depth_consistency.model_id': {
    title: 'Depth Model',
    description: (
      <>
        The Depth Anything V2 perceptor used for the anchor. Loading is lazy: selecting the model here does not download
        it; enabling depth does. The Depth Anything V2 Hugging Face weights are published under
        CC-BY-NC-4.0 (see the{' '}
        <a
          className="text-blue-500"
          href="https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf"
          target="_blank"
        >
          DA2-Small model card
        </a>
        ). This is disclosure, not a substitute for the project license review.
      </>
    ),
  },
  'depth_consistency.input_size': {
    title: 'Input Size',
    description: (
      <>
        The square resolution DA2 receives. 518 is the shipped baseline. 1024 is an experimental preset for large-VRAM
        setups and is only promoted to a default with measured memory, throughput, and quality evidence.
      </>
    ),
  },
  'depth_consistency.loss_min_t': {
    title: 'Minimum Timestep',
    description: (
      <>
        The anchor loss only applies to training steps whose flow-matching timestep ratio is at least this value (0 to
        1). Raise it to skip the anchor at low-noise steps.
      </>
    ),
  },
  'depth_consistency.loss_max_t': {
    title: 'Maximum Timestep',
    description: (
      <>
        The anchor loss only applies to training steps whose flow-matching timestep ratio is at most this value (0 to
        1). Lower it to skip the anchor at high-noise steps. Must be greater than or equal to the minimum.
      </>
    ),
  },
  'depth_consistency.preview_every': {
    title: 'Preview Every',
    description: <>Write four-panel Krea depth preview tiles every N optimizer steps. Set to 0 to disable previews.</>,
  },
  'depth_consistency.ssi_weight': {
    title: 'SSI Weight',
    description: <>Weight of the scale-and-shift-invariant depth term in the anchor loss.</>,
  },
  'depth_consistency.grad_weight': {
    title: 'Gradient Weight',
    description: <>Weight of the multi-scale image-gradient depth term in the anchor loss.</>,
  },
  'depth_consistency.grad_scales': {
    title: 'Gradient Scales',
    description: <>Number of dyadic scales used by the multi-scale gradient depth term.</>,
  },
  'depth_consistency.pixel_blur_sigma': {
    title: 'Pixel Blur Sigma',
    description: (
      <>
        Optional Gaussian blur applied to the decoded pixels before DA2 runs. 0 disables it. Useful for reducing
        high-frequency decode artifacts before depth estimation.
      </>
    ),
  },
  'depth_consistency.grad_checkpoint': {
    title: 'Gradient Checkpointing',
    description: <>Runs DA2 with gradient checkpointing to lower activation memory at the cost of recomputation.</>,
  },
  'depth_consistency.preview_only': {
    title: 'Preview Only',
    description: (
      <>
        Loads the perceptor and writes previews, but never adds the anchor loss or suppresses diffusion. Useful for
        inspection. Preview evaluation runs without autograd and only at the configured Preview Every cadence within
        the timestep window.
      </>
    ),
  },
  'depth_consistency.preview_max_keep': {
    title: 'Preview Max Keep',
    description: <>Maximum number of preview tile sets to retain on disk; older sets are pruned.</>,
  },
  'train.loss_split': {
    title: 'Loss Split',
    description: (
      <>
        Controls whether the diffusion loss and the depth anchor loss are summed every step or applied on alternating
        steps. The three serialized states are distinct: Auto leaves the key absent; Sum (off) stores an explicit null;
        Alternate stores 'diffusion_depth'. The resolver precedence (sec. 4.1 Step 2) is, in order:
        <br />
        <br />
        1a. dataset value == 'sum' --&gt; None (per-dataset force off)
        <br />
        1b. dataset value set (not 'sum') --&gt; that value
        <br />
        2. global explicitly set --&gt; global value
        <br />
        3. autodetect (nothing set) --&gt; 'diffusion_depth' if depth weight &gt; 0 else None
        <br />
        <br />
        A per-dataset override always wins over the global setting. With Auto selected and depth disabled, the anchor
        contributes nothing. Batch size alone never changes which step parity is selected.
      </>
    ),
  },
  'datasets.depth_loss_weight': {
    title: 'Per-Dataset Depth Loss Weight',
    description: (
      <>
        Overrides the global depth loss weight for this dataset only. Leave empty to inherit the global value. A value
        greater than 0 activates the depth anchor for this dataset even when the global object is otherwise disabled.
      </>
    ),
  },
  'datasets.depth_loss_min_t': {
    title: 'Per-Dataset Min Timestep',
    description: <>Overrides the global minimum anchor timestep for this dataset only. Leave empty to inherit.</>,
  },
  'datasets.depth_loss_max_t': {
    title: 'Per-Dataset Max Timestep',
    description: <>Overrides the global maximum anchor timestep for this dataset only. Leave empty to inherit.</>,
  },
  'datasets.loss_split': {
    title: 'Per-Dataset Loss Split',
    description: (
      <>
        Overrides the global loss split for this dataset only. Auto inherits the global / autodetect result; Sum forces
        the anchor off for this dataset (the resolver maps 'sum' to None); Alternate forces diffusion/depth alternation.
      </>
    ),
  },
  'normal_id.loss_weight': {
    title: 'Surface-Normal Anchor',
    description: (
      <>
        Enables a frozen Sapiens 0.3B surface-normal perceptor as a perceptual anchor. The anchor decodes the predicted
        clean latent through the VAE, runs Sapiens to produce a per-pixel unit normal map, and adds a cosine + L1 loss
        against the cached ground-truth normals of the training image. Enable sets this weight to 0.01 (a starting value,
        not a performance claim); disable sets it to 0. There is no separate on/off flag. The Sapiens weights download
        lazily the first time normal loss is enabled. Unlike depth, the normal anchor does not participate in the
        diffusion/depth loss-split alternation -- it fires every step within its timestep window. Like depth it decodes
        x0 under gradient, so Low VRAM is disabled while normal is active.
      </>
    ),
  },
  'normal_id.loss_min_t': {
    title: 'Minimum Timestep',
    description: (
      <>
        The normal anchor only applies to training steps whose flow-matching timestep ratio is at least this value
        (0 to 1). Defaults to 0.4 so the anchor skips the low-noise steps where surface detail is already determined.
      </>
    ),
  },
  'normal_id.loss_max_t': {
    title: 'Maximum Timestep',
    description: (
      <>
        The normal anchor only applies to training steps whose flow-matching timestep ratio is at most this value
        (0 to 1). Defaults to 0.8. Must be greater than or equal to the minimum.
      </>
    ),
  },
  'normal_id.preview_every': {
    title: 'Preview Every',
    description: (
      <>
        Every N normal-active steps, write a four-panel preview tile (GT RGB | GT normal | predicted RGB | predicted
        normal) to the normal_previews folder. Set to 0 to disable. Previews are taken from the training-merged base
        (Krea 2 Turbo merges its training adapter at +1.0 for training), so they may differ from final samples.
      </>
    ),
  },
  'datasets.normal_loss_weight': {
    title: 'Per-Dataset Normal Loss Weight',
    description: (
      <>
        Overrides the global normal-anchor loss weight for this dataset only. Leave empty to inherit the global setting;
        set to 0 to disable the normal anchor for this dataset.
      </>
    ),
  },
  'datasets.normal_loss_min_t': {
    title: 'Per-Dataset Minimum Timestep',
    description: <>Overrides the global normal-anchor minimum timestep for this dataset only. Leave empty to inherit.</>,
  },
  'datasets.normal_loss_max_t': {
    title: 'Per-Dataset Maximum Timestep',
    description: <>Overrides the global normal-anchor maximum timestep for this dataset only. Leave empty to inherit.</>,
  },
  'body_proportion.loss_weight': {
    title: 'Body-Proportion Anchor',
    description: (
      <>
        Enables a frozen ViTPose-Plus-Base pose estimator as a perceptual anchor. The anchor decodes the
        predicted clean latent through the VAE, runs ViTPose to get 17 COCO keypoints, derives 8
        pose-invariant bone-length ratios (10 with head), and adds a visibility-weighted L1 loss plus a
        missing-keypoint penalty against cached GT ratios. Enable sets this weight to 0.01 (a starting
        value, not a performance claim); disable sets it to 0. ViTPose weights download lazily on first
        enable. Body-proportion does not participate in the diffusion/depth loss-split alternation. Like
        the other anchors it decodes x0 under gradient, so Low VRAM is disabled while it is active.
      </>
    ),
  },
  'body_proportion.loss_min_t': {
    title: 'Minimum Timestep',
    description: <>The body-proportion anchor only applies on steps whose timestep ratio is at least this (0 to 1).</>,
  },
  'body_proportion.loss_max_t': {
    title: 'Maximum Timestep',
    description: <>The body-proportion anchor only applies on steps whose timestep ratio is at most this (0 to 1).</>,
  },
  'body_proportion.include_head': {
    title: 'Include Head Ratios',
    description: (
      <>
        Add two head ratios (nose-to-shoulder height and ear-to-ear width) to the 8 body ratios. Useful for
        character likenesses where head proportions matter. Changing this invalidates the GT cache.
      </>
    ),
  },
  'datasets.body_proportion_loss_weight': {
    title: 'Per-Dataset Body-Proportion Weight',
    description: <>Overrides the global body-proportion loss weight for this dataset only. Leave empty to inherit; 0 disables.</>,
  },
  'datasets.body_proportion_loss_min_t': {
    title: 'Per-Dataset Minimum Timestep',
    description: <>Overrides the global body-proportion minimum timestep for this dataset only. Leave empty to inherit.</>,
  },
  'datasets.body_proportion_loss_max_t': {
    title: 'Per-Dataset Maximum Timestep',
    description: <>Overrides the global body-proportion maximum timestep for this dataset only. Leave empty to inherit.</>,
  },
  'face_id.identity_loss_weight': {
    title: 'Face-Identity Anchor',
    description: (
      <>
        Enables a frozen ArcFace (w600k_r50, InsightFace buffalo_l) as a face-identity anchor. The anchor
        decodes the predicted clean latent through the VAE, crops the face (using the cached face bbox),
        runs ArcFace, and adds a bias-corrected cosine-similarity loss against the cached GT identity
        embedding. A SCRFD quality gate skips generated regions where no face is detected. Enable sets this
        weight to 0.05 (a starting value, not a performance claim); disable sets it to 0. Requires the
        manual dep install (insightface + onnx2torch + onnxruntime-gpu with the CPU-shadowing fix). The
        ArcFace model downloads lazily on first enable. Does not participate in diffusion/depth loss-split.
        Like the other anchors it decodes x0 under gradient, so Low VRAM is disabled while it is active.
      </>
    ),
  },
  'face_id.identity_loss_min_t': {
    title: 'Minimum Timestep',
    description: <>The identity anchor only applies on steps whose timestep ratio is at least this (0 to 1).</>,
  },
  'face_id.identity_loss_max_t': {
    title: 'Maximum Timestep',
    description: <>The identity anchor only applies on steps whose timestep ratio is at most this (0 to 1).</>,
  },
  'face_id.identity_loss_min_cos': {
    title: 'Min Cosine Similarity',
    description: (
      <>
        Floor (bias-corrected cosine) below which the loss does not push. Prevents pushing on generated
        regions that score weakly against the reference identity (hallucinated or low-quality faces).
      </>
    ),
  },
  'datasets.identity_loss_weight': {
    title: 'Per-Dataset Identity Loss Weight',
    description: <>Overrides the global face-identity loss weight for this dataset only. Leave empty to inherit; 0 disables.</>,
  },
  'datasets.identity_loss_min_t': {
    title: 'Per-Dataset Minimum Timestep',
    description: <>Overrides the global identity-anchor minimum timestep for this dataset only. Leave empty to inherit.</>,
  },
  'datasets.identity_loss_max_t': {
    title: 'Per-Dataset Maximum Timestep',
    description: <>Overrides the global identity-anchor maximum timestep for this dataset only. Leave empty to inherit.</>,
  },
  'depth_consistency.mask_source': {
    title: 'Depth Mask Source',
    description: (
      <>
        Restricts the depth-anchor loss to a cached region. None applies the full-image loss; Subject uses
        the person mask (body + clothing); Body uses only skin/hair/limbs. Subject/Body require
        Auto-Masking to be enabled, or the trainer will refuse to start. The depth loss gracefully
        degrades to full-image for any sample whose mask is missing.
      </>
    ),
  },
  'subject_mask.enabled': {
    title: 'Auto-Masking',
    description: (
      <>
        Extracts per-image person/body/clothing masks via YOLO (person detection) + SAM 2 (silhouette) +
        SegFormer-clothes (semantic source of truth). Masks are cached at preflight time (non-
        differentiable) and used to region-weight the diffusion loss and restrict the perceptual anchors.
        SegFormer is the semantic source of truth; YOLO detects people; SAM 2 provides a reference
        silhouette. Models download lazily on first enable. Requires ultralytics + transformers + opencv.
      </>
    ),
  },
  'subject_mask.background_loss_weight': {
    title: 'Background Loss Weight',
    description: <>Multiplies the diffusion loss OUTSIDE the person region by this factor. Lower (eg. 0) de-emphasizes the background; null/1.0 inherits (no change).</>,
  },
  'subject_mask.body_loss_weight': {
    title: 'Body Loss Weight',
    description: <>Multiplies the diffusion loss INSIDE the body region (skin/hair/limbs) by this factor. Higher (eg. 2) emphasizes identity-relevant regions.</>,
  },
  'subject_mask.clothing_loss_weight': {
    title: 'Clothing Loss Weight',
    description: <>Multiplies the diffusion loss INSIDE the clothing region by this factor.</>,
  },
  'subject_mask.perceptual_restrict_to_body': {
    title: 'Restrict Perceptual Anchors to Body',
    description: <>When on, the surface-normal anchor loss is computed only inside the body region (per-sample normalized so unrestricted items keep their magnitude).</>,
  },
  'subject_mask.sam_size': {
    title: 'SAM 2 Size',
    description: <>The SAM 2 checkpoint size used for the reference silhouette. Small is the default; larger is slower but sharper.</>,
  },
  'body_shape.loss_weight': {
    title: 'Body-Shape Anchor',
    description: (
      <>
        Enables a frozen HybrIK ResNet-34 to regress 10-dim SMPL body-shape betas from the decoded x0 and
        match them against cached GT via L1 (cosine-gated). Requires the HybrIK checkpoint
        (Google-Drive-only; install gdown for auto-download, or place hybrik_resnet34.pth in
        ~/.cache/hybrik/). Distinct from body-proportion (ViTPose ratios). Enable sets this weight to 0.05.
        Like the other anchors it decodes x0 under gradient, so Low VRAM is disabled while it is active.
      </>
    ),
  },
  'body_shape.loss_min_t': {
    title: 'Minimum Timestep',
    description: <>The body-shape anchor only applies on steps whose timestep ratio is at least this (0 to 1). Defaults to 0.4.</>,
  },
  'body_shape.loss_max_t': {
    title: 'Maximum Timestep',
    description: <>The body-shape anchor only applies on steps whose timestep ratio is at most this (0 to 1). Defaults to 0.8.</>,
  },
  'body_shape.loss_min_cos': {
    title: 'Min Cosine Similarity',
    description: <>Floor (beta cosine) below which the loss does not push, preventing pushing on poorly-matched body shapes.</>,
  },
  'datasets.body_shape_loss_weight': {
    title: 'Per-Dataset Body-Shape Weight',
    description: <>Overrides the global body-shape loss weight for this dataset only. Leave empty to inherit; 0 disables.</>,
  },
  'vae_anchor.loss_weight': {
    title: 'VAE Anchor',
    description: (
      <>
        Cross-VAE perceptual anchor: decodes the predicted x0 through the training model's VAE, encodes
        those pixels with a SEPARATE frozen Flux 2 VAE encoder, and matches the multi-scale features
        against cached GT via cosine similarity (5 levels, higher-resolution weighted more). The Flux 2
        VAE downloads from Hugging Face on first enable. Anchors to the Flux 2 VAE feature space
        (independent of Krea 2's VAE). Like the other anchors it decodes x0 under gradient, so Low VRAM
        is disabled while it is active.
      </>
    ),
  },
  'vae_anchor.loss_min_t': {
    title: 'Minimum Timestep',
    description: <>The VAE anchor only applies on steps whose timestep ratio is at least this (0 to 1).</>,
  },
  'vae_anchor.loss_max_t': {
    title: 'Maximum Timestep',
    description: <>The VAE anchor only applies on steps whose timestep ratio is at most this (0 to 1). Defaults to 0.5 (low-noise region where structure is set).</>,
  },
  'train.optimizer.rose': {
    title: 'Rose Optimizer',
    description: (
      <>
        Rose (Range-Of-Slice Equilibration) is a <strong>stateless</strong> optimizer: it keeps no
        per-parameter momentum or variance buffers, so optimizer-state memory is zero. It rescales
        each gradient slice by its <code>|max| - min</code> range.
        <br />
        <br />
        Its learning rate is <strong>not comparable to Adam's</strong> — range-based normalization
        produces very different effective step sizes, so tune <code>lr</code> independently rather
        than reusing Adam defaults.
        <br />
        <br />
        The default <code>compute_dtype: fp64</code> promotes parameters and gradients to FP64 for
        the update. This trades speed and working memory for update precision and is recommended
        because the intermediate range/division arithmetic benefits from it.
        <br />
        <br />
        In this release Rose is selectable here with documented backend parameters. Advanced
        parameters (<code>weight_decay</code>, <code>wd_schedule</code>, <code>centralize</code>,{' '}
        <code>stabilize</code>, <code>bf16_sr</code>, <code>compute_dtype</code>) are YAML/config
        only and are not exposed as GUI controls.
      </>
    ),
  },

  'train.guidance_loss_target': {
    title: 'Guidance Loss Target',
    description: (
      <>
        For contrastive guidance loss, this is the target CGF to amplify predictions to. 

      </>
    ),
  },
  'train.guidance_loss_sigma_min': {
    title: 'Guidance Loss Sigma Min',
    description: (
      <>
        Skip the contrastive guidance-loss correction (and its extra unconditional forward) on
        steps whose pre-shift base sigma (<code>timesteps / 1000</code>, where 1 is pure noise) is
        below this threshold. Default 0 always applies. For MiniMax-H3, 0.15 skips the clean end of
        the draw where per-sample label noise dominates the guidance gap — cheaper steps, and the
        audio stream benefits most. This is <em>not</em> a high-noise skip.
      </>
    ),
  },

  'datasets.caption_dropout_rate': {
    title: 'Caption Dropout Rate',
    description: (
      <>
        Caption dropout rate is the probability that the caption for an image will be dropped (replaced with a blank
        caption) for any given training step. For example, a value of 0.05 will drop the caption around 5% of the time.
        Dropping captions helps the model learn the concept being trained without relying entirely on the caption,
        and helps preserve the model&apos;s ability to generate without a prompt. If a trigger word is set, the trigger
        word is still used when the caption is dropped, so the model still associates the dropped samples with your
        trigger word. Regularization images, or images without a trigger word, drop to a fully blank caption.
        <br />
        <br />
        Caption dropout also works when caching text embeddings. An additional embedding for the dropout caption
        (blank, or the trigger word alone) is cached to disk alongside the normal one, and it is randomly swapped in
        at train time at this rate.
      </>
    ),
  },
  'network.pretrained_lora_path': {
    title: 'Resume from LoRA',
    summary: 'Start from an existing LoRA instead of training from scratch.',
    description: 'Path to a .safetensors LoRA. Weights load as the starting point for this run.',
  },
  'network.dropout': {
    title: 'Network Dropout',
    summary: 'Randomly drop LoRA weights during training to reduce overfitting.',
    description: 'Leave empty for no dropout. Typical values are 0.05–0.1.',
  },
  'network.transformer_only': {
    title: 'Transformer Only',
    summary: 'Train LoRA only on transformer blocks, not every linear layer.',
    description: 'On by default for diffusion transformers. Turn off only if you know you need extra modules.',
  },
  'network.only_if_contains': {
    title: 'Train Only Layers',
    summary: 'Only create LoRA on layers whose names contain these substrings.',
    description: (
      <>
        Comma-separated name fragments. Example: <code>single_transformer_blocks.7.proj_out</code>. Empty means all
        layers (minus Ignore). Ignore wins when both match.
      </>
    ),
  },
  'network.ignore_if_contains': {
    title: 'Ignore Layers',
    summary: 'Skip LoRA on layers whose names contain these substrings.',
    description: (
      <>
        Comma-separated name fragments. Some architectures pre-fill this (experts, gates). Ignore takes priority over
        Train Only.
      </>
    ),
  },
  'datasets.mask_path': {
    title: 'Mask Dataset',
    summary: 'Folder of masks matching your images so loss focuses on the subject.',
    description: 'Same filenames as the target dataset. White/high values are trained more; black is down-weighted.',
  },
  'datasets.mask_min_value': {
    title: 'Mask Min Value',
    summary: 'Floor for masked-out pixels so they still get a little loss.',
    description: '0 ignores the background completely. 0.1 is a safe default so the rest of the image does not collapse.',
  },
  'datasets.invert_mask': {
    title: 'Invert Mask',
    summary: 'Swap subject and background if your masks are the wrong way around.',
    description: 'Use when white is background instead of the subject.',
  },
  'datasets.shuffle_tokens': {
    title: 'Shuffle Tokens',
    summary: 'Shuffle caption tokens each step so the model cannot memorize order.',
    description: 'Keep Tokens (below) stays at the front when shuffling.',
  },
  'datasets.keep_tokens': {
    title: 'Keep Tokens',
    summary: 'How many leading caption tokens stay fixed when shuffling.',
    description: 'Use 1 to keep a trigger word first. 0 shuffles everything.',
  },
  'datasets.token_dropout_rate': {
    title: 'Token Dropout',
    summary: 'Drop individual caption tokens at random.',
    description: 'Unlike caption dropout (whole caption), this removes pieces of the prompt.',
  },
  'datasets.replacements': {
    title: 'Caption Replacements',
    summary: 'Rewrite caption text before training. One rule per line: old|new.',
    description: 'Example: person|woman. Applied to every caption in this dataset.',
  },
  'datasets.cache_latents': {
    title: 'Cache Latents in RAM',
    summary: 'Keep VAE latents in memory instead of (or as well as) disk.',
    description: 'Faster than disk cache, uses a lot of RAM. Disk cache is the checkbox above.',
  },
  'datasets.num_workers': {
    title: 'Data Workers',
    summary: 'Background processes that load images.',
    description: 'Default is 2. Raise on fast disks if the GPU waits on data.',
  },
  'datasets.pin_memory': {
    title: 'Pin Memory',
    summary: 'Faster CPU→GPU copies. Uses extra host RAM.',
    description: 'Usually leave off unless you know the dataloader is the bottleneck.',
  },
  'datasets.buckets': {
    title: 'Buckets',
    summary: 'Group images by aspect ratio instead of forcing one size.',
    description: 'On by default. Turn off only for debugging.',
  },
  'datasets.bucket_tolerance': {
    title: 'Bucket Tolerance',
    summary: 'Pixel step used when grouping aspect ratios.',
    description: '64 is the usual value. Smaller means more buckets and more VRAM fragmentation.',
  },
  'datasets.random_crop': {
    title: 'Random Crop',
    summary: 'Crop a random window instead of a center crop.',
    description: 'Helps when images are larger than the training resolution.',
  },
  'datasets.square_crop': {
    title: 'Square Crop',
    summary: 'Force a square crop before bucketing.',
    description: 'Use for models that dislike wide/tall buckets.',
  },
  'datasets.random_scale': {
    title: 'Random Scale',
    summary: 'Jitter scale slightly before cropping.',
    description: 'Mild augmentation. Off by default.',
  },
  'save.save_format': {
    title: 'Save Format',
    summary: 'How checkpoints are written: Diffusers folder or a single safetensors file.',
    description: 'Diffusers is required for some full models. LoRAs usually use safetensors.',
  },
  'save.push_to_hub': {
    title: 'Push to Hub',
    summary: 'Upload each save to Hugging Face.',
    description: 'Needs a token on the Settings page. Set a repo id below.',
  },
  'save.hf_repo_id': {
    title: 'Hub Repo ID',
    summary: 'username/model-name on Hugging Face.',
    description: 'Created if it does not exist. Private repos need a write token with repo access.',
  },
  'save.hf_private': {
    title: 'Private Repo',
    summary: 'Make the Hugging Face repo private.',
    description: 'Only applies when Push to Hub is on.',
  },
  'train.lr_scheduler': {
    title: 'LR Scheduler',
    summary: 'How learning rate changes over the run. Constant is the safe default.',
    description: 'Cosine/linear decay can help long runs. Constant is what most LoRAs should use.',
  },
  'train.unet_lr': {
    title: 'Unet / Transformer LR',
    summary: 'Override LR for the transformer. Empty uses the main learning rate.',
    description: 'Leave empty unless you are training multiple parts at different rates.',
  },
  'train.text_encoder_lr': {
    title: 'Text Encoder LR',
    summary: 'Override LR for the text encoder. Empty uses the main learning rate.',
    description: 'Only matters if Train Text Encoder is on.',
  },
  'train.train_text_encoder': {
    title: 'Train Text Encoder',
    summary: 'Also train the text encoder, not just the LoRA on the transformer.',
    description: 'Usually off for Flux/Qwen/HiDream. Can overfit captions quickly.',
  },
  'train.min_snr_gamma': {
    title: 'Min-SNR Gamma',
    summary: 'Down-weight very noisy timesteps so they do not dominate loss.',
    description: '5 is a common value. Empty disables it. Do not combine with SNR Gamma.',
  },
  'train.snr_gamma': {
    title: 'SNR Gamma',
    summary: 'Fixed SNR weighting. Empty disables it.',
    description: 'Prefer Min-SNR Gamma unless you have a reason. Do not set both.',
  },
  'train.noise_offset': {
    title: 'Noise Offset',
    summary: 'Adds a little extra noise so dark/bright scenes train better.',
    description: '0.05–0.1 is the usual range. 0 is off.',
  },
  'train.min_denoising_steps': {
    title: 'Min Denoising Step',
    summary: 'Do not train on timesteps below this.',
    description: 'Leave at 0 unless you are excluding near-clean latents.',
  },
  'train.max_denoising_steps': {
    title: 'Max Denoising Step',
    summary: 'Do not train on timesteps above this.',
    description: 'Leave at 999 unless you are excluding near-pure-noise latents.',
  },
  'model.compile_mode': {
    title: 'Compile Mode',
    summary: 'torch.compile strategy. Default is safest.',
    description: 'max-autotune is slower to start. reduce-overhead helps small batches.',
  },
  'model.compile_fullgraph': {
    title: 'Compile Fullgraph',
    summary: 'Require a single compiled graph. Fails if the model breaks the graph.',
    description: 'Leave off unless you are chasing extra speed and know it compiles cleanly.',
  },
  'model.compile_dynamic': {
    title: 'Compile Dynamic',
    summary: 'Allow varying shapes (resolutions, batch) in the compiled model.',
    description: 'On by default so multi-resolution training does not recompile every step.',
  },
  'model.cache_size_limit': {
    title: 'Compile Cache Limit',
    summary: 'How many compiled graphs to keep. Empty uses PyTorch’s default.',
    description: 'Raise if you train many resolutions and see constant recompiles.',
  },
  'model.quantize_exclude': {
    title: 'Quantize Exclude',
    summary: 'Layer-name substrings to leave in full precision.',
    description: 'Comma-separated. Use if quantizing a layer makes training NaN.',
  },
  'model.vae_path': {
    title: 'VAE Path',
    summary: 'Optional replacement VAE. Empty uses the model’s default.',
    description: 'Huggingface id or local path.',
  },
  'model.te_name_or_path': {
    title: 'Text Encoder Path',
    summary: 'Optional replacement text encoder. Empty uses the model’s default.',
    description: 'Huggingface id or local path.',
  },
  'model.extras_name_or_path': {
    title: 'Extras Path',
    summary: 'Optional extra weights (tokenizer, scheduler, etc.). Empty uses the model path.',
    description: 'Only needed when those files live somewhere other than the base model.',
  },
  'sample.neg': {
    title: 'Negative Prompt',
    summary: 'Applied to every sample unless a prompt overrides it.',
    description: 'Often unused on flow-matching models with guidance 1.',
  },
  'sample.format': {
    title: 'Sample Format',
    summary: 'File format for preview images.',
    description: 'jpg is smaller. png is lossless. webp is a middle ground.',
  },
  'datasets.prefetch_factor': {
    title: 'Prefetch Factor',
    summary: 'Batches each worker prepares ahead of the GPU.',
    description: 'Default is 2. Only matters when Data Workers is above 0.',
  },
};

export const getDoc = (key: string | null | undefined): ConfigDoc | null => {
  if (key && key in docs) {
    return docs[key];
  }
  return null;
};

export default docs;
