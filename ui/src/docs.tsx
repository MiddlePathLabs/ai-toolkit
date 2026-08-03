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
  'train.guidance_loss_target': {
    title: 'Guidance Loss Target',
    description: (
      <>
        For contrastive guidance loss, this is the target CGF to amplify predictions to. 
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
};

export const getDoc = (key: string | null | undefined): ConfigDoc | null => {
  if (key && key in docs) {
    return docs[key];
  }
  return null;
};

export default docs;
