# Identity Isolation: reference audit and adaptations

## Revision and licensing

Inspected local Fizgig HEAD `d63ca56dd307ccda820a0304c54473c9ba7ce29b` (`v8.4.0-5-gd63ca56d-dirty`). Release tag `v8.4.0` resolves to `e2f7ccfe33b754910a11981978a6ee1658f18923`. Initial read-only `git diff v8.4.0 --stat` enumerated only `.github/workflows/sync-v8.yml`, README, Docker README, release-note additions, `src/fizgig/web/server.py`, and a removed output `.gitkeep`; isolation implementation files have no differences against the release tag. Local untracked documentation and removed output `.gitkeep` were left untouched. Fizgig is reference material, never a modification target.

Fizgig source is Apache-2.0, copyright 2026 Peter Neill. Adapted modules identify their origin and changes. Its NOTICE also credits AI-Toolkit (Ostris, LLC, MIT) for DOP. No unrelated GPL model/sampler implementation is incorporated.

Stranger assets have a **separate license**, CC BY 4.0: [pinned dataset card](https://huggingface.co/datasets/ShootTheSound/fizgig-multi-person-strangers/raw/8aeb77502234781cb5dca524994c80efcaa51024/README.md). Collection: ShootTheSound/Fizgig multi-person strangers; 32 synthetic Qwen Image 2.1 portraits, base model, 25 steps, CFG 3, seeds 1000–1031, 768×768. The card and metadata do not establish real-world identity separation. Asset preparation preserves attribution; composites are adaptations (crop, resize, pairing with user-provided photos). User-image rights remain the user's responsibility.

## Source-to-requirement map

Paths below are relative to the Fizgig root. Source code is authoritative; statements about empirical effectiveness in its release notes are not independently verified.

| Mechanism | Authoritative symbols | Actual behavior/defaults | Toolkit reuse and adaptation |
|---|---|---|---|
| Fused conditioning | `src/fizgig/families/multi_person.py`: `subject`, `subject_as_written`, `fuse`; `dataset/image_dataset.py` caption loading | Case-insensitive word-boundary sequential literal replacements: exact trigger/comma/class phrase, trigger/class phrase, then bare trigger. Concatenates trigger and final class noun; does not register a tokenizer token. Files untouched. | Existing DTO caption preprocessing and content-keyed text embeddings. Explicit classes and collision checks; apply every subject mapping across folders to meet multi-trigger requirement. |
| Stranger acquisition/cache | `multi_person.py`: `REPO`, `REVISION`, `_local_strangers`, `strangers`, `download_strangers` | Pin `8aeb77502234781cb5dca524994c80efcaa51024`; reference automatically downloads absent assets. Local marker + nonempty manifest + file existence, no hashes. Operational directory `user_data/multi_person_strangers`. | Explicit asset-preparation command only; existing local assets supported; pinned hash verification. No training/test network access. |
| Compositor | `multi_person.py`: `_square`, `build`; `scripts/multi_person.py`: `main` | 16 per subject; center square crop/LANCZOS to 512; 1024×512 sheet. `Random(seed*1000+subject_index)`, shuffled stranger order cycling, random source choice and side. Stranger described first regardless of side. Caption ends `two people side by side, two close-up portraits`. CLI seed42; launcher does not forward run seed. | Separate generated dataset blocks; explicit seed/count; content-hashed inputs and output completeness, rather than filename-only cache. |
| Multiple subjects | `launch.py`: `multi_person_subjects`, `multi_person_plan`, dataset TOML generation; form subject count | 1–4 form subjects, one folder/trigger each; no new joint-identity loss. Composites pair one trained subject with a stranger, never two trained subjects. Styles/things have special source modes, people get composites. | Person-identity port; explicit per-folder metadata and independent statistics. Broader style/animal training is not claimed. |
| DOP | `families/train.py`: `_dop_step`, `_dop_captions`, `_load_dop_cond`; `families/lora.py`: `set_trainable_multiplier` | Fresh noisy state independent of main objective. Reference with trainable adapter disabled/no_grad, student enabled; same DOP state and conditioning, float32 MSE, weight/accumulation. Other frozen adapters may remain. | Extend existing Toolkit teacher/student path (same ordinary noisy state), not replace it. Preserve auxiliary losses/masking. This noise-sampling difference is explicit. |
| Multi-trigger DOP | `train.py`: `_dop_pairs`, `_dop_caption`; `multi_person.py`: `dop_caption` | Trigger/class lists, singleton class broadcast, empty class removal; boundary substitutions; returns None when no trigger. Text caches keyed by full paths. | All configured raw/fused identities removed from preservation captions; captions without triggers remain usable for Toolkit full-frame preservation. Cached/live use same transform. |
| Strided DOP | `launch.py` DOP flags; `train.py` main loop `global_step` | Launcher interval2, weight2. Counts consumed loader batches including skipped items, first selected step0. Accumulation divides by configured factor, including partial end windows. | Successful optimizer-update-window clock; all contributing microbatches share decision. AMP/OOM skips do not advance phase. Not equivalent to reference cadence under accumulation/skips, nor mathematically identical to every-step DOP. |
| Composite LR | `train.py` `lr_boost`/`_update`; `launch.py` flags | Prefix substring match, multiplier2, temporary group-LR scaling/restoration. Accumulation averages multipliers. Automagic3 ignores multiplier; fused fine-tune bypasses it. | Explicit dataset category, homogeneous windows, actual optimizer LR scaling using existing runtime adapter. Unsupported combinations rejected; never replace with loss weighting or average categories. |
| Subject balancing | `launch.py`: `_media_count`, `_balance_repeats` | Optional approximate repeats `max(1,round(max_count/count))`; originals only, composites repeat1. Media counts may precede later eligibility filtering. | Existing repeats; report effective loader probabilities including bucket rounding and exposure estimates. Fixed configured step budget. |
| Preview transforms | `launch.py` sample/prompt preparation | Flatten all subject replacement rules for previews, even though source dataset transforms are per-folder. | Same shared caption policy before preview text encoding/caching. |

## Material source/document discrepancies

- `MULTI_PERSON.md` says `zwxem, a young blonde woman` becomes `zwxemwoman`. Source only infers the noun across adjectives; its actual transform is `zwxemwoman, a young blonde woman`. This port follows source rather than deleting descriptive words.
- Release-note “single token” wording is not implemented: fusion is a string operation. Actual tokenizer subword pieces depend on tokenizer and context.
- Cross-folder source captions only receive their own subject's rules; this port deliberately applies all mappings, as requested for multi-identity captions.
- Balancing is optional, despite release wording that suggests automatic balancing.
- Source accumulation averages composite LR factors; requested homogeneous optimizer categories intentionally reject that behavior.
- Source resume stores less than broad CLI claims: no explicit dataloader cursor or dedicated noise-generator state, and scheduler is reconstructed. Toolkit's existing raw optimizer/scheduler/RNG/sampler checkpoint infrastructure is retained.
- Fizgig's launcher disables loss-watch/per-image LR/look warmup and automatic recaptioning for Isolation. This port does not silently switch off compatible fork-specific masking, auxiliary losses, instrumentation, or runtime LR scales. Those are additional objectives/policies, not part of the minimal reference recipe; keep them independently disabled for a clean reference-style baseline, and record them when intentionally enabled.

## Existing fork boundaries

Initial Toolkit HEAD `13cbb75f5a2402fe3a178cc382fc7734654bd37f`, clean working tree. Audited `SDTrainer.py` existing DOP, full-frame preservation, manual/automatic masks and perceptual losses; `BaseSDTrainProcess.py` successful-update clock, raw training-state capture, sampler restoration, transient optimizer windows and category routing; `config_modules.py`; caption/latent/text-cache mixins and bucket/pool data loading. These are integration points, not replacement targets. Restrictions are explicit admission errors rather than silent feature disabling. Ordinary disabled-feature cache keys, loss/optimizer path, and dataset preparation remain unchanged.

## Locally observed Krea tokenization

Krea's `_load_text_encoder` loads `AutoTokenizer` and `Qwen2TokenizerFast` from `Qwen/Qwen3-VL-4B-Instruct` (or `model_kwargs.text_encoder_path`). `src/text_encoder.py::encode_krea_prompt` uses a fixed instruction prefix, truncates the prefix+prompt to `max_length+34`, appends a separately tokenized assistant suffix, and slices the first 34 feature positions. No isolation token is added to its vocabulary.

Executed both tokenizer constructors with `local_files_only=True` using the packaged interpreter; the local Transformers implementation reported `Qwen2Tokenizer` for each. No model/tokenizer download was requested. The production prefix encoded to exactly 34 tokens, matching the configured slice index. Representative prompt pieces (the `Ġ` glyph denotes leading whitespace):

| Input | Observed pieces |
|---|---|
| `zwxem, a woman smiling in a park` | `z`, `wx`, `em`, `,`, `Ġa`, `Ġwoman`, `Ġsmiling`, `Ġin`, `Ġa`, `Ġpark` |
| `zwxemwoman smiling in a park` | `z`, `wx`, `em`, `woman`, `Ġsmiling`, `Ġin`, `Ġa`, `Ġpark` |
| `zwxemwoman and qrxtwoman sitting together` | `z`, `wx`, `em`, `woman`, `Ġand`, `Ġqr`, `xt`, `woman`, `Ġsitting`, `Ġtogether` |
| `zwxemwoman and qrxtman sitting together` | `z`, `wx`, `em`, `woman`, `Ġand`, `Ġqr`, `xt`, `man`, `Ġsitting`, `Ġtogether` |
| `a woman and a man sitting together` | `a`, `Ġwoman`, `Ġand`, `Ġa`, `Ġman`, `Ġsitting`, `Ġtogether` |

The fused and preservation examples were also tokenized behind the actual production prefix; after the 34-position slice they had the same listed prompt pieces. This demonstrates string fusion and whitespace-token differences, **not a single-token identity**, encoder hidden-state behavior, or improved generations. Similar triggers can share subwords; distinct strings do not guarantee separable conditioning. Alternate tokenizer files/versions, punctuation, languages, and context can change splits. Long prompts can truncate identities. Inspect the exact tokenizer used by the intended run.
