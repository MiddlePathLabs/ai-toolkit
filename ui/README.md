This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Sample matrix

The job's **Samples** tab keeps one column per configured sample and orders rows by training step. Raw-weight samples (`raw_weights: true`) stay in their configured columns alongside EMA samples. Missing or deleted samples leave empty cells instead of shifting the remaining columns.

Each thumbnail shows the training step, seed, and inference **Steps**; **Raw** identifies non-EMA samples. Video overlays hide during playback and return when paused or ended. Arrow keys in the viewer move between samples horizontally and between training steps in the same column vertically, skipping empty cells.

New training samples include their configured column index and resolved seed/steps in the filename. Existing `RAW_` samples with pass-local counters are mapped using the job's saved raw/non-raw sample configuration. Legacy seed/steps labels also use that configuration, so changing it can affect historical labels; unresolved random seeds display **Unknown**.

After updating, rebuild and restart the UI to load the new screen. Restart training when convenient to enable the new filename metadata; existing samples do not need renaming.

### Navigation and display

- Sticky column headings show the sample number, weight type, and a short prompt. Hover a heading to read the full prompt.
- **Thumbnail size** offers Fit, Small, Medium, and Large. Larger thumbnails scroll horizontally instead of wrapping samples into different rows.
- **Show metadata** controls overlays in the matrix, single-sample viewer, and comparison viewer.
- **Training step** filters to all steps, the latest step, or a specific step. **Jump to step** and **Go** restore all steps and scroll to the selected row.
- The arrow buttons move to the first or last visible row. Sample cards can also be opened with Enter or Space.

### Comparing samples

**Compare samples** opens two panes. The right pane can stay on this job or switch to another training job. With the same job on both sides, the panes share one training-row selector and keep independent sample selectors. The initial pair prefers available raw and non-raw samples with matching saved prompts and resolved seeds. Mismatched or unknown seeds are identified rather than treated as a controlled comparison. Non-raw weights are labeled EMA only when the job enables EMA.

Choosing another job gives each pane its own training step. The dialog lists every step for this job, not only the step selected in the matrix filter. **Sync steps** keeps both panes on the same step and leaves a pane empty when that job did not sample it. **Show nearest step** moves the empty pane to the closest sample and labels both step numbers. **Match prompt** selects the same prompt on the other job, preferring the same weight type and then the same resolved seed. The first cross-job pair uses trained, non-raw weights. Notices name prompt, seed, and step mismatches. If batch size, gradient accumulation, or datasets differ, the same step is not the same amount of training. Legacy `gradient_accumulation_steps` is not the same pace as `gradient_accumulation`: it counts every batch as a step. Sample files store the training step, not an epoch.

For two videos, **Play both**, **Pause both**, **Restart both**, and **Seek both videos** control both panes together. Playback pauses while either video buffers and stops at the shorter video's duration. Videos are muted. Images and text can also be compared; audio samples use independent players.

### Missing cells

- **Not generated** means a recorded planned output is not present.
- **Deleted** means a successful UI deletion was recorded. The label survives reload, including when every sample in a row was deleted.
- **Not available** means the cell has no media and no plan or deletion record, as can happen with older samples.

New training runs write expected filenames under `samples/.sample-plans` before generation. UI sample deletions write individual markers under `samples/.deleted`; dataset deletions do not. Restart training to enable plan recording. Earlier deletions and files removed outside the UI have no deletion marker and cannot be identified as **Deleted** retroactively.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
