import assert from 'node:assert/strict';
import test from 'node:test';

import {
  depthModelOptions,
  getDepthToggleUpdates,
  isLowVramLocked,
} from '../src/app/jobs/new/depthUiPolicy';

test('enabling depth forces Low VRAM off and locks the control', () => {
  assert.deepEqual(getDepthToggleUpdates(true), {
    lossWeight: 0.001,
    lowVram: false,
  });
  assert.equal(isLowVramLocked(true), true);
});

test('disabling depth leaves Low VRAM unchanged and unlocks the control', () => {
  const updates = getDepthToggleUpdates(false);
  assert.equal(updates.lossWeight, 0);
  assert.equal('lowVram' in updates, false);
  assert.equal(isLowVramLocked(false), false);
});

test('the depth model selector offers Large, Base, and Small', () => {
  assert.deepEqual(
    depthModelOptions.map(option => option.value),
    [
      'depth-anything/Depth-Anything-V2-Large-hf',
      'depth-anything/Depth-Anything-V2-Base-hf',
      'depth-anything/Depth-Anything-V2-Small-hf',
    ],
  );
});
