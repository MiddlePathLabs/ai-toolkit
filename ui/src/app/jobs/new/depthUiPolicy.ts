export const depthModelOptions = [
  {
    value: 'depth-anything/Depth-Anything-V2-Large-hf',
    label: 'Depth Anything V2 Large',
  },
  {
    value: 'depth-anything/Depth-Anything-V2-Base-hf',
    label: 'Depth Anything V2 Base',
  },
  {
    value: 'depth-anything/Depth-Anything-V2-Small-hf',
    label: 'Depth Anything V2 Small',
  },
];

export function getDepthToggleUpdates(enabled: boolean): {
  lossWeight: number;
  lowVram?: boolean;
} {
  if (enabled) {
    return { lossWeight: 0.001, lowVram: false };
  }
  return { lossWeight: 0 };
}

export function isLowVramLocked(depthEnabled: boolean): boolean {
  return depthEnabled;
}
