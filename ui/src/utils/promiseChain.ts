// Minimal per-key async transaction serializer.
//
// Guards read-modify-write sequences against the logical lost-update race:
// two overlapping transactions that both read the same prior state and each
// write only their own change (individual atomic file writes do NOT prevent
// this -- the corruption is logical, not physical). Callers pass their own
// Map so unrelated features never share a chain; entries remove themselves
// once idle so long-lived processes don't accumulate them.
//
// Pure promise logic, no React/network -- ui/testing/identity-eval.test.ts
// covers the ordering guarantees (it guards the selection-save path).
export function chainOn<T>(chains: Map<string, Promise<unknown>>, key: string, task: () => Promise<T>): Promise<T> {
  // prior is either a fresh resolved promise or a stored never-rejecting
  // chain, so a failed earlier transaction can never block later ones
  const prior = chains.get(key) ?? Promise.resolve();
  const run = prior.then(task);
  const stored = run.then(
    () => undefined,
    () => undefined,
  );
  chains.set(key, stored);
  void stored.then(() => {
    if (chains.get(key) === stored) chains.delete(key);
  });
  return run;
}
