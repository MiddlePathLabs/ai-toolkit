import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT } from './paths';

export interface PythonPathOptions {
  toolkitRoot?: string;
  platform?: NodeJS.Platform;
  env?: NodeJS.ProcessEnv;
  exists?: (candidate: string) => boolean;
}

const getCandidates = ({
  toolkitRoot,
  platform,
}: Required<Pick<PythonPathOptions, 'toolkitRoot' | 'platform'>>): string[] => {

  const executable = platform === 'win32' ? 'python.exe' : 'python';
  const embedded = 'python_embeded';
  const virtualEnvironmentExecutable = platform === 'win32' ? 'Scripts' : 'bin';
  return [
    path.join(toolkitRoot, '..', embedded, executable),
    path.join(toolkitRoot, embedded, executable),
    path.join(toolkitRoot, '.venv', virtualEnvironmentExecutable, executable),
    path.join(toolkitRoot, 'venv', virtualEnvironmentExecutable, executable),
  ];
};

// This is the one interpreter resolver used by admission, scripts, and launch.
// An explicit override is authoritative even when it points at a missing binary;
// the caller then receives the real launch/admission error instead of silently
// running a different environment.
export const resolvePythonPath = (options: PythonPathOptions = {}): string => {
  const platform = options.platform ?? process.platform;
  const toolkitRoot = options.toolkitRoot ?? TOOLKIT_ROOT;
  const env = options.env ?? process.env;
  const configured = env.AI_TOOLKIT_PYTHON?.trim();
  if (configured) return configured;

  const exists = options.exists ?? fs.existsSync;
  const candidates = getCandidates({ toolkitRoot, platform });
  const resolved = candidates.find(exists);
  if (resolved) return resolved;
  return platform === 'win32' ? 'python.exe' : 'python3';
};

// Windows detached processes need the console-free executable from the same
// environment. This derives from resolvePythonPath rather than maintaining a
// second candidate list.
export const resolveDetachedPythonPath = (options: PythonPathOptions = {}): string => {
  const pythonPath = resolvePythonPath(options);
  if ((options.platform ?? process.platform) !== 'win32') return pythonPath;

  const exists = options.exists ?? fs.existsSync;
  const pythonwPath = path.join(path.dirname(pythonPath), 'pythonw.exe');
  return exists(pythonwPath) ? pythonwPath : pythonPath;
};
