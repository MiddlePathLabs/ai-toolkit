export interface ComparisonVideo {
  duration: number;
  currentTime: number;
  readyState: number;
  seeking: boolean;
  ended: boolean;
  error: { code: number; message: string } | null;
  play: () => Promise<void>;
  pause: () => void;
  addEventListener: (type: string, listener: EventListener) => void;
  removeEventListener: (type: string, listener: EventListener) => void;
}

export interface ComparisonPlaybackState {
  playing: boolean;
  pending: boolean;
  currentTime: number;
  duration: number;
  error: string | null;
}

export interface ComparisonPlayback {
  play: () => void;
  pause: () => void;
  seek: (time: number) => void;
  restart: () => void;
  dispose: () => void;
}

export const EMPTY_COMPARISON_PLAYBACK: ComparisonPlaybackState = {
  playing: false,
  pending: false,
  currentTime: 0,
  duration: 0,
  error: null,
};

interface FrameScheduler {
  request: (callback: () => void) => number;
  cancel: (handle: number) => void;
}

const MAX_DRIFT_SECONDS = 0.08;

export function getCommonVideoDuration(videos: ComparisonVideo[]): number {
  if (videos.some(video => !Number.isFinite(video.duration) || video.duration <= 0)) return 0;
  return videos.length ? Math.min(...videos.map(video => video.duration)) : 0;
}

export function createComparisonPlayback(
  videos: [ComparisonVideo, ComparisonVideo],
  onChange: (state: ComparisonPlaybackState) => void,
  scheduler: FrameScheduler = {
    request: callback => requestAnimationFrame(callback),
    cancel: handle => cancelAnimationFrame(handle),
  },
): ComparisonPlayback {
  let requested = false;
  let playing = false;
  let starting = false;
  let disposed = false;
  let generation = 0;
  let frame: number | null = null;
  let error: string | null = null;
  let lastState = { ...EMPTY_COMPARISON_PLAYBACK };
  const waiting = new Set<ComparisonVideo>();
  const listeners: (() => void)[] = [];

  function pauseVideos() {
    videos.forEach(video => video.pause());
  }

  function publish() {
    if (disposed) return;
    const duration = getCommonVideoDuration(videos);
    const state = {
      playing,
      pending: requested && !playing,
      currentTime: Math.min(duration, Math.max(0, videos[0].currentTime || 0)),
      duration,
      error,
    };
    if (
      state.playing === lastState.playing &&
      state.pending === lastState.pending &&
      state.duration === lastState.duration &&
      state.error === lastState.error &&
      Math.abs(state.currentTime - lastState.currentTime) < 0.04
    )
      return;
    lastState = state;
    onChange(state);
  }

  function schedule() {
    if (disposed || !requested || frame !== null) return;
    frame = scheduler.request(() => {
      frame = null;
      reconcile();
      publish();
      schedule();
    });
  }

  function hold() {
    generation++;
    starting = false;
    playing = false;
    pauseVideos();
    publish();
    schedule();
  }

  function fail(message: string) {
    requested = false;
    error = message;
    hold();
    if (frame !== null) scheduler.cancel(frame);
    frame = null;
  }

  function seek(time: number) {
    const duration = getCommonVideoDuration(videos);
    if (!duration || !Number.isFinite(time) || disposed) return;
    const target = Math.min(duration, Math.max(0, time));
    hold();
    try {
      videos.forEach(video => {
        if (Math.abs(video.currentTime - target) > 0.001) video.currentTime = target;
      });
    } catch {
      fail('Could not seek both videos. Reload the samples and try again.');
    }
    publish();
  }

  function finish() {
    requested = false;
    seek(getCommonVideoDuration(videos));
    hold();
    if (frame !== null) scheduler.cancel(frame);
    frame = null;
  }

  function start() {
    if (starting || playing || disposed) return;
    starting = true;
    const attempt = ++generation;
    let attempts: Promise<void>[];
    try {
      attempts = videos.map(video => video.play());
    } catch (cause) {
      const detail = cause instanceof Error ? ` ${cause.message}` : '';
      fail(`Could not play both videos.${detail}`);
      return;
    }
    Promise.all(attempts)
      .then(() => {
        if (attempt !== generation || disposed || !requested) {
          if (disposed || !requested || (!playing && !starting)) pauseVideos();
          return;
        }
        starting = false;
        playing = true;
        publish();
      })
      .catch(cause => {
        if (attempt !== generation || disposed) return;
        const detail = cause instanceof Error ? ` ${cause.message}` : '';
        fail(`Could not play both videos.${detail}`);
      });
  }

  function reconcile() {
    if (!requested || disposed) return;
    const duration = getCommonVideoDuration(videos);
    if (duration && videos.some(video => video.ended || video.currentTime >= duration)) {
      finish();
      return;
    }
    if (!duration || waiting.size || videos.some(video => video.seeking || video.readyState < 3)) {
      if (playing || starting) hold();
      return;
    }
    if (Math.abs(videos[0].currentTime - videos[1].currentTime) > MAX_DRIFT_SECONDS) {
      seek(videos[0].currentTime);
      return;
    }
    start();
  }

  function play() {
    if (disposed) return;
    requested = true;
    error = null;
    const duration = getCommonVideoDuration(videos);
    if (duration && videos.some(video => video.ended || video.currentTime >= duration)) seek(0);
    reconcile();
    publish();
    schedule();
  }

  function pause() {
    requested = false;
    hold();
    if (frame !== null) scheduler.cancel(frame);
    frame = null;
  }

  videos.forEach((video, index) => {
    function listen(type: string, handler: () => void) {
      video.addEventListener(type, handler);
      listeners.push(() => video.removeEventListener(type, handler));
    }
    const ready = () => {
      if (!video.seeking && video.readyState >= 3) waiting.delete(video);
      reconcile();
      publish();
    };
    const blocked = () => {
      waiting.add(video);
      if (requested) hold();
    };
    ['loadedmetadata', 'durationchange', 'loadeddata', 'canplay', 'canplaythrough', 'progress'].forEach(type =>
      listen(type, ready),
    );
    ['waiting', 'stalled', 'seeking'].forEach(type => listen(type, blocked));
    listen('seeked', ready);
    listen('timeupdate', () => {
      reconcile();
      publish();
    });
    listen('ended', finish);
    listen('error', () => fail(`${index === 0 ? 'Left' : 'Right'} video could not be loaded or decoded.`));
  });
  pauseVideos();
  publish();

  return {
    play,
    pause,
    seek,
    restart: () => seek(0),
    dispose: () => {
      disposed = true;
      requested = false;
      generation++;
      pauseVideos();
      if (frame !== null) scheduler.cancel(frame);
      frame = null;
      listeners.forEach(remove => remove());
      waiting.clear();
    },
  };
}
