// src/lib/opencvCrossing.js
//
// Main-thread interface to /public/opencv-worker.js.
//
// OpenCV itself runs inside a Web Worker so loading opencv.js,
// initializing WASM, Canny, HoughLinesP, etc. never freeze React.

let worker = null;
let reqId = 0;

const pending = new Map();

const REQUEST_TIMEOUT_MS = 60000;

// Prevent React StrictMode from starting multiple warmups.
let warmupPromise = null;


// ---------------------------------------------------------
// WORKER
// ---------------------------------------------------------

function getWorker() {
  if (worker) {
    return worker;
  }

  console.log('[OpenCV] Creating worker...');

  worker = new Worker('/opencv-worker.js');


  // Messages coming back from the worker
  worker.onmessage = (event) => {
    const data = event.data || {};

    const {
      id,
      type,
      success,
      result,
      error,
      message,
    } = data;


    // Worker debug messages don't correspond to a request.
    if (type === 'log') {
      console.log(
        '[OpenCV Worker]',
        message
      );

      return;
    }
    if (type !== 'response') {
      console.warn(
        '[OpenCV] Unknown worker message:',
        data
      );
      return;
    }

    const entry = pending.get(id);

    if (!entry) {
      console.warn(
        '[OpenCV] Response received for unknown request:',
        id
      );

      return;
    }


    clearTimeout(entry.timer);

    pending.delete(id);


    if (success) {
      entry.resolve(result);
    } else {
      entry.reject(
        new Error(
          error ||
          'Unknown OpenCV worker error'
        )
      );
    }
  };


  // Fatal worker crash
  worker.onerror = (event) => {
    console.error(
      '[OpenCV] Worker crashed:',
      event
    );


    for (const [, entry] of pending) {
      clearTimeout(entry.timer);

      entry.reject(
        new Error(
          'OpenCV worker crashed: ' +
          (event.message || 'unknown error')
        )
      );
    }


    pending.clear();


    if (worker) {
      worker.terminate();
    }

    worker = null;

    // Allow a new warmup attempt after crash.
    warmupPromise = null;
  };


  return worker;
}


// ---------------------------------------------------------
// GENERIC REQUEST
// ---------------------------------------------------------

function sendRequest(
  message,
  transfer = []
) {
  const w = getWorker();

  const id = ++reqId;


  return new Promise(
    (resolve, reject) => {

      const timer = setTimeout(() => {

        if (!pending.has(id)) {
          return;
        }

        pending.delete(id);

        reject(
          new Error(
            'OpenCV request timed out after 60 seconds'
          )
        );

      }, REQUEST_TIMEOUT_MS);


      pending.set(id, {
        resolve,
        reject,
        timer,
      });


      try {

        w.postMessage(
          {
            id,
            ...message,
          },
          transfer
        );

      } catch (error) {

        clearTimeout(timer);

        pending.delete(id);

        reject(error);
      }
    }
  );
}


// ---------------------------------------------------------
// OPENCV WARMUP
// ---------------------------------------------------------

/**
 * Starts loading OpenCV in the background.
 *
 * Safe to call multiple times.
 *
 * React StrictMode may execute useEffect twice during development,
 * so all calls share the same warmup Promise.
 */
export function warmOpenCv() {

  if (warmupPromise) {
    return warmupPromise;
  }


  console.log(
    '[OpenCV] Starting warmup...'
  );


  warmupPromise = sendRequest({
    type: 'warmup',
  })
    .then((result) => {

      console.log(
        '[OpenCV] Warmup complete'
      );

      return result;
    })
    .catch((error) => {

      console.warn(
        '[OpenCV] Warmup failed:',
        error.message
      );

      // Allow future retry.
      warmupPromise = null;

      throw error;
    });


  return warmupPromise;
}


// ---------------------------------------------------------
// CAPTURE VIDEO FRAME
// ---------------------------------------------------------

function frameToImageData(
  videoEl,
  maxWidth = 320
) {

  if (!videoEl) {
    throw new Error(
      'Video element is unavailable'
    );
  }


  if (
    !videoEl.videoWidth ||
    !videoEl.videoHeight
  ) {
    throw new Error(
      'Camera frame is not ready yet'
    );
  }


  const scale = Math.min(
    1,
    maxWidth / videoEl.videoWidth
  );


  const width = Math.max(
    1,
    Math.round(
      videoEl.videoWidth * scale
    )
  );


  const height = Math.max(
    1,
    Math.round(
      videoEl.videoHeight * scale
    )
  );


  const canvas =
    document.createElement('canvas');


  canvas.width = width;
  canvas.height = height;


  const ctx = canvas.getContext(
    '2d',
    {
      willReadFrequently: true,
    }
  );


  if (!ctx) {
    throw new Error(
      'Could not create canvas context'
    );
  }


  ctx.drawImage(
    videoEl,
    0,
    0,
    width,
    height
  );


  return ctx.getImageData(
    0,
    0,
    width,
    height
  );
}


// ---------------------------------------------------------
// RUN OPENCV SCAN
// ---------------------------------------------------------

/**
 * Run classical OpenCV checks against the current camera frame.
 *
 * @param {HTMLVideoElement} videoEl
 *
 * @returns {
 *   Promise<{
 *      crossing: {
 *          detected: boolean,
 *          stripeCount: number
 *      } | null,
 *
 *      puddle: {
 *          detected: boolean,
 *          coveragePct: number
 *      } | null,
 *
 *      lighting: {
 *          lowLight: boolean,
 *          brightness: number
 *      } | null
 *   }>
 * }
 */
export async function runOpenCvScan(
  videoEl
) {

  if (!videoEl) {
    throw new Error(
      'Camera is unavailable'
    );
  }


  if (videoEl.readyState < 2) {
    throw new Error(
      'Camera is not ready yet'
    );
  }


  const imageData =
    frameToImageData(videoEl);


  console.log(
    '[OpenCV] Sending frame:',
    imageData.width,
    'x',
    imageData.height
  );


  /*
   * Transfer instead of copy.
   *
   * After postMessage this ArrayBuffer belongs
   * to the worker.
   */
  const buffer =
    imageData.data.buffer;


  const result =
    await sendRequest(
      {
        type: 'scan',

        width:
          imageData.width,

        height:
          imageData.height,

        buffer,
      },

      [buffer]
    );


  console.log(
    '[OpenCV] Scan result:',
    result
  );


  return result;
}