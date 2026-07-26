// public/opencv-worker.js

let cvReadyPromise = null;


// =========================================================
// LOGGING
// =========================================================

function log(message) {
  self.postMessage({
    type: 'log',
    message,
  });
}


// =========================================================
// OPENCV INITIALIZATION
// =========================================================

function isCvReady() {
  return !!(
    self.cv &&
    typeof self.cv.Mat === 'function' &&
    typeof self.cv.cvtColor === 'function'
  );
}

function ensureCv() {
  // Already initialized
  if (isCvReady()) {
    return Promise.resolve();
  }

  // Initialization already running
  if (cvReadyPromise) {
    return cvReadyPromise;
  }

  cvReadyPromise = new Promise((resolve, reject) => {
    log('Loading OpenCV...');

    let finished = false;
    let poll = null;

    const timeout = setTimeout(() => {
      if (finished) return;

      finished = true;

      if (poll) {
        clearInterval(poll);
      }

      cvReadyPromise = null;

      reject(
        new Error(
          'OpenCV runtime initialization timed out'
        )
      );
    }, 45000);


    const complete = () => {
      if (finished) return;
      if (!isCvReady()) return;

      finished = true;

      clearTimeout(timeout);

      if (poll) {
        clearInterval(poll);
      }

      log('OpenCV initialized successfully');

      // CRITICAL:
      // resolve with NOTHING.
      // Never resolve with self.cv.
      resolve();
    };


    // Configure old Emscripten build before loading.
    self.Module = {
      onRuntimeInitialized() {
        log('Emscripten runtime initialized');

        setTimeout(
          complete,
          0
        );
      },

      print(text) {
        console.log(
          '[OpenCV]',
          text
        );
      },

      printErr(text) {
        console.warn(
          '[OpenCV]',
          text
        );
      },
    };


    try {
      importScripts('/opencv.js');

      log('opencv.js downloaded');


      // Some builds initialize synchronously.
      if (isCvReady()) {
        complete();
        return;
      }


      // Some OpenCV builds expose their own callback.
      if (self.cv) {
        const oldCallback =
          self.cv.onRuntimeInitialized;

        self.cv.onRuntimeInitialized = () => {
          log(
            'cv.onRuntimeInitialized fired'
          );

          if (
            typeof oldCallback === 'function'
          ) {
            try {
              oldCallback();
            } catch (error) {
              console.warn(
                '[opencv-worker] Previous callback failed:',
                error
              );
            }
          }

          setTimeout(
            complete,
            0
          );
        };
      }


      // Final fallback.
      poll = setInterval(() => {
        if (finished) {
          clearInterval(poll);
          return;
        }

        if (isCvReady()) {
          complete();
        }
      }, 100);

    } catch (error) {
      finished = true;

      clearTimeout(timeout);

      if (poll) {
        clearInterval(poll);
      }

      cvReadyPromise = null;

      reject(
        new Error(
          'Failed to load OpenCV: ' +
          (
            error?.message ||
            String(error)
          )
        )
      );
    }
  });


  return cvReadyPromise;
}

// =========================================================
// SAFE DETECTOR WRAPPER
// =========================================================

function safeRun(fn, label) {
  try {
    return fn();
  } catch (error) {
    console.error(
      `[opencv-worker] ${label} failed:`,
      error
    );

    return null;
  }
}


// =========================================================
// CROSSING DETECTION
// =========================================================

function runCrossing(cv, src) {
  let roi = null;
  let gray = null;
  let blurred = null;
  let edges = null;
  let lines = null;

  try {
    const startY =
      Math.floor(src.rows * 0.40);

    const roiHeight =
      src.rows - startY;

    if (roiHeight <= 0) {
      throw new Error(
        'Invalid crossing ROI'
      );
    }

    roi = src.roi(
      new cv.Rect(
        0,
        startY,
        src.cols,
        roiHeight
      )
    );


    gray = new cv.Mat();

    cv.cvtColor(
      roi,
      gray,
      cv.COLOR_RGBA2GRAY
    );


    blurred = new cv.Mat();

    cv.GaussianBlur(
      gray,
      blurred,
      new cv.Size(5, 5),
      0
    );


    edges = new cv.Mat();

    cv.Canny(
      blurred,
      edges,
      50,
      150
    );


    lines = new cv.Mat();

    cv.HoughLinesP(
      edges,
      lines,
      1,
      Math.PI / 180,
      30,
      Math.max(
        20,
        Math.floor(
          roi.cols * 0.20
        )
      ),
      20
    );


    const horizontalLines = [];


    for (
      let i = 0;
      i < lines.rows;
      i++
    ) {
      const offset = i * 4;

      const x1 =
        lines.data32S[offset];

      const y1 =
        lines.data32S[offset + 1];

      const x2 =
        lines.data32S[offset + 2];

      const y2 =
        lines.data32S[offset + 3];


      const dx = x2 - x1;
      const dy = y2 - y1;


      const length =
        Math.sqrt(
          dx * dx +
          dy * dy
        );


      if (
        length <
        roi.cols * 0.15
      ) {
        continue;
      }


      const angle =
        Math.abs(
          Math.atan2(
            dy,
            dx
          ) *
          180 /
          Math.PI
        );


      if (
        angle < 20 ||
        angle > 160
      ) {
        horizontalLines.push(
          (y1 + y2) / 2
        );
      }
    }


    horizontalLines.sort(
      (a, b) => a - b
    );


    const distinct = [];


    for (
      const y of horizontalLines
    ) {
      if (
        !distinct.length ||
        Math.abs(
          y -
          distinct[
            distinct.length - 1
          ]
        ) > 8
      ) {
        distinct.push(y);
      }
    }


    return {
      detected:
        distinct.length >= 3,

      stripeCount:
        distinct.length,
    };

  } finally {
    roi?.delete();
    gray?.delete();
    blurred?.delete();
    edges?.delete();
    lines?.delete();
  }
}


// =========================================================
// PUDDLE / REFLECTIVE SURFACE
// =========================================================

function runPuddle(cv, src) {
  let roi = null;
  let rgb = null;
  let hsv = null;
  let low = null;
  let high = null;
  let mask = null;

  try {
    const startY =
      Math.floor(
        src.rows * 0.55
      );

    const roiHeight =
      src.rows - startY;


    if (roiHeight <= 0) {
      throw new Error(
        'Invalid puddle ROI'
      );
    }


    roi = src.roi(
      new cv.Rect(
        0,
        startY,
        src.cols,
        roiHeight
      )
    );


    rgb = new cv.Mat();

    cv.cvtColor(
      roi,
      rgb,
      cv.COLOR_RGBA2RGB
    );


    hsv = new cv.Mat();

    cv.cvtColor(
      rgb,
      hsv,
      cv.COLOR_RGB2HSV
    );


    low = new cv.Mat(
      hsv.rows,
      hsv.cols,
      hsv.type(),
      [0, 0, 170, 0]
    );


    high = new cv.Mat(
      hsv.rows,
      hsv.cols,
      hsv.type(),
      [180, 70, 255, 255]
    );


    mask = new cv.Mat();

    cv.inRange(
      hsv,
      low,
      high,
      mask
    );


    const brightPixels =
      cv.countNonZero(mask);


    const totalPixels =
      roi.rows * roi.cols;


    const coveragePct =
      Math.round(
        (
          brightPixels /
          totalPixels
        ) *
        1000
      ) / 10;


    return {
      detected:
        coveragePct > 12,

      coveragePct,
    };

  } finally {
    roi?.delete();
    rgb?.delete();
    hsv?.delete();
    low?.delete();
    high?.delete();
    mask?.delete();
  }
}


// =========================================================
// LIGHTING
// =========================================================

function runLighting(cv, src) {
  let gray = null;

  try {
    gray = new cv.Mat();

    cv.cvtColor(
      src,
      gray,
      cv.COLOR_RGBA2GRAY
    );


    const brightness =
      cv.mean(gray)[0];


    return {
      lowLight:
        brightness < 60,

      brightness:
        Math.round(brightness),
    };

  } finally {
    gray?.delete();
  }
}


// =========================================================
// CREATE SOURCE MAT
// =========================================================

function createSourceMat(
  cv,
  width,
  height,
  buffer
) {
  const pixels =
    new Uint8ClampedArray(
      buffer
    );


  const expected =
    width *
    height *
    4;


  if (
    pixels.length !==
    expected
  ) {
    throw new Error(
      `Pixel buffer mismatch: expected ${expected}, received ${pixels.length}`
    );
  }


  const src =
    new cv.Mat(
      height,
      width,
      cv.CV_8UC4
    );


  src.data.set(pixels);


  return src;
}


// =========================================================
// WORKER MESSAGE HANDLER
// =========================================================

self.onmessage = async (event) => {
  const {
    id,
    type,
    width,
    height,
    buffer,
  } = event.data || {};


  // =======================================================
  // WARMUP
  // =======================================================

  if (type === 'warmup') {
    try {
      log(
        `Warmup request ${id} started`
      );


      await ensureCv();


      /*
       * If you see this message, await ensureCv()
       * definitely completed.
       */
      log(
        `Warmup request ${id} complete`
      );


      self.postMessage({
        id,
        type: 'response',
        success: true,

        result: {
          warmedUp: true,
        },
      });

    } catch (error) {
      console.error(
        '[opencv-worker] Warmup failed:',
        error
      );


      self.postMessage({
        id,
        type: 'response',
        success: false,

        error:
          error?.message ||
          String(error),
      });
    }

    return;
  }


  // =======================================================
  // SCAN
  // =======================================================

  if (type === 'scan') {
    let src = null;

    try {
      log(
        `Scan request ${id} started`
      );

await ensureCv();

const cv = self.cv;

if (!cv || typeof cv.Mat !== 'function') {
  throw new Error(
    'OpenCV reported ready but cv API is unavailable'
  );
}


      if (
        !width ||
        !height ||
        !buffer
      ) {
        throw new Error(
          'Invalid image data received by worker'
        );
      }


      log(
        `Processing ${width}x${height} frame`
      );


      src =
        createSourceMat(
          cv,
          width,
          height,
          buffer
        );


      const crossing =
        safeRun(
          () =>
            runCrossing(
              cv,
              src
            ),
          'crossing'
        );


      const puddle =
        safeRun(
          () =>
            runPuddle(
              cv,
              src
            ),
          'puddle'
        );


      const lighting =
        safeRun(
          () =>
            runLighting(
              cv,
              src
            ),
          'lighting'
        );


      const result = {
        crossing,
        puddle,
        lighting,
      };


      log(
        `Scan request ${id} complete`
      );


      self.postMessage({
        id,
        type: 'response',
        success: true,
        result,
      });

    } catch (error) {
      console.error(
        '[opencv-worker] Scan failed:',
        error
      );


      self.postMessage({
        id,
        type: 'response',
        success: false,

        error:
          error?.message ||
          String(error),
      });

    } finally {
      src?.delete();
    }

    return;
  }


  // =======================================================
  // UNKNOWN REQUEST
  // =======================================================

  self.postMessage({
    id,
    type: 'response',
    success: false,

    error:
      `Unknown OpenCV request type: ${type}`,
  });
};