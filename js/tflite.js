const MODEL_INPUT_SIZE = {
  width: 256,
  height: 256,
}

let model;
let inputVideo;
let outputCanvas;
let outputCtx;
let segmentationMaskCanvas;
let segmentationMaskCtx;
let segmentationMask;
let inputTensor;
let prevOutputTensor;
const outputPixelsCount = MODEL_INPUT_SIZE.width * MODEL_INPUT_SIZE.height;
let outputWidth;
let outputHeight;
let background;
let startProcessingTime;
let fpsEle;
let fps = 0;

async function loadModel() {
  await tf.setBackend('webgl');
  model = await tflite.loadTFLiteModel('/pretrained/segvid_mnv2_port256_portrait_video.tflite');

  console.log('Model loaded');
}

async function initWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  inputVideo.srcObject = stream;

  return new Promise((resolve) => {
    inputVideo.onloadedmetadata = () => {
      inputVideo.play();
      resolve();
    };
  });
}

function preProcess() {
  !startProcessingTime && (startProcessingTime = Number(new Date()));

  segmentationMaskCtx.drawImage(
    inputVideo,
    0,
    0,
    outputWidth,
    outputHeight,
    0,
    0,
    MODEL_INPUT_SIZE.width,
    MODEL_INPUT_SIZE.height
  );

  let img = tf.browser.fromPixels(segmentationMaskCanvas, 3);
  img = tf.expandDims(tf.div(img, 255.0), 0);

  let prior = prevOutputTensor || tf.zeros([ MODEL_INPUT_SIZE.height, MODEL_INPUT_SIZE.width, 1 ])
  prior = tf.expandDims(prior, 0);

  inputTensor = tf.concat([ img, prior ], 3);
}

function postProcess() {
  // Mask
  outputCtx.globalCompositeOperation = 'copy';
  outputCtx.filter = 'blur(4px)';
  outputCtx.drawImage(
    segmentationMaskCanvas,
    0,
    0,
    MODEL_INPUT_SIZE.width,
    MODEL_INPUT_SIZE.height,
    0,
    0,
    outputWidth,
    outputHeight
  );

  // Foreground
  outputCtx.globalCompositeOperation = 'source-in';
  outputCtx.filter = 'none';
  outputCtx.drawImage(inputVideo, 0, 0);

  // Background
  outputCtx.globalCompositeOperation = 'destination-over';
  outputCtx.drawImage(
    background,
    0,
    0,
    outputWidth,
    outputHeight
  );

  outputCanvas.style.opacity = 1;

  // FPS
  fps += 1;
  const endTime = Number(new Date());

  if (endTime - startProcessingTime >= 1000) {
    fps = (fps * 1000 / (endTime - startProcessingTime)).toFixed(1);

    fpsEle.innerText = `frame: ${fps}/s`;

    fps = 0;
    startProcessingTime = endTime;
  }
}

async function processFrame() {
  preProcess();

  let outputTensor = model.predict(inputTensor);
  outputTensor = tf.reshape(outputTensor, [ MODEL_INPUT_SIZE.height, MODEL_INPUT_SIZE.width, 1 ]);

  const outputData = outputTensor.dataSync();
  prevOutputTensor = outputTensor;
  
  for (let i = 0; i < outputPixelsCount; i++) {
    segmentationMask.data[(i * 4) + 3] = outputData[i] * 255;
  }
  segmentationMaskCtx.putImageData(segmentationMask, 0, 0);

  // Cleanup
  inputTensor.dispose();
  outputTensor.dispose();

  postProcess();

  requestAnimationFrame(processFrame);
}

async function main() {
  inputVideo = document.getElementById('webcam');
  outputCanvas = document.getElementById('outputCanvas');
  outputCtx = outputCanvas.getContext('2d');
  outputWidth = inputVideo.width;
  outputHeight = inputVideo.height;

  segmentationMaskCanvas = document.createElement('canvas');
  segmentationMaskCanvas.width = MODEL_INPUT_SIZE.width;
  segmentationMaskCanvas.height = MODEL_INPUT_SIZE.height;
  segmentationMaskCtx = segmentationMaskCanvas.getContext('2d');
  segmentationMask = new ImageData(MODEL_INPUT_SIZE.width, MODEL_INPUT_SIZE.height);

  outputCanvas.height = outputHeight;
  outputCanvas.width = outputWidth;

  background = document.createElement('img');
  background.crossOrigin = 'anonymous';
  background.src = '/image/bg/bg1.jpg';

  fpsEle = document.getElementById('fps');

  await loadModel();
  await initWebcam();
  processFrame();
}

window.onload = function() {
  main();
}