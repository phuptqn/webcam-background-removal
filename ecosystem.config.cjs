module.exports = {
  name: 'webcam_background_removal',
  script: 'serve',
  instances: 1,
  autorestart: true,
  watch: true,
  time: true,
  env: {
    NODE_ENV: 'development',
    PM2_SERVE_PATH: '.',
    PM2_SERVE_PORT: 3001,
    PM2_SERVE_SPA: 'true',
    PM2_SERVE_HOMEPAGE: '/index.html'
  }
};