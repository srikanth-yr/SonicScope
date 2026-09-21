const ui = {
  button: document.querySelector('#monitor-button'), state: document.querySelector('#state-text'),
  apiDot: document.querySelector('#api-dot'), apiStatus: document.querySelector('#api-status'),
  canvas: document.querySelector('#waveform'), sampleRate: document.querySelector('#sample-rate'),
  events: document.querySelector('#events'), voice: document.querySelector('#voice-toggle'),
};
const context = ui.canvas.getContext('2d');
const detector = {
  stream: null, audio: null, alertAudio: null, source: null, processor: null, processorSink: null, analyser: null,
  sending: false, lastAlert: {}, active: false,
};

function eventName(label) { return label === 'footsteps' ? 'Footsteps detected' : 'Gunshots detected'; }

function updateCard(label, confirmed = false) {
  const recentlyAlerted = confirmed || Date.now() - (detector.lastAlert[label] || 0) < 3000;
  document.querySelector(`#${label}-state`).textContent = recentlyAlerted ? 'Detection confirmed — alert sent' : 'Listening for event';
  document.querySelector(`#${label}-card`).classList.toggle('active', recentlyAlerted);
}

async function playBeep() {
  const audio = detector.audio?.state && detector.audio.state !== 'closed'
    ? detector.audio
    : (detector.alertAudio?.state && detector.alertAudio.state !== 'closed' ? detector.alertAudio : new AudioContext());
  if (audio !== detector.audio) detector.alertAudio = audio;
  if (audio.state === 'suspended') await audio.resume();
  const oscillator = audio.createOscillator();
  const gain = audio.createGain();
  oscillator.type = 'sine'; oscillator.frequency.value = 880;
  gain.gain.setValueAtTime(0.0001, audio.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.12, audio.currentTime + 0.012);
  gain.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + 0.16);
  oscillator.connect(gain); gain.connect(audio.destination);
  oscillator.start(); oscillator.stop(audio.currentTime + 0.17);
}

function speak(label) {
  if (!('speechSynthesis' in window)) {
    ui.state.textContent = 'Audio beep is active, but this browser does not support voice feedback.';
    return;
  }
  const utterance = new SpeechSynthesisUtterance(eventName(label));
  utterance.lang = navigator.language || 'en-US'; utterance.rate = 1; utterance.pitch = 1; utterance.volume = 1;
  utterance.onerror = (event) => {
    if (event.error !== 'interrupted' && event.error !== 'canceled') ui.state.textContent = 'Voice feedback could not be played. Check browser and system sound settings.';
  };
  window.speechSynthesis.cancel();
  // Chrome can drop an utterance submitted in the same task as cancel().
  window.setTimeout(() => { window.speechSynthesis.resume(); window.speechSynthesis.speak(utterance); }, 190);
}

function alertUser(label) {
  if (!ui.voice.checked) return;
  playBeep().catch(() => { ui.state.textContent = 'Audio alerts require permission to play sound.'; });
  speak(label);
}

function runDemo(label) {
  const button = document.querySelector(`#demo-${label}`);
  const sample = new Audio(`/api/demo-audio/${label}`);
  button.disabled = true;
  sample.addEventListener('ended', () => {
    detector.lastAlert[label] = Date.now();
    updateCard(label, true); logEvent(label);
    // The sample is the demo sound; follow it with only the requested voice feedback.
    if (ui.voice.checked) speak(label);
    button.disabled = false;
  }, { once: true });
  sample.addEventListener('error', () => {
    ui.state.textContent = 'The demo sound could not be played.';
    button.disabled = false;
  }, { once: true });
  sample.play().catch(() => {
    ui.state.textContent = 'The demo sound needs browser audio permission.';
    button.disabled = false;
  });
}

function logEvent(label) {
  const empty = ui.events.querySelector('.empty-log'); if (empty) empty.remove();
  const item = document.createElement('li');
  item.innerHTML = `<strong>${eventName(label)}</strong><span class="time">${new Date().toLocaleTimeString()}</span>`;
  ui.events.prepend(item); while (ui.events.children.length > 8) ui.events.lastElementChild.remove();
}

function confirmed(label, detected) {
  const elapsed = Date.now() - (detector.lastAlert[label] || 0);
  if (detected && elapsed > 5000) {
    detector.lastAlert[label] = Date.now(); alertUser(label); logEvent(label); return true;
  }
  return false;
}

function drawWaveform() {
  requestAnimationFrame(drawWaveform);
  const width = ui.canvas.width, height = ui.canvas.height; context.fillStyle = '#0c1419'; context.fillRect(0, 0, width, height);
  context.strokeStyle = '#23343c'; context.lineWidth = 1; context.beginPath(); context.moveTo(0, height / 2); context.lineTo(width, height / 2); context.stroke();
  if (!detector.analyser) return;
  const points = new Uint8Array(detector.analyser.fftSize); detector.analyser.getByteTimeDomainData(points);
  context.strokeStyle = '#4cc9a6'; context.lineWidth = 2; context.beginPath();
  points.forEach((value, index) => { const x = index * width / (points.length - 1); const y = (value / 255) * height; index ? context.lineTo(x, y) : context.moveTo(x, y); }); context.stroke();
}

async function sendAudio(samples, rate) {
  if (detector.sending) return; detector.sending = true;
  const pcm = new Int16Array(samples.length); for (let i = 0; i < samples.length; i++) pcm[i] = Math.max(-1, Math.min(1, samples[i])) * 32767;
  const bytes = new Uint8Array(pcm.buffer); let binary = ''; for (const byte of bytes) binary += String.fromCharCode(byte);
  try {
    const response = await fetch('/api/infer', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pcm16: btoa(binary), sample_rate: rate }) });
    if (!response.ok) throw new Error('Inference unavailable');
    const result = await response.json();
    for (const label of ['footsteps', 'gunshots']) updateCard(label, confirmed(label, result.detections?.[label]));
  } catch (error) { ui.state.textContent = 'Connection to the inference service was interrupted.'; }
  finally { detector.sending = false; }
}

async function start() {
  try {
    detector.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    detector.audio = new AudioContext(); await detector.audio.resume();
    detector.source = detector.audio.createMediaStreamSource(detector.stream); detector.analyser = detector.audio.createAnalyser(); detector.analyser.fftSize = 2048;
    detector.processor = detector.audio.createScriptProcessor(4096, 1, 1); detector.processorSink = detector.audio.createGain(); detector.processorSink.gain.value = 0;
    const buffer = []; const framesNeeded = Math.ceil(detector.audio.sampleRate * 2); const hopSamples = Math.ceil(detector.audio.sampleRate * 0.5);
    // A gunshot can be shorter than one recording block.  Analyze overlapping
    // two-second windows so a shot near a block boundary is still fully heard.
    detector.processor.onaudioprocess = (event) => {
      buffer.push(...event.inputBuffer.getChannelData(0));
      if (buffer.length >= framesNeeded) {
        sendAudio(buffer.slice(0, framesNeeded), detector.audio.sampleRate);
        buffer.splice(0, hopSamples);
      }
    };
    detector.source.connect(detector.analyser); detector.source.connect(detector.processor); detector.processor.connect(detector.processorSink); detector.processorSink.connect(detector.audio.destination);
    if ('speechSynthesis' in window) { window.speechSynthesis.getVoices(); window.speechSynthesis.resume(); }
    const emptyLog = ui.events.querySelector('.empty-log');
    if (emptyLog) emptyLog.textContent = 'Listening for confirmed detections.';
    detector.active = true; ui.button.textContent = 'Stop monitoring'; ui.button.classList.add('running'); ui.state.textContent = 'Live microphone monitoring is active. Voice background suppression is enabled.'; ui.sampleRate.textContent = `${detector.audio.sampleRate} Hz`;
  } catch (error) { ui.state.textContent = 'Microphone access is required to monitor live audio.'; }
}

function stop() {
  detector.stream?.getTracks().forEach(track => track.stop()); detector.audio?.close();
  detector.active = false; detector.analyser = null; detector.audio = null; window.speechSynthesis?.cancel();
  ui.button.textContent = 'Start monitoring'; ui.button.classList.remove('running'); ui.state.textContent = 'Monitoring stopped.';
}

ui.button.addEventListener('click', () => detector.active ? stop() : start());
document.querySelector('#test-alert').addEventListener('click', () => { alertUser('footsteps'); logEvent('footsteps'); });
document.querySelector('#demo-footsteps').addEventListener('click', () => runDemo('footsteps'));
document.querySelector('#demo-gunshots').addEventListener('click', () => runDemo('gunshots'));
document.querySelector('#clear-log').addEventListener('click', () => { ui.events.innerHTML = '<li class="empty-log">No recent detections.</li>'; });
async function health() { try { const response = await fetch('/api/health'); const data = await response.json(); const ready = data.status === 'ready'; ui.apiDot.className = `status-dot ${ready ? 'ready' : 'offline'}`; ui.apiStatus.textContent = ready ? 'Model ready' : 'Model unavailable'; ui.button.disabled = !ready; if (ready) ui.state.textContent = 'Model loaded. Ready to monitor live audio.'; } catch { ui.apiStatus.textContent = 'Service offline'; } }
drawWaveform(); health();
