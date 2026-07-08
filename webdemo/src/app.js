// PocketTTS webdemo application modules, AudioMass-style: each IIFE talks to
// the rest of the app only through PKTTS.fireEvent / listenFor.
(function (w, app) {
'use strict';

var SR = 24000;
var ASSET_BASE = String (w.PKTTS_ASSET_BASE || '').replace (/\/+$/, '');

function assetUrl (path) {
	path = String (path || '').replace (/^\/+/, '');
	return ASSET_BASE ? ASSET_BASE + '/' + path : new URL (path, w.location.href).href;
}

	function moduleWorker (url) {
		var u = new URL (url, w.location.href);
		if (u.origin === w.location.origin) return new Worker (u.href, { type: 'module' });
		var code = 'import ' + JSON.stringify (u.href) + ';';
		var blob = new Blob ([code], { type: 'text/javascript' });
		var blobUrl = URL.createObjectURL (blob);
		var worker = new Worker (blobUrl, { type: 'module' });
		setTimeout (function () { URL.revokeObjectURL (blobUrl); }, 30000);
		return worker;
	}

function isMobileBrowser () {
	var ua = navigator.userAgent || '';
	return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test (ua) ||
		(navigator.maxTouchPoints > 1 && /Macintosh/i.test (ua)) ||
		(w.matchMedia && w.matchMedia ('(pointer: coarse)').matches &&
			Math.min (screen.width || 9999, screen.height || 9999) <= 900);
}

app.isMobileBrowser = isMobileBrowser;
app.assetUrl = assetUrl;
app.assetBase = ASSET_BASE;

function copyBuffer (buf) {
	if (!buf) return null;
	if (buf instanceof ArrayBuffer) return buf.slice (0);
	if (ArrayBuffer.isView (buf)) {
		return buf.buffer.slice (buf.byteOffset, buf.byteOffset + buf.byteLength);
	}
	return null;
}

// ── EngineBridge: the TTS worker <-> event bus ──────────────────────────────
(function () {
	// Engine selection: ?engine=native|ts. Default is "native" (the full
	// pocket_tts.cpp compiled with Emscripten); "ts" is the JS orchestration.
	// Only an explicit query param is persisted (key v2: earlier auto-persist
	// wrote 'ts' and would pin old visitors to the slow engine).
	var qp = new URLSearchParams (w.location.search).get ('engine');
	if (qp === 'ts' || qp === 'native' || qp === 'split') {
		try { localStorage.setItem ('pktts_engine2', qp); } catch (e) {}
	}
	var engine = qp || localStorage.getItem ('pktts_engine2') || 'native';
	if (engine !== 'ts' && engine !== 'split') engine = 'native';
	// Phones always default to native: ts exceeds Safari's memory budget and
	// split ties on speed while costing +300MB. An explicit ?engine= param
	// still wins (testing); only persisted pins are ignored.
	if (engine !== 'native' && !qp && w.matchMedia ('(max-width: 720px)').matches) engine = 'native';
	app.state.engine = engine;
	var worker = moduleWorker (assetUrl (engine === 'split' ? 'src/tts-worker-split.js'
		: engine === 'native' ? 'src/tts-worker-native.js'
		: 'src/tts-worker.js'));

	worker.onerror = function (e) {
		// a wasm OOM abort kills the module irrecoverably - say so plainly
		var failedCloneKey = cloneInFlightKey;
		cloneInFlightKey = null;
		app.fireEvent ('EngineError', { during: 'engine',
			key: failedCloneKey,
			message: (e && e.message) || 'engine crashed - reload the page' });
	};
	worker.onmessage = function (e) {
		var d = e.data;
		switch (d.type) {
			case 'progress':    app.fireEvent ('EngineProgress', d); break;
			case 'ready':
				console.log ('[pktts] engine=' + engine + ' variant=' + (d.variant || '-') +
					' pool=' + (d.pool || '-'));
				app.state.engineInfo = d.variant
					? engine + ' · ' + d.variant + ' · pool ' + d.pool +
					  (d.spin === 0 ? ' · nospin' : '')
					: engine;
				app.fireEvent ('EngineProgress', { label: 'Preparing voices', pct: 0.98 });
				worker.postMessage ({ type: 'loadPresets', presets: [
					{ key: 'alba', label: 'ALBA',
					  url: assetUrl ('presets/alba.emb') }
				]});
				app.fireEvent ('VoicesRestoreRequest');
				break;
			case 'voice':
				if (cloneInFlightKey && d.key === cloneInFlightKey) cloneInFlightKey = null;
				app.state.ready = true;
				app.fireEvent ('VoiceReady', d);
				app.fireEvent ('EngineReady');
				// Silent prewarm on the preset: the first Speak skips the
				// one-time conditioning/allocation/tier-up costs. (Keyed on
				// builtin, not on arrival order — restored cloned voices can
				// land first and used to swallow the trigger entirely.)
				if (d.builtin && !prewarmSent) {
					prewarmSent = true;
					worker.postMessage ({ type: 'prewarm', voiceKey: d.key });
				}
				break;
			case 'speakStarted': app.fireEvent ('SpeakStarted'); break;
			case 'chunk':        app.fireEvent ('SpeakChunk', new Float32Array (d.samples)); break;
			case 'speakDone':    app.fireEvent ('SpeakDone', d); break;
			case 'voiceExport':  app.fireEvent ('VoiceExportReady', d); break;
			case 'voiceExportError': app.fireEvent ('VoiceExportError', d); break;
			case 'bench':
				console.log ('[pktts] bench: AR ' + d.ar.toFixed (2) + 'ms/step, dec15 ' +
					d.dec15.toFixed (0) + 'ms (' + (d.dec15 / 15).toFixed (1) + 'ms/frame), pool=' +
					d.pool + ', variant=' + d.variant);
				app.fireEvent ('EngineBench', d);
				break;
			case 'error':
				if (cloneInFlightKey && d.during === 'clone') {
					d.key = cloneInFlightKey;
					cloneInFlightKey = null;
				}
				app.fireEvent ('EngineError', d);
				break;
		}
	};

	var prewarmSent = false;
	var shuttingDown = false;
	var cloneInFlightKey = null;
	function shutdownPage (event) {
		if (event && event.persisted) return;
		if (shuttingDown) return;
		shuttingDown = true;
		app.state.speaking = false;
		app.state.playing = false;
		app.fireEvent ('PageClosing');
		try { worker.postMessage ({ type: 'shutdown' }); } catch (e) {}
		try { worker.postMessage ({ type: 'stop' }); } catch (e) {}
		try { worker.terminate (); } catch (e) {}
	}
	w.addEventListener ('pagehide', shutdownPage, { capture: true });
	w.addEventListener ('pageshow', function (event) {
		if (event.persisted && shuttingDown) w.location.reload ();
	}, { capture: true });

	app.listenFor ('StopRequest', function () {
		if (shuttingDown) return;
		worker.postMessage ({ type: 'stop' });
	});
	// Warm a voice the moment it's selected: by the time the user has typed
	// (or reached the Speak button), its conditioning cache is ready.
	app.listenFor ('VoiceSelected', function (key) {
		if (shuttingDown) return;
		if (app.state.ready) worker.postMessage ({ type: 'prewarmVoice', voiceKey: key });
	});
	app.listenFor ('SpeakRequest', function (req) {
		if (shuttingDown) return;
		worker.postMessage ({ type: 'speak', text: req.text, voiceKey: req.voiceKey,
			temperature: (req.temperature === 0 || req.temperature > 0) ? req.temperature : 0.45,
			seed: (Math.random () * 0xffffffff) >>> 0 });
	});
	app.listenFor ('CloneRequest', function (req) {
		if (shuttingDown) return;
		if (cloneInFlightKey) {
			app.fireEvent ('EngineProgress', { label: 'Voice cloning is already running', pct: 0.5 });
			return;
		}
		cloneInFlightKey = req.key;
		try {
			var copy = req.audio.slice ();
			worker.postMessage ({ type: 'clone', key: req.key, label: req.label, audio: copy },
				[ copy.buffer ]);
		} catch (err) {
			cloneInFlightKey = null;
			app.fireEvent ('EngineError', { during: 'clone', key: req.key,
				message: String (err && err.message || err) });
		}
	});
	app.listenFor ('VoiceRestore', function (v) {
		if (shuttingDown) return;
		worker.postMessage ({ type: 'restoreVoice', key: v.key, label: v.label,
			embDims: v.embDims, emb: v.emb }, [ v.emb ]);
	});
	app.listenFor ('VoiceDrop', function (key) {
		if (shuttingDown) return;
		worker.postMessage ({ type: 'dropVoice', key: key });
	});
	app.listenFor ('RequestVoiceExport', function (v) {
		if (shuttingDown) return false;
		worker.postMessage ({ type: 'exportVoice', key: v.key, label: v.label });
		return true;
	});

		// On-device tuning knobs: ?pool=N (thread pool), ?spin=0 (no spin-wait),
		// ?variant=fixed|growth (memory layout), ?bench=1 (isolated timings).
		var q = new URLSearchParams (w.location.search);
		var spinParam = q.get ('spin');
		var spinDefault = /Windows/i.test (navigator.userAgent || '') ? 0 : 1;
		worker.postMessage ({ type: 'init',
			assetBase: ASSET_BASE,
			modelsUrl: assetUrl ('models').replace (/\/$/, ''),
			threads: Math.min (4, (navigator.hardwareConcurrency || 4)),
			pool: parseInt (q.get ('pool'), 10) || 0,
			arPool: parseInt (q.get ('arpool'), 10) || 0,
			decPool: parseInt (q.get ('decpool'), 10) || 0,
			spin: spinParam === '0' ? 0 : spinParam === '1' ? 1 : spinDefault,
			variant: q.get ('variant') || '',
			bench: q.get ('bench') === '1' });
	})();

// ── VoicesStore: OPFS persistence for cloned voices ─────────────────────────
(function () {
	function dirp () { return navigator.storage.getDirectory ().then (function (r) {
		return r.getDirectoryHandle ('voices', { create: true }); }); }

	app.listenFor ('VoicePersist', function (v) {
		var copy = copyBuffer (v.emb);
		if (copy) {
			app.state.voicePayloads[v.key] = {
				key: v.key, label: v.label, embDims: v.embDims || null, emb: copy
			};
		}
		dirp ().then (function (dir) {
			return dir.getFileHandle (v.key + '.json', { create: true }).then (function (fh) {
				return fh.createWritable ();
			}).then (function (ws) {
				var meta = JSON.stringify ({ key: v.key, label: v.label, embDims: v.embDims });
				return ws.write (meta).then (function () { return ws.close (); });
			}).then (function () {
				return dir.getFileHandle (v.key + '.bin', { create: true });
			}).then (function (fh) { return fh.createWritable (); })
			.then (function (ws) {
				return ws.write (v.emb).then (function () { return ws.close (); });
			});
		}).catch (function () { /* OPFS unavailable: session-only voices */ });
	});

	app.listenFor ('VoiceDrop', function (key) {
		delete app.state.voicePayloads[key];
		dirp ().then (function (dir) {
			return Promise.allSettled ([
				dir.removeEntry (key + '.json'),
				dir.removeEntry (key + '.bin')
			]);
		}).catch (function () {});
	});

	app.listenFor ('VoicesRestoreRequest', function () {
		dirp ().then (async function (dir) {
			for await (var [name, fh] of dir.entries ()) {
				if (!/\.json$/.test (name)) continue;
				try {
					var meta = JSON.parse (await (await fh.getFile ()).text ());
					var bin = await (await (await dir.getFileHandle (meta.key + '.bin')).getFile ()).arrayBuffer ();
					app.state.voicePayloads[meta.key] = {
						key: meta.key, label: meta.label, embDims: meta.embDims || null, emb: bin.slice (0)
					};
					app.fireEvent ('VoiceRestore', { key: meta.key, label: meta.label,
						embDims: meta.embDims, emb: bin });
				} catch (e) { /* skip broken entries */ }
			}
		}).catch (function () {});
	});
})();

// ── Player: streaming playback through an AudioWorklet ring buffer ──────────
	(function () {
		var ctx = null, node = null, pending = [];
		var closing = false;
		var idleCloseTimer = 0;
		var playbackActive = false;

	// Mobile pre-buffer: phones generate closer to realtime and can stutter
	// if playback starts on the first chunk. Hold ~1s of audio before the
	// worklet starts draining, so brief generation dips don't underrun.
		var isMobile = w.matchMedia && w.matchMedia ('(max-width: 720px)').matches;
		var PREBUFFER_SAMPLES = SR * 0.8; // 800ms at 24kHz
		var preBuf = [], preCount = 0, buffering = false;

		function clearIdleClose () {
			if (idleCloseTimer) clearTimeout (idleCloseTimer);
			idleCloseTimer = 0;
		}

		function closePlayer () {
			clearIdleClose ();
			playbackActive = false;
			preBuf = []; preCount = 0; buffering = false;
			var n = node, c = ctx;
			node = null; ctx = null;
			if (n) {
				try { n.port.postMessage ({ flush: true, end: true }); } catch (e) {}
				try { n.port.onmessage = null; } catch (e) {}
				try { n.port.close (); } catch (e) {}
				try { n.disconnect (); } catch (e) {}
			}
			if (c && c.state !== 'closed') {
				try { c.close ().catch (function () {}); } catch (e) {}
			}
		}

		function scheduleIdleClose () {
			clearIdleClose ();
			idleCloseTimer = setTimeout (closePlayer, 750);
		}

		function flushPreBuffer () {
			if (!node) return;
			for (var i = 0; i < preBuf.length; ++i) node.port.postMessage (preBuf[i]);
			preBuf = []; preCount = 0; buffering = false;
		}

	// Small synthesized room: exponentially decaying noise as the impulse
	// response. Playback-only — downloads and the AudioMass handoff stay dry.
	function makeImpulse (audioCtx, seconds, decayPow) {
		var rate = audioCtx.sampleRate;
		var len = Math.floor (rate * seconds);
		var buf = audioCtx.createBuffer (2, len, rate);
		for (var ch = 0; ch < 2; ++ch) {
			var d = buf.getChannelData (ch);
			for (var i = 0; i < len; ++i) {
				d[i] = (Math.random () * 2 - 1) * Math.pow (1 - i / len, decayPow);
			}
		}
		return buf;
	}

			function ensure () {
				if (closing) return Promise.reject (new Error ('page closing'));
				clearIdleClose ();
				if (ctx && !node) closePlayer ();
				if (ctx) return Promise.resolve ();
				ctx = new AudioContext ({ sampleRate: SR });
				return ctx.audioWorklet.addModule (assetUrl ('src/playback-worklet.js')).then (function () {
					node = new AudioWorkletNode (ctx, 'pktts-player');

			var dry = ctx.createGain ();
			dry.gain.value = 1.0;
			node.connect (dry);
			dry.connect (ctx.destination);

			var wetHP = ctx.createBiquadFilter ();
			wetHP.type = 'highpass';
			wetHP.frequency.value = 260; // keep the tail out of the mud
			var verb = ctx.createConvolver ();
			verb.buffer = makeImpulse (ctx, 1.3, 3.4);
				var wet = ctx.createGain ();
				wet.gain.value = 0.2;
				node.connect (wetHP);
				wetHP.connect (verb);
				verb.connect (wet);
				wet.connect (ctx.destination);

					node.port.onmessage = function (e) {
						if (e.data.ended) {
							playbackActive = false;
							app.fireEvent ('PlaybackEnded');
							scheduleIdleClose ();
						}
					else if (e.data.played !== undefined) app.fireEvent ('PlaybackProgress', e.data.played);
				};
				}).catch (function (err) {
					closePlayer ();
					throw err;
				});
			}

			function reportPlaybackError (err) {
				if (closing) return;
				playbackActive = false;
				console.warn ('[pktts] playback setup failed', err);
				app.fireEvent ('EngineError', { during: 'audio playback', message: String (err && err.message || err) });
			}
	
		app.listenFor ('SpeakStarted', function () {
			if (closing) return;
			playbackActive = true;
			preBuf = []; preCount = 0; buffering = isMobile;
			ensure ().then (function () {
				if (closing || !ctx || !node) return;
				ctx.resume ();
				node.port.postMessage ({ flush: true });
			}).catch (reportPlaybackError);
		});
	app.listenFor ('SpeakChunk', function (samples) {
		if (!node) return;
		var copy = samples.slice ();
		if (buffering) {
			preBuf.push (copy);
			preCount += copy.length;
			if (preCount >= PREBUFFER_SAMPLES) flushPreBuffer ();
			return;
		}
		node.port.postMessage (copy);
	});
	app.listenFor ('SpeakDone', function () {
		if (!node) return;
		if (buffering) flushPreBuffer (); // shorter than the prebuffer: release it all
		node.port.postMessage ({ end: true });
	});
		app.listenFor ('ReplayRequest', function () {
			var a = app.state.lastAudio;
			if (closing || !a || !a.length) return;
			ensure ().then (function () {
				if (closing || !ctx || !node) return;
				playbackActive = true;
				ctx.resume ();
				node.port.postMessage ({ flush: true });
				node.port.postMessage (a.slice ());
				node.port.postMessage ({ end: true });
				app.fireEvent ('ReplayStarted');
			}).catch (reportPlaybackError);
		});
			app.listenFor ('StopRequest', function () {
				if (node && playbackActive) {
					node.port.postMessage ({ flush: true, end: true });
					setTimeout (function () { app.fireEvent ('PlaybackEnded'); }, 0);
				}
			closePlayer ();
		});
	// Free the audio thread while the AudioMass editor is open: two live
	// AudioContexts (ours at 24k, the editor's at 48k) share one device and
	// cost realtime mixing headroom that reads as editor sluggishness.
		app.listenFor ('AudioMassOpen', function () {
			closePlayer ();
		});
	// While recording, silence our output context: on iOS the audio session
	// drops mic gain sharply when another context is playing (reads as
	// "recording too close to silent").
		app.listenFor ('RecordStarted', function () {
			closePlayer ();
		});
		app.listenFor ('PageClosing', function () {
			closing = true;
			closePlayer ();
		});
	})();

// ── Recorder: mic capture via the AudioMass worklet ─────────────────────────
(function () {
	var ctx = null, node = null, stream = null, chunks = [], recording = false;
	var stopRequested = false, startSeq = 0;

	function stopStream (s) {
		if (s) s.getTracks ().forEach (function (t) { t.stop (); });
	}
	function cleanupRecorder () {
		recording = false;
		stopStream (stream);
		if (ctx) ctx.close ().catch (function () {});
		ctx = null; node = null; stream = null; chunks = [];
	}

	app.listenFor ('RecordStart', function () {
		if (recording || stream || ctx) return;
		var mobile = isMobileBrowser ();
		stopRequested = false;
		var seq = ++startSeq;
		navigator.mediaDevices.getUserMedia ({ audio: {
			echoCancellation: false, noiseSuppression: false, autoGainControl: !mobile
		}}).then (function (s) {
			if (stopRequested || seq !== startSeq) {
				stopStream (s);
				return Promise.reject ({ canceled: true });
			}
			stream = s;
			ctx = new AudioContext ();
			ctx.resume (); // iOS creates contexts suspended even inside a gesture
			return ctx.audioWorklet.addModule (assetUrl ('vendor/recorder-worklet.js')).then (function () {
				if (stopRequested || seq !== startSeq) {
					cleanupRecorder ();
					return;
				}
				var srcNode = ctx.createMediaStreamSource (stream);
				node = new AudioWorkletNode (ctx, 'pk-recorder', { processorOptions: { size: 4096 } });
				chunks = [];
				recording = true;
				node.port.onmessage = function (e) {
					if (e.data === 0) { finish (); return; }
					if (!recording) return;
					chunks.push (new Float32Array (e.data));
					app.fireEvent ('RecordProgress', {
						seconds: chunks.length * 4096 / ctx.sampleRate,
						latest: chunks[chunks.length - 1]
					});
				};
				srcNode.connect (node);
				app.fireEvent ('RecordStarted', { sampleRate: ctx.sampleRate });
			});
		}).catch (function (err) {
			if ((err && err.canceled) || stopRequested || seq !== startSeq) return;
			app.fireEvent ('RecordError', String (err && err.message || err));
		});
	});

	function finish () {
		if (!ctx || !stream) return;
		var total = 0, i;
		for (i = 0; i < chunks.length; ++i) total += chunks[i].length;
		var all = new Float32Array (total), o = 0;
		for (i = 0; i < chunks.length; ++i) { all.set (chunks[i], o); o += chunks[i].length; }
		var rate = ctx.sampleRate;
		cleanupRecorder ();
		ctx = null; node = null; stream = null; chunks = [];
		app.fireEvent ('RecordDone', { audio: all, sampleRate: rate });
	}

	app.listenFor ('RecordStop', function () {
		stopRequested = true;
		startSeq++;
		if (!recording) { cleanupRecorder (); return; }
		recording = false;
		if (node) node.port.postMessage (0); // flush -> finish()
		else cleanupRecorder ();
	});
	app.listenFor ('PageClosing', function () {
		stopRequested = true;
		startSeq++;
		cleanupRecorder ();
	});
})();

// ── AudioTools: decode, resample, validate ──────────────────────────────────
(function () {
	app.decodeFile = function (file) {
		return file.arrayBuffer ().then (function (buf) {
			var probe = new AudioContext ();
			return probe.decodeAudioData (buf).then (function (audioBuf) {
				probe.close ();
				return audioBuf;
			});
		});
	};

	app.resampleTo24k = function (audioBuf) {
		var frames = Math.ceil (audioBuf.duration * SR);
		var off = new OfflineAudioContext (1, frames, SR);
		var src = off.createBufferSource ();
		src.buffer = audioBuf;
		src.connect (off.destination);
		src.start ();
		return off.startRendering ().then (function (rendered) {
			return rendered.getChannelData (0).slice ();
		});
	};

	app.resampleRaw = function (data, fromRate) {
		var buf = new AudioBuffer ({ length: data.length, numberOfChannels: 1, sampleRate: fromRate });
		buf.copyToChannel (data, 0);
		return app.resampleTo24k (buf);
	};

	app.levelMobileMicForClone = function (audio24k) {
		if (!isMobileBrowser ()) return { audio: audio24k, gain: 1 };

		var n = audio24k.length, i, peak = 0, sumSq = 0;
		for (i = 0; i < n; ++i) {
			var x = audio24k[i], a = Math.abs (x);
			sumSq += x * x;
			if (a > peak) peak = a;
		}

		var rms = Math.sqrt (sumSq / (n || 1));
		// Do not turn silence or room noise into "speech"; validation below
		// should still reject recordings that were genuinely too quiet.
		if (peak < 0.008 || rms < 0.0025) return { audio: audio24k, gain: 1 };

		var targetPeak = 0.82;
		var gain = Math.min (12, targetPeak / peak);
		if (gain <= 1.08) return { audio: audio24k, gain: 1 };

		var out = new Float32Array (n);
		for (i = 0; i < n; ++i) {
			var y = audio24k[i] * gain;
			out[i] = y < -0.98 ? -0.98 : (y > 0.98 ? 0.98 : y);
		}
		return { audio: out, gain: gain, rawRms: rms, rawPeak: peak };
	};

	// Voice-sample validation: hard errors block, warnings inform.
	app.validateVoice = function (audio24k) {
		var n = audio24k.length, i;
		var dur = n / SR;
		var errors = [], warnings = [];
		if (dur < 6) errors.push ('Too short — record at least 6 seconds (10–15s works best).');
		if (dur > 10) warnings.push ('Longer than 10s — only the first 10 seconds are used (trim to choose which).');

		var sumSq = 0, clipped = 0, peak = 0;
		for (i = 0; i < n; ++i) {
			var a = Math.abs (audio24k[i]);
			sumSq += audio24k[i] * audio24k[i];
			if (a > 0.985) clipped++;
			if (a > peak) peak = a;
		}
		var rms = Math.sqrt (sumSq / (n || 1));
		if (rms < 0.008) errors.push ('The recording is close to silent — try again, closer to the microphone.');
		if (clipped / n > 0.01) warnings.push ('The audio clips — lower the input level or speak a little further away.');

		// Speech-likeness: syllabic energy modulation. 50ms frame RMS series
		// of speech has high variance relative to its mean; steady tones,
		// music beds and broadband noise are much flatter.
		var win = Math.floor (SR * 0.05), frames = Math.floor (n / win);
		var series = [], mean = 0;
		for (i = 0; i < frames; ++i) {
			var s = 0;
			for (var j = i * win; j < (i + 1) * win; ++j) s += audio24k[j] * audio24k[j];
			var v = Math.sqrt (s / win);
			series.push (v); mean += v;
		}
		mean /= (frames || 1);
		var varSum = 0;
		for (i = 0; i < frames; ++i) varSum += (series[i] - mean) * (series[i] - mean);
		var cv = mean > 0 ? Math.sqrt (varSum / (frames || 1)) / mean : 0;
		var active = series.filter (function (v) { return v > mean * 0.3; }).length / (frames || 1);
		if (cv < 0.35) warnings.push ('This does not sound like plain speech (music or steady noise?) — cloning quality may suffer.');
		if (active < 0.25) warnings.push ('Mostly silence — trim dead air for a better clone.');

		return { ok: errors.length === 0, errors: errors, warnings: warnings,
			stats: { duration: dur, rms: rms, peak: peak, cv: cv } };
	};
})();

// ── Exporter: WAV + MP3 download, AudioMass handoff ─────────────────────────
(function () {
	function wavBlob (samples) {
		var pcm = new Int16Array (samples.length);
		for (var i = 0; i < samples.length; ++i) {
			var x = Math.max (-1, Math.min (1, samples[i]));
			pcm[i] = Math.round (x * 32767);
		}
		var head = new ArrayBuffer (44), dv = new DataView (head);
		function str (o, s) { for (var i = 0; i < s.length; ++i) dv.setUint8 (o + i, s.charCodeAt (i)); }
		str (0, 'RIFF'); dv.setUint32 (4, 36 + pcm.byteLength, true); str (8, 'WAVE');
		str (12, 'fmt '); dv.setUint32 (16, 16, true); dv.setUint16 (20, 1, true);
		dv.setUint16 (22, 1, true); dv.setUint32 (24, SR, true);
		dv.setUint32 (28, SR * 2, true); dv.setUint16 (32, 2, true); dv.setUint16 (34, 16, true);
		str (36, 'data'); dv.setUint32 (40, pcm.byteLength, true);
		return new Blob ([head, pcm.buffer], { type: 'audio/wav' });
	}

	function mp3Blob (samples) {
		var enc = new lamejs.Mp3Encoder (1, SR, 128);
		var pcm = new Int16Array (samples.length);
		for (var i = 0; i < samples.length; ++i) {
			var x = Math.max (-1, Math.min (1, samples[i]));
			pcm[i] = Math.round (x * 32767);
		}
		var parts = [], BLOCK = 1152;
		for (var o = 0; o < pcm.length; o += BLOCK) {
			var out = enc.encodeBuffer (pcm.subarray (o, o + BLOCK));
			if (out.length) parts.push (out);
		}
		var end = enc.flush ();
		if (end.length) parts.push (end);
		return new Blob (parts, { type: 'audio/mpeg' });
	}

	function download (blob, name) {
		var a = document.createElement ('a');
		a.href = URL.createObjectURL (blob);
		a.download = name;
		a.click ();
		setTimeout (function () { URL.revokeObjectURL (a.href); }, 5000);
	}

	function safeName (name) {
		return String (name || 'pocket-tts-voice')
			.trim ()
			.replace (/[/\\?%*:|"<>]/g, '-')
			.replace (/\s+/g, '-')
			.replace (/-+/g, '-')
			.replace (/^-|-$/g, '')
			.slice (0, 80) || 'pocket-tts-voice';
	}

	app.listenFor ('DownloadWav', function () {
		if (app.state.lastAudio) download (wavBlob (app.state.lastAudio), 'pocket-tts.wav');
	});
	app.listenFor ('DownloadMp3', function () {
		if (app.state.lastAudio) download (mp3Blob (app.state.lastAudio), 'pocket-tts.mp3');
	});
	app.listenFor ('DownloadVoice', function () {
		var v = app.state.voices.find (function (q) { return q.key === app.state.activeVoice; });
		if (!v || v.builtin) return false;
		return app.fireEvent ('RequestVoiceExport', { key: v.key, label: v.label }) !== false;
	});
	app.listenFor ('VoiceExportReady', function (pack) {
		var stem = safeName (pack.label || pack.key);
		var layout = {};
		(pack.files || []).forEach (function (f, i) {
			var name = stem + f.ext;
			if (f.kind === 'wav') layout.wav = 'voices/' + name;
			else layout[f.kind] = 'voices/.cache/' + name;
			setTimeout (function () {
				download (new Blob ([f.buffer], { type: f.type || 'application/octet-stream' }), name);
			}, i * 120);
		});
		setTimeout (function () {
			var meta = JSON.stringify ({
				name: stem,
				label: pack.label || '',
				source: 'PocketTTS-RAVEN web demo',
				native_layout: layout,
				usage: 'Copy the .wav to voices/ and cache files to voices/.cache/ with the same filename stem.'
			}, null, 2);
			download (new Blob ([meta], { type: 'application/json' }), stem + '.json');
		}, ((pack.files || []).length + 1) * 120);
	});
	app.listenFor ('VoiceExportError', function () {
		console.warn ('[pktts] custom voice export failed');
	});

	// Upsample 24k -> 48k before handing audio to AudioMass: iOS Safari's
	// OfflineAudioContext (which its pitch/stretch effects render through)
	// misbehaves at non-native sample rates.
	app.upsample48k = function (samples24k) {
		var len48 = samples24k.length * 2;
		var octx = new OfflineAudioContext (1, len48, 48000);
		var buf = octx.createBuffer (1, samples24k.length, SR);
		buf.getChannelData (0).set (samples24k);
		var src = octx.createBufferSource ();
		src.buffer = buf;
		src.connect (octx.destination);
		src.start ();
		return octx.startRendering ().then (function (rendered) {
			return rendered.getChannelData (0);
		});
	};
	app.wavBlob48k = function (samples24k) {
		return app.upsample48k (samples24k).then (function (s48) {
			var pcm = new Int16Array (s48.length);
			for (var i = 0; i < s48.length; ++i) {
				var x = Math.max (-1, Math.min (1, s48[i]));
				pcm[i] = Math.round (x * 32767);
			}
			var head = new ArrayBuffer (44), dv = new DataView (head);
			var w = function (o, t) { for (var k = 0; k < t.length; ++k) dv.setUint8 (o + k, t.charCodeAt (k)); };
			w (0, 'RIFF'); dv.setUint32 (4, 36 + pcm.byteLength, true); w (8, 'WAVE');
			w (12, 'fmt '); dv.setUint32 (16, 16, true); dv.setUint16 (20, 1, true);
			dv.setUint16 (22, 1, true); dv.setUint32 (24, 48000, true);
			dv.setUint32 (28, 96000, true); dv.setUint16 (32, 2, true); dv.setUint16 (34, 16, true);
			w (36, 'data'); dv.setUint32 (40, pcm.byteLength, true);
			return new Blob ([head, pcm.buffer], { type: 'audio/wav' });
		});
	};

	app.listenFor ('OpenInAudioMass', function () {
		if (!app.state.lastAudio) return;
		app.wavBlob48k (app.state.lastAudio).then (function (b) {
			return b.arrayBuffer ();
		}).then (function (buf) {
			app.fireEvent ('AudioMassOpen', buf); // ui.js owns the modal/iframe
		});
	});
})();

})(window, window.PKTTS);
