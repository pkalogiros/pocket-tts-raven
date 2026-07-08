// DOM wiring for the PocketTTS webdemo — same bus, AudioMass-style.
(function (w, d, app) {
'use strict';

var SR = 24000;
function el (id) { return d.getElementById (id); }

// toast helper, available to all modules below
var toastTimer = null;
function toast (msg, ms) {
	var t = el ('toast');
	t.textContent = msg;
	t.classList.add ('show');
	clearTimeout (toastTimer);
	toastTimer = setTimeout (function () { t.classList.remove ('show'); }, ms || 3200);
}

function activeVoiceRecord () {
	return app.state.voices.find (function (v) { return v.key === app.state.activeVoice; }) || null;
}

function syncVoiceDownloadButton () {
	var b = el ('dl-voice');
	if (!b) return;
	var v = activeVoiceRecord ();
	b.hidden = (app.isMobileBrowser && app.isMobileBrowser ()) || app.state.engine === 'ts' || !v || !!v.builtin;
}

if (w.PKTTS_UNSUPPORTED) {
	function showUnsupported () {
		var overlay = el ('overlay');
		if (overlay) {
			overlay.classList.remove ('gone', 'hidden');
			overlay.classList.add ('unsupported');
		}
		var msg = el ('unsupported-msg');
		if (msg && w.PKTTS_UNSUPPORTED) {
			var body = msg.querySelector ('span');
			if (body) body.textContent = w.PKTTS_UNSUPPORTED;
		}
		var status = el ('status');
		if (status) status.textContent = 'Browser not supported';
	}
	if (d.readyState === 'loading') d.addEventListener ('DOMContentLoaded', showUnsupported);
	else showUnsupported ();
	return;
}

// ── Version stamp: which build is this client actually running? ─────────────
(function () {
	fetch ('version').then (function (r) { return r.text (); }).then (function (v) {
		console.log ('[pktts] server build: ' + v);
	}).catch (function () {});
})();

// ── Mobile default text: phones get a one-liner instead of the poem ─────────
(function () {
	if (!w.matchMedia ('(max-width: 720px)').matches) return;
	d.addEventListener ('DOMContentLoaded', function () {
		el ('text').value = "Beware: the nearest mirror hides a portal to the abyss\u2014look, but don't step!";
	});
	if (d.readyState !== 'loading') {
		el ('text').value = "Beware: the nearest mirror hides a portal to the abyss\u2014look, but don't step!";
	}
})();

// ── Theme toggle: gothic crypt (default) <-> boring modern ──────────────────
(function () {
	var btn = el ('theme-toggle');
	if (!btn) return;
	function apply () {
		var clean = d.documentElement.getAttribute ('data-theme') === 'clean';
		btn.textContent = clean ? 'make it cool!' : 'make it boring';
		btn.title = clean ? 'Restore the gothic 90s experience'
			: 'Too spooky? Switch to modern minimalist mode';
		var source = d.querySelector ('.guestbook');
		if (source) source.textContent = clean ? 'View source on Github' : '✍ sign the guestbook';
	}
	btn.onclick = function () {
		var clean = d.documentElement.getAttribute ('data-theme') === 'clean';
		if (clean) d.documentElement.removeAttribute ('data-theme');
		else d.documentElement.setAttribute ('data-theme', 'clean');
		try { localStorage.setItem ('pktts_theme', clean ? 'crypt' : 'clean'); } catch (e) {}
		apply ();
		app.fireEvent ('ThemeChange');
	};
	apply ();
})();

// ── The raven hops when poked ────────────────────────────────────────────────
(function () {
	d.querySelectorAll ('.pixel-raven').forEach (function (raven) {
		raven.addEventListener ('click', function () {
			raven.classList.remove ('hop');
			void raven.getBoundingClientRect (); // restart the animation
			raven.classList.add ('hop');
		});
		raven.addEventListener ('animationend', function () {
			raven.classList.remove ('hop');
		});
	});
})();

// ── Loading overlay ─────────────────────────────────────────────────────────
(function () {
	app.listenFor ('EngineProgress', function (p) {
		el ('load-label').textContent = p.label;
		el ('load-bar').style.width = Math.round ((p.pct || 0) * 100) + '%';
	});
	app.listenFor ('EngineReady', function () {
		var overlay = el ('overlay');
		overlay.classList.remove ('hidden');
		overlay.classList.add ('gone');
		setTimeout (function () {
			if (overlay.classList.contains ('gone')) overlay.classList.add ('hidden');
		}, 200);
	});
	app.listenFor ('EngineError', function (e) {
		el ('load-label').textContent = 'Engine error: ' + e.message;
		el ('status').textContent = 'Engine error during ' + e.during + ': ' + e.message;
	});
})();

// ── Voice chips ─────────────────────────────────────────────────────────────
(function () {
	function render () {
		var host = el ('voices');
		host.innerHTML = '';
		app.state.voices.forEach (function (v) {
			var b = d.createElement ('button');
			b.className = 'chip' + (app.state.activeVoice === v.key ? ' active' : '');
			b.appendChild (d.createTextNode (v.label));
			if (v.builtin) {
				var tag = d.createElement ('span');
				tag.className = 'tag';
				tag.textContent = 'preset';
				b.appendChild (tag);
			} else {
				var x = d.createElement ('span');
				x.className = 'x';
				x.textContent = '×';
				x.title = 'Remove this voice';
				x.onclick = function (e) {
					e.stopPropagation ();
					app.state.voices = app.state.voices.filter (function (q) { return q.key !== v.key; });
					if (app.state.activeVoice === v.key) {
						app.state.activeVoice = app.state.voices.length ? app.state.voices[0].key : null;
					}
					app.fireEvent ('VoiceDrop', v.key);
					toast ('Removed "' + v.label + '"');
					render ();
				};
				b.appendChild (x);
			}
			b.onclick = function () {
				app.state.activeVoice = v.key;
				app.state.userPickedVoice = true;
				app.fireEvent ('VoiceSelected', v.key);
			};
			host.appendChild (b);
		});
		var add = d.createElement ('button');
		add.className = 'chip add';
		add.textContent = '+ Clone a voice';
		add.title = 'Clone from an audio file or your microphone — processed on this device';
		add.onclick = function () { app.fireEvent ('CloneModalOpen'); };
		host.appendChild (add);
		syncVoiceDownloadButton ();
	}
	app.listenFor ('VoiceReady', function (v) {
		var isNew = !app.state.voices.some (function (x) { return x.key === v.key; });
		if (isNew) app.state.voices.push ({ key: v.key, label: v.label, builtin: v.builtin });
		// The preset is the default: it claims selection even if a restored
		// cloned voice arrived first — unless the user already chose one.
		if (!app.state.activeVoice || (v.builtin && !app.state.userPickedVoice)) {
			app.state.activeVoice = v.key;
		}
		if (!v.builtin && v.emb) {
			app.fireEvent ('VoicePersist', { key: v.key, label: v.label, embDims: v.embDims, emb: v.emb });
			toast ('Voice "' + v.label + '" is ready — hit Speak to try it');
		}
		render ();
	});
	app.listenFor ('VoiceSelected', render);
})();

	// ── Speak flow + last-audio assembly ────────────────────────────────────────
	(function () {
		var parts = [];
		var speechActive = false;
		var t0 = 0, firstChunkAt = 0;
	var RAVEN_REJECT = 'THE RAVEN DOES NOT APPROVE OF THIS';

		function assembleParts () {
			var total = 0, i;
			for (i = 0; i < parts.length; ++i) total += parts[i].length;
			if (!total) {
				parts = [];
				return null;
			}
			var all = new Float32Array (total), o = 0;
			for (i = 0; i < parts.length; ++i) { all.set (parts[i], o); o += parts[i].length; }
			parts = [];
			return { audio: all, total: total };
		}

	function decodeB64 (s) { return atob (s); }
	function decodeB64List (list) {
		return list.map (function (s) { return decodeB64 (s); });
	}

	function reEscape (s) {
		return s.replace (/[.*+?^${}()|[\]\\]/g, '\\$&');
	}

	function flexibleWordPattern (word) {
		return word.split ('').map (function (ch) {
			// Keep this narrow: tolerate separators/repeated letters, but only
			// inside directed-hostility patterns below.
			return reEscape (ch) + (ch === 'u' ? '*' : '+');
		}).join ('[^a-z0-9]*');
	}

	var severeSlurs = decodeB64List ([
		'bmlnZ2Vy', 'bmlnZ2E=', 'bmlnZ3Vo',
		'Y2hpbms=', 'c3BpYw==', 'a2lrZQ==', 'Z29vaw==',
		'd2V0YmFjaw==', 'cGFraQ==', 'cmFnaGVhZA==',
		'ZmFnZ290'
	]);
	var fBomb = decodeB64 ('ZnVjaw==');
	var roughInsult = decodeB64 ('c2hpdA==');
	var hostilePhrasePatterns = decodeB64List ([
		'a2lsbCB5b3Vyc2VsZg==', 'Z28ga2lsbCB5b3Vyc2VsZg==',
		'a3lz', 'Z28gZGll', 'Z28ganVtcCBvZmY=', 'anVtcCBvZmY='
	]).map (function (phrase) {
		return new RegExp ('\\b' + phrase.split (/\s+/).map (reEscape).join ('\\s+') + '\\b', 'i');
	});
	var blockedPatterns = [
		// Direct hostile commands. Isolated profanity is allowed.
		new RegExp ('\\b(?:go\\s+)?' + flexibleWordPattern (fBomb) + '\\s+(?:you|yourself)\\b', 'i'),
		/\b(?:go\s+)?f\s+(?:you|yourself)\b/i,
		new RegExp ('\\byou\\s+(?:piece\\s+of\\s+' + flexibleWordPattern (roughInsult) +
			'|worthless\\s+(?:piece\\s+of\\s+)?' + flexibleWordPattern (roughInsult) + ')\\b', 'i')
	].concat (hostilePhrasePatterns);

	function normalizeForBlocklist (text) {
		text = text.normalize ? text.normalize ('NFKC') : text;
		return text.replace (/[\u200B-\u200D\uFEFF]/g, '');
	}

	function foldLeetForBlocklist (text) {
		return text.toLowerCase ()
			.replace (/[4@]/g, 'a')
			.replace (/[3]/g, 'e')
			.replace (/[1!|]/g, 'i')
			.replace (/[0]/g, 'o')
			.replace (/[5$]/g, 's')
			.replace (/[7+]/g, 't');
	}

	function normalizeFilteredToken (raw) {
		var obfuscated = /[^A-Za-z0-9]/.test (raw) || /[013457@!|$+]/.test (raw);
		var repeated = /([A-Za-z])\1{2,}/.test (raw);
		var text = raw.toLowerCase ()
			.replace (/[4@]/g, 'a')
			.replace (/[3]/g, 'e')
			.replace (/[1!|]/g, 'i')
			.replace (/[0]/g, 'o')
			.replace (/[5$]/g, 's')
			.replace (/[7+]/g, 't')
			.replace (/[^a-z]/g, '?')
			.replace (/\?+/g, '?')
			.replace (/^\?+|\?+$/g, '');
		return {
			text: text,
			compact: text.replace (/\?/g, ''),
			obfuscated: obfuscated,
			repeated: repeated
		};
	}

	function collapseRepeats (s) {
		return s.replace (/([a-z])\1{2,}/g, '$1$1');
	}

	function editDistanceOneOrLess (a, b) {
		var i = 0, j = 0, edits = 0;
		while (i < a.length && j < b.length) {
			if (a[i] === '?' || a[i] === b[j]) { i++; j++; continue; }
			if (++edits > 1) return false;
			if (a.length > b.length) i++;
			else if (b.length > a.length) j++;
			else { i++; j++; }
		}
		edits += (a.length - i) + (b.length - j);
		return edits <= 1;
	}

	function edgeMatches (token, target) {
		var clean = token.replace (/\?/g, '');
		return clean.length &&
			clean[0] === target[0] &&
			clean[clean.length - 1] === target[target.length - 1];
	}

	function tokenMatchesSevereSlur (token, target, obfuscated) {
		if (token.compact === target) return true;
		if (token.compact.length > 4 && token.compact.endsWith ('s') &&
			token.compact.slice (0, -1) === target) return true;
		if (!obfuscated && !token.repeated) return false;

		var compactForms = [token.compact, collapseRepeats (token.compact)];
		if (token.compact.length > 4 && token.compact.endsWith ('s')) {
			compactForms.push (token.compact.slice (0, -1));
		}
		if (compactForms.some (function (form) {
			return edgeMatches (form, target) &&
				Math.abs (form.length - target.length) <= 1 &&
				editDistanceOneOrLess (form, target);
		})) return true;

		var wildcardForms = [token.text, collapseRepeats (token.text)];
		if (token.text.length > 4 && token.text.endsWith ('s')) wildcardForms.push (token.text.slice (0, -1));
		return wildcardForms.some (function (form) {
			return edgeMatches (form, target) &&
				Math.abs (form.length - target.length) <= 1 &&
				editDistanceOneOrLess (form, target);
		});
	}

	function containsSevereSlur (text) {
		var runs = text.match (/[A-Za-z0-9@#$*!|+\-_.~'\/\\]{3,28}/g) || [];
		return runs.some (function (raw) {
			var token = normalizeFilteredToken (raw);
			if (!token.text || !token.compact) return false;
			return severeSlurs.some (function (slur) {
				return tokenMatchesSevereSlur (token, slur, token.obfuscated);
			});
		});
	}

	function blockedText (text) {
		var normalized = normalizeForBlocklist (text);
		var folded = foldLeetForBlocklist (normalized);
		return containsSevereSlur (normalized) ||
			blockedPatterns.some (function (re) {
				return re.test (normalized) || re.test (folded);
			});
	}

	function speakableCheck (text) {
		// The 4k-piece vocab covers Latin text + punctuation; anything else
		// (emoji, CJK, …) has no voice pieces and will be skipped or mumbled.
		var odd = text.match (/[^\t\n\r\u0020-\u007E\u00A0-\u024F\u2010-\u2027\u20AC]/g);
		if (odd) {
			var uniq = Array.from (new Set (odd)).slice (0, 6).join (' ');
			toast ('Heads up: "' + uniq + '" can’t be spoken by this model and will be skipped.');
		}
	}

		function syncButton () {
			el ('speak').textContent = (app.state.speaking || app.state.playing) ? 'Stop' : 'Speak';
		}
		var lastSpeakRequestAt = 0;
		function requestSpeak () {
			if (app.state.speaking || app.state.playing) {
				// Stop halts generation AND playback; button returns to Speak
				// when the (flushed) playback signals it has ended.
				app.fireEvent ('StopRequest');
				return;
			}
			if (!app.state.ready) return;
			var now = performance.now ();
			if (now - lastSpeakRequestAt < 300) return;
			var text = el ('text').value.trim ();
			if (!text) { toast ('Type something first.'); return; }
			if (!app.state.activeVoice) { toast ('Pick a voice first.'); return; }
			if (blockedText (text)) {
				toast (RAVEN_REJECT, 4200);
			el ('status').textContent = RAVEN_REJECT;
				return;
			}
			lastSpeakRequestAt = now;
			speakableCheck (text);
			app.fireEvent ('SpeakRequest', { text: text, voiceKey: app.state.activeVoice,
				temperature: app.state.temperature });
	}
	el ('speak').onclick = requestSpeak;

	// temperature slider (persisted)
	(function () {
		var slider = el ('temp'), out = el ('temp-val');
		var saved = parseFloat (localStorage.getItem ('pktts_temp'));
		// Mobile default runs warmer: short phrases read better with more
		// expressiveness; desktop (long-form poem) stays steadier at 0.5.
		var defTemp = w.matchMedia ('(max-width: 720px)').matches ? 0.6 : 0.5;
		app.state.temperature = (saved >= 0.1 && saved <= 0.9) ? saved : defTemp;
		slider.value = app.state.temperature;
		out.textContent = app.state.temperature.toFixed (2);
		slider.oninput = function () {
			app.state.temperature = parseFloat (slider.value);
			out.textContent = app.state.temperature.toFixed (2);
			localStorage.setItem ('pktts_temp', slider.value);
			// Temperature applies at generation time; adjusting it against
			// already-generated audio deserves a gentle heads-up.
			if (app.state.playing || app.state.speaking) {
				toast ("Re-run generation to hear temperature's effect", 2600);
			}
		};
	})();
	d.addEventListener ('keydown', function (e) {
		if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault (); requestSpeak (); }
	});

	app.listenFor ('EngineReady', function () {
		el ('speak').disabled = false;
		el ('speak').title = 'Cmd/Ctrl+Enter';
		el ('status').textContent = 'Ready';
	});
	app.listenFor ('EngineRestarting', function () {
		el ('speak').disabled = true;
		el ('speak').title = 'Engine is reloading';
		el ('status').textContent = 'Reloading engine…';
	});
	app.listenFor ('EngineBench', function (b) {
		// visible on-device (?bench=1): per-frame budget is AR + dec/frame;
		// realtime needs < 80ms.
		var perFrame = b.ar + b.dec15 / 15;
		toast ('bench: AR ' + b.ar.toFixed (1) + 'ms + dec ' + (b.dec15 / 15).toFixed (1) +
			'ms/frame = ~' + (80 / perFrame).toFixed (1) + 'x realtime ceiling · pool ' +
			b.pool + ' · ' + b.variant, 12000);
	});

	app.listenFor ('SpeakStarted', function () {
		parts = [];
		speechActive = true;
		t0 = performance.now ();
		firstChunkAt = 0;
		stoppedByUser = false;
		app.state.speaking = true;
		app.state.playing = true;
		app.state.lastAudio = null;
		syncButton ();
		el ('status').textContent = 'Preparing voice…';
		el ('actions').classList.add ('disabled');
		el ('wave-hint').classList.add ('gone');
	});
	app.listenFor ('SpeakChunk', function (c) {
		if (!speechActive) return;
		if (!firstChunkAt) {
			firstChunkAt = performance.now ();
			el ('status').textContent = 'Speaking… (first audio in ' +
				Math.round (firstChunkAt - t0) + 'ms)';
		}
		parts.push (c);
	});
	app.listenFor ('SpeakDone', function () {
		if (!speechActive) { parts = []; return; }
		speechActive = false;
		var assembled = assembleParts ();
		if (!assembled) return;
		app.state.lastAudio = assembled.audio;
		app.state.speaking = false;
		var wall = (performance.now () - t0) / 1000;
		var dur = assembled.total / SR;
		syncButton (); // stays "Stop" while playback continues
		el ('status').textContent = dur.toFixed (1) + 's of audio in ' + wall.toFixed (1) +
			's (' + (dur / wall).toFixed (1) + 'x realtime)';
		el ('actions').classList.remove ('disabled');
	});
	var stoppedByUser = false;
	var stopFallback = null;
	app.listenFor ('PlaybackEnded', function () {
		var wasPlaying = app.state.playing;
		app.state.playing = false;
		clearTimeout (stopFallback);
		syncButton ();
		if (!app.state.speaking && wasPlaying && !stoppedByUser) {
			toast ('Done — Replay, download, or Edit in AudioMass', 2600);
		}
		stoppedByUser = false;
	});
	app.listenFor ('ReplayStarted', function () {
		app.state.playing = true;
		syncButton ();
	});
	// Opening the editor (or starting a recording) suspends our AudioContext,
	// so the worklet's "ended" reply never arrives — clear the state directly.
	app.listenFor ('AudioMassOpen', function () {
		app.state.playing = false;
		syncButton ();
	});
	app.listenFor ('RecordStarted', function () {
		app.state.playing = false;
		syncButton ();
	});
	app.listenFor ('StopRequest', function () {
		app.state.speaking = false; // engine abort follows; playback flush
		stoppedByUser = true;
		if (speechActive) {
			var assembled = assembleParts ();
			if (assembled) {
				app.state.lastAudio = assembled.audio;
				el ('status').textContent = 'Stopped — ' + (assembled.total / SR).toFixed (1) +
					's of audio ready';
				el ('actions').classList.remove ('disabled');
				el ('wave-hint').classList.add ('gone');
			}
		}
		speechActive = false;
		// The worklet acks the flush with "ended" within a tick — but if the
		// AudioContext is suspended (background tab, iOS), that ack never
		// comes. Never leave the button stuck on Stop.
		clearTimeout (stopFallback);
		stopFallback = setTimeout (function () {
			if (app.state.playing) {
				app.state.playing = false;
				syncButton ();
			}
		}, 500);
	});
	app.listenFor ('EngineError', function (e) {
		app.state.speaking = false;
		app.state.playing = false;
		speechActive = false;
		parts = [];
		syncButton ();
		toast ('Engine error during ' + e.during + ' — see console. Reload to retry.', 6000);
	});

	el ('replay').onclick = function () { app.fireEvent ('ReplayRequest'); };
	el ('dl-voice').onclick = function () {
		if (app.fireEvent ('DownloadVoice') === false) toast ('No custom voice file is available to download yet.');
		else toast ('Preparing custom voice files…', 1800);
	};
	el ('dl-wav').onclick = function () { app.fireEvent ('DownloadWav'); };
	el ('dl-mp3').onclick = function () { app.fireEvent ('DownloadMp3'); };
	el ('edit-am').onclick = function () { app.fireEvent ('OpenInAudioMass'); };
	app.listenFor ('VoiceExportError', function () {
		toast ('Could not export this voice as native PocketTTS files.', 4200);
	});
})();

// ── Waveform: progressive peaks + playhead ──────────────────────────────────
(function () {
	var canvas, ctx2d, peaks = [], samplesSeen = 0, bucket = 0, bucketMax = 0;
	var lastPlayed = -1; // remembered playhead so growth redraws don't wipe it
	var SAMPLES_PER_PEAK = 600; // 25ms columns

	function ensure () {
		if (canvas) return;
		canvas = el ('wave');
		ctx2d = canvas.getContext ('2d');
	}
	function resize () {
		ensure ();
		var r = canvas.getBoundingClientRect ();
		canvas.width = r.width * devicePixelRatio;
		canvas.height = r.height * devicePixelRatio;
		draw (-1);
	}
	function draw (playedSamples) {
		ensure ();
		var W = canvas.width, H = canvas.height, n = peaks.length;
		ctx2d.clearRect (0, 0, W, H);
		if (!n) return;
		// stretch across the full canvas width regardless of clip length
		var bw = W / n;
		var gap = bw > 3 ? Math.min (2 * devicePixelRatio, bw * 0.25) : 0;
		var playCol = playedSamples >= 0 ? (playedSamples / SAMPLES_PER_PEAK) : -1;
		for (var i = 0; i < n; ++i) {
			var h = Math.max (2, peaks[i] * H * 0.92);
			var clean = d.documentElement.getAttribute ('data-theme') === 'clean';
			ctx2d.fillStyle = i <= playCol ? (clean ? '#1C2024' : '#E43D4F')
				: (clean ? '#C9CFD4' : '#42305A');
			ctx2d.fillRect (i * bw, (H - h) / 2, Math.max (1, bw - gap), h);
		}
	}

	app.listenFor ('SpeakStarted', function () {
		peaks = []; samplesSeen = 0; bucket = 0; bucketMax = 0;
		lastPlayed = -1;
		draw (-1);
	});
	app.listenFor ('SpeakChunk', function (c) {
		for (var i = 0; i < c.length; ++i) {
			var a = Math.abs (c[i]);
			if (a > bucketMax) bucketMax = a;
			if (++bucket >= SAMPLES_PER_PEAK) {
				peaks.push (bucketMax);
				bucket = 0; bucketMax = 0;
			}
		}
		samplesSeen += c.length;
		draw (lastPlayed); // keep the played region: no flicker while growing
	});
	app.listenFor ('SpeakDone', function () { draw (lastPlayed); });
	app.listenFor ('PlaybackProgress', function (played) {
		lastPlayed = played;
		draw (played);
	});
	app.listenFor ('PlaybackEnded', function () {
		lastPlayed = -1;
		draw (-1);
	});
	app.listenFor ('ReplayStarted', function () {
		lastPlayed = 0;
		draw (0);
	});
	app.listenFor ('ThemeChange', function () { draw (lastPlayed); });

	w.addEventListener ('resize', resize);
	d.addEventListener ('DOMContentLoaded', resize);
	setTimeout (resize, 0);
})();

// ── AudioMass fullscreen modal ──────────────────────────────────────────────
(function () {
	var loaded = false, ready = false, queued = null;

	function frame () { return el ('am-frame'); }
	function shouldUnloadEditor () {
		var keep = !!w.PKTTS_KEEP_AUDIOMASS;
		try { keep = keep || new URLSearchParams (w.location.search).get ('keepAudioMass') === '1'; } catch (e) {}
		return !keep && app.isMobileBrowser && app.isMobileBrowser ();
	}
	function send (buf) {
		frame ().contentWindow.postMessage ({ pkttsWav: buf, name: 'pocket-tts.wav' }, '*');
	}
	function stopEditorPlayback () {
		try {
			var pk = frame ().contentWindow.PKAudioEditor;
			if (pk && pk.fireEvent) {
				pk.fireEvent ('RequestStop');
				pk.fireEvent ('RequestPause');
			}
		} catch (e) { /* editor not booted yet */ }
	}
	function unloadEditorIfMobile () {
		if (!shouldUnloadEditor ()) return;
		loaded = false;
		ready = false;
		queued = null;
		try { frame ().src = 'about:blank'; } catch (e) {}
	}

	w.addEventListener ('message', function (ev) {
		if (ev.data && ev.data.pkttsEditorReady) {
			ready = true;
			if (queued) { send (queued); queued = null; }
		}
		if (ev.data && ev.data.pkttsLoaded) {
			el ('status').textContent = 'Loaded into AudioMass';
			focusEditor ();
		}
	});

	function focusEditor () {
		el ('edit-am').blur ();
		try { frame ().contentWindow.focus (); } catch (e) {}
	}

	function openWith (buf, forClone) {
		el ('am-use').style.display = forClone ? '' : 'none';
		el ('am-copy-mobile').textContent = forClone ?
			'Edit the waveform, then hit "Use this audio".' :
			'Edit the waveform, then export from AudioMass.';
		el ('am-modal').classList.add ('show');
		if (!loaded) {
			loaded = true;
			queued = buf;
			frame ().src = 'audiomass/index.html?skipintro=1';
		}
		else if (ready) send (buf);
		else queued = buf;
		setTimeout (focusEditor, 60);
	}
	app.listenFor ('AudioMassOpen', function (buf) { openWith (buf, false); });
	app.listenFor ('AudioMassOpenForClone', function (buf) { openWith (buf, true); });

	// Same-origin iframe: lift the (possibly edited) buffer straight out of
	// AudioMass and hand it back to the clone flow.
	var amUse = el ('am-use');
	if (amUse) amUse.onclick = function () {
		try {
			var ws = frame ().contentWindow.PKAudioEditor.engine.wavesurfer;
			var b = ws.backend.buffer;
			var mono = new Float32Array (b.getChannelData (0)); // copy out of the iframe
			app.resampleRaw (mono, b.sampleRate).then (function (audio24k) {
				app.fireEvent ('CloneSourceEdited', audio24k);
			});
			el ('am-modal').classList.remove ('show');
			stopEditorPlayback ();
			unloadEditorIfMobile ();
		} catch (e) {
			toast ('Could not read the edited audio back — use File > Export instead.', 5000);
		}
	};

	function closeEditor () {
		el ('am-modal').classList.remove ('show');
		// Pause any playback inside AudioMass (same-origin iframe, so we can
		// speak its event bus directly); the session itself stays alive.
		stopEditorPlayback ();
		unloadEditorIfMobile ();
	}
	el ('am-close').onclick = closeEditor;
	d.addEventListener ('keydown', function (e) {
		if (e.key === 'Escape' && el ('am-modal').classList.contains ('show')) {
			e.preventDefault ();
			closeEditor ();
		}
	});
})();

// ── Clone modal ─────────────────────────────────────────────────────────────
		(function () {
			var pending = null; // validated 24k audio awaiting confirm
			var recTimer = null;
			var cloneBusy = false;
			var activeCloneKey = null;
			var cloneReadyListener = null;
			var activeTab = 'file';

		function isOpen () { return el ('clone-modal').classList.contains ('show'); }
		function open () {
			d.documentElement.classList.add ('clone-modal-open');
			d.body.classList.add ('clone-modal-open');
			el ('clone-modal').classList.add ('show');
			tab ('file');
			reset ();
		}
		function close () {
			if (!isOpen ()) return;
			if (cloneBusy) {
				setReportMessage ('busy', 'Voice cloning is still running');
				return;
			}
			el ('clone-modal').classList.remove ('show');
			d.documentElement.classList.remove ('clone-modal-open');
			d.body.classList.remove ('clone-modal-open');
			app.fireEvent ('RecordStop');
			reset ();
		}
		function setCloneBusy (on) {
			cloneBusy = !!on;
			['clone-close', 'tab-file', 'tab-mic', 'file-input', 'record', 'clip-edit', 'clone-name']
				.forEach (function (id) {
					var node = el (id);
					if (node) node.disabled = cloneBusy;
				});
			el ('clone-x').setAttribute ('aria-disabled', cloneBusy ? 'true' : 'false');
			el ('clone-confirm').disabled = cloneBusy || !pending;
		}
			function reset () {
				pending = null;
				clipAudio = null;
				clipSource = '';
				el ('clip-preview').style.display = 'none';
				clearReport ();
				el ('clone-confirm').disabled = true;
				el ('rec-time').textContent = '0.0s';
				el ('rec-bar').style.width = '0%';
				el ('record').textContent = 'Start recording';
				el ('record').classList.remove ('recording');
				el ('record').disabled = false;
				el ('record').title = '';
				el ('tab-file').disabled = false;
			}
			function tab (which) {
				activeTab = which;
				el ('tab-file').classList.toggle ('active', which === 'file');
				el ('tab-mic').classList.toggle ('active', which === 'mic');
				el ('pane-file').style.display = which === 'file' ? '' : 'none';
				el ('pane-mic').style.display = which === 'mic' ? '' : 'none';
				if (clipAudio && clipSource === which) {
					el ('clip-preview').style.display = '';
					applySelection ();
				}
				else {
					el ('clip-preview').style.display = 'none';
					pending = null;
					el ('clone-confirm').disabled = true;
				}
			}

		function clearReport () {
			el ('clone-report').textContent = '';
		}

		function addReportMessage (kind, text) {
			var msg = d.createElement ('div');
			msg.className = 'msg' + (kind ? ' ' + kind : '');
			msg.textContent = text;
			el ('clone-report').appendChild (msg);
		}

		function setReportMessage (kind, text) {
			clearReport ();
			addReportMessage (kind, text);
		}

		app.listenFor ('CloneModalOpen', open);
	el ('clone-close').onclick = close;
	el ('clone-x').onclick = close;
	el ('clone-x').addEventListener ('keydown', function (e) {
		if (e.key === 'Enter' || e.key === ' ') { e.preventDefault (); close (); }
	});
	d.addEventListener ('keydown', function (e) {
		if (e.key !== 'Escape' || e.defaultPrevented || !isOpen ()) return;
		e.preventDefault ();
		close ();
		});
			el ('tab-file').onclick = function () { if (!cloneBusy) tab ('file'); };
			el ('tab-mic').onclick = function () { if (!cloneBusy) tab ('mic'); };
			el ('clone-name').addEventListener ('input', function () {
				if (this.value.length > 24) this.value = this.value.slice (0, 24);
			});

			function report (r, sourceLabel) {
			clearReport ();
			r.errors.forEach (function (m) { addReportMessage ('err', m); });
			r.warnings.forEach (function (m) { addReportMessage ('warn', m); });
			if (r.ok) {
				addReportMessage ('ok', sourceLabel + ' — ' + r.stats.duration.toFixed (1) +
					's of audio, ready to clone.');
			}
			el ('clone-confirm').disabled = cloneBusy || !r.ok;
		}

		function accept (audio24k, label) {
			if (cloneBusy) return;
			var r = app.validateVoice (audio24k);
			pending = r.ok ? audio24k : null;
			report (r, label);
	}

	// ── clip preview: waveform + draggable trim region ──────────────────
	var clipAudio = null, clipLabel = '', clipSource = '', selA = 0, selB = 1;
	var SR24 = 24000, MIN_SEL_S = 2;

	function fmtT (sec) {
		var m = Math.floor (sec / 60), r = sec - m * 60;
		return m + ':' + (r < 10 ? '0' : '') + r.toFixed (1);
	}
	function percentile (xs, p) {
		if (!xs.length) return 0;
		var sorted = xs.slice ().sort (function (a, b) { return a - b; });
		return sorted[Math.max (0, Math.min (sorted.length - 1, Math.floor ((sorted.length - 1) * p)))];
	}
	function sustainedBoundary (active, fromStart) {
		var need = 3, span = 4, i, j, hits;
		if (fromStart) {
			for (i = 0; i < active.length; ++i) {
				hits = 0;
				for (j = i; j < Math.min (active.length, i + span); ++j) if (active[j]) hits++;
				if (hits >= need) return i;
			}
		}
		else {
			for (i = active.length - 1; i >= 0; --i) {
				hits = 0;
				for (j = i; j >= Math.max (0, i - span + 1); --j) if (active[j]) hits++;
				if (hits >= need) return i;
			}
		}
		return -1;
	}
	function suggestSpeechSelection (audio) {
		var n = audio ? audio.length : 0, dur = n / SR24;
		if (dur < 2.5) return { a: 0, b: 1 };

		var win = Math.floor (SR24 * 0.05), frames = Math.floor (n / win);
		if (frames < 4) return { a: 0, b: 1 };

		var series = [], totalSq = 0, i, j, s, x;
		for (i = 0; i < frames; ++i) {
			s = 0;
			for (j = i * win; j < (i + 1) * win; ++j) {
				x = audio[j];
				s += x * x;
			}
			totalSq += s;
			series.push (Math.sqrt (s / win));
		}

		var globalRms = Math.sqrt (totalSq / Math.max (1, frames * win));
		if (globalRms < 0.006) return { a: 0, b: 1 };

		var noise = percentile (series, 0.2);
		var threshold = Math.max (0.0025, globalRms * 0.16);
		if (noise < globalRms * 0.55) threshold = Math.max (threshold, noise * 3.2);

		var active = series.map (function (v) { return v >= threshold; });
		var start = sustainedBoundary (active, true);
		var end = sustainedBoundary (active, false);
		if (start < 0 || end < start || (end - start + 1) * 0.05 < 1.0) return { a: 0, b: 1 };

		var pad = 4; // keep about 200ms around the detected voice.
		start = Math.max (0, start - pad);
		end = Math.min (frames - 1, end + pad);
		var a = (start * win) / n;
		var b = Math.min (1, ((end + 1) * win) / n);

		var minDur = Math.min (6, dur);
		if ((b - a) * dur < minDur) {
			var half = (minDur / dur) / 2;
			var mid = (a + b) / 2;
			a = mid - half;
			b = mid + half;
			if (a < 0) { b -= a; a = 0; }
			if (b > 1) { a -= b - 1; b = 1; }
			a = Math.max (0, a);
		}

		// Ignore tiny trims; they only make the UI look twitchy.
		if (a < 0.02 && b > 0.98) return { a: 0, b: 1 };
		if ((a * dur) + ((1 - b) * dur) < 0.3) return { a: 0, b: 1 };
		return { a: a, b: b };
	}
	function drawClip () {
		var c = el ('clip-wave'), ctx = c.getContext ('2d');
		var W = c.width = c.clientWidth * devicePixelRatio;
		var H = c.height = c.clientHeight * devicePixelRatio;
		ctx.clearRect (0, 0, W, H);
		if (!clipAudio) return;
		var clean = d.documentElement.getAttribute ('data-theme') === 'clean';
		ctx.fillStyle = clean ? '#9AA4AE' : '#E43D4F';
		var cols = 220, per = Math.max (1, Math.floor (clipAudio.length / cols));
		for (var i = 0; i < cols; ++i) {
			var maxA = 0;
			for (var j = i * per, e = Math.min (clipAudio.length, j + per); j < e; ++j) {
				var a = Math.abs (clipAudio[j]);
				if (a > maxA) maxA = a;
			}
			var h = Math.max (2, maxA * H * 0.92);
			ctx.fillRect (i * (W / cols), (H - h) / 2, W / cols - 1, h);
		}
	}
	function layoutSel () {
		var wrap = el ('clip-wave');
		var w = wrap.clientWidth;
		el ('clip-shade-l').style.left = '2px';
		el ('clip-shade-l').style.width = Math.max (0, selA * w - 2) + 'px';
		el ('clip-shade-r').style.left = (selB * w) + 'px';
		el ('clip-shade-r').style.width = Math.max (0, w - selB * w - 2) + 'px';
		el ('clip-handle-l').style.left = Math.max (0, selA * w - 6) + 'px';
		el ('clip-handle-r').style.left = Math.min (w - 12, selB * w - 6) + 'px';
		var dur = clipAudio ? clipAudio.length / SR24 : 0;
		el ('clip-sel').textContent = '✂ ' + fmtT (selA * dur) + ' – ' + fmtT (selB * dur) +
			' (' + ((selB - selA) * dur).toFixed (1) + 's selected)';
	}
	function selectedSlice () {
		var s0 = Math.floor (selA * clipAudio.length);
		var s1 = Math.ceil (selB * clipAudio.length);
		return clipAudio.slice (s0, s1);
	}
	function applySelection () {
		if (!clipAudio) return;
		var slice = selectedSlice ();
		var r = app.validateVoice (slice);
		pending = r.ok ? slice : null;
		report (r, clipLabel + (selB - selA < 0.999 ? ' (trimmed)' : ''));
	}
		function showClip (audio, label, source) {
			clipAudio = audio;
			clipLabel = label;
			clipSource = source || clipSource || activeTab;
			var suggested = suggestSpeechSelection (audio);
			selA = suggested.a; selB = suggested.b;
			el ('clip-preview').style.display = clipSource === activeTab ? '' : 'none';
			drawClip ();
			layoutSel ();
			if (clipSource === activeTab) applySelection ();
			else {
				pending = null;
				el ('clone-confirm').disabled = true;
			}
		}
	// handle dragging (pointer events cover touch)
	['l', 'r'].forEach (function (side) {
			var h = el ('clip-handle-' + side);
			if (!h) return;
			h.addEventListener ('pointerdown', function (ev) {
				if (cloneBusy) return;
				ev.preventDefault ();
				h.setPointerCapture (ev.pointerId);
			var move = function (mv) {
				var rect = el ('clip-wave').getBoundingClientRect ();
				var f = Math.min (1, Math.max (0, (mv.clientX - rect.left) / rect.width));
				var minGap = clipAudio ? Math.min (0.999, MIN_SEL_S / (clipAudio.length / SR24)) : 0.05;
				if (side === 'l') selA = Math.min (f, selB - minGap);
				else selB = Math.max (f, selA + minGap);
				layoutSel ();
			};
			var up = function () {
				h.removeEventListener ('pointermove', move);
				h.removeEventListener ('pointerup', up);
				applySelection (); // validate once per drag, not per pixel
			};
			h.addEventListener ('pointermove', move);
			h.addEventListener ('pointerup', up);
		});
	});
	// hand the (full) clip to AudioMass for cleanup; edited audio comes back
		// via CloneSourceEdited
		if (el ('clip-edit')) el ('clip-edit').onclick = function () {
			if (cloneBusy) return;
			if (!clipAudio) return;
			app.wavBlob48k (clipAudio).then (function (b) {
			return b.arrayBuffer ();
		}).then (function (buf) {
			app.fireEvent ('AudioMassOpenForClone', buf);
		});
		};
		app.listenFor ('CloneSourceEdited', function (audio24k) {
			if (cloneBusy) return;
			showClip (audio24k, 'Edited audio');
			toast ('Edited audio loaded — adjust the trim if needed, then clone.', 4000);
	});

			// file path
			function handleFile (file) {
				if (cloneBusy) return;
				setReportMessage ('', 'Decoding ' + file.name + '…');
			app.decodeFile (file).then (function (buf) {
				return app.resampleTo24k (buf);
			}).then (function (audio) {
				showClip (audio, file.name, 'file');
			}).catch (function (e) {
				setReportMessage ('err', 'Could not decode this file (' + e + ')');
			});
		}
		el ('file-input').onchange = function () {
			if (cloneBusy) return;
			if (this.files && this.files[0]) handleFile (this.files[0]);
		};
		var drop = el ('pane-file');
		drop.ondragover = function (e) {
			e.preventDefault ();
			if (!cloneBusy) drop.classList.add ('over');
		};
		drop.ondragleave = function () { drop.classList.remove ('over'); };
		drop.ondrop = function (e) {
			e.preventDefault ();
			drop.classList.remove ('over');
			if (cloneBusy) return;
			if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFile (e.dataTransfer.files[0]);
		};

	// mic path: live waveform + duration guidance (aim 5-15s)
	var live, liveCtx;
	function drawLive (chunk) {
		if (!live) { live = el ('rec-wave'); liveCtx = live.getContext ('2d'); }
		var W = live.width = live.clientWidth * devicePixelRatio;
		var H = live.height = live.clientHeight * devicePixelRatio;
		// scroll: draw the last ~2s from ring
		liveRing.push (chunk);
		var total = 0, i;
		for (i = 0; i < liveRing.length; ++i) total += liveRing[i].length;
		while (liveRing.length > 1 && total - liveRing[0].length > 2 * 48000) {
			total -= liveRing[0].length; liveRing.shift ();
		}
		liveCtx.clearRect (0, 0, W, H);
		liveCtx.fillStyle = d.documentElement.getAttribute ('data-theme') === 'clean'
			? '#B3392F' : '#E43D4F';
		var cols = 160, per = Math.max (1, Math.floor (total / cols)), c = 0, maxA = 0, x = 0, k = 0;
		for (i = 0; i < liveRing.length; ++i) {
			var ch = liveRing[i];
			for (var j = 0; j < ch.length; ++j) {
				var a = Math.abs (ch[j]);
				if (a > maxA) maxA = a;
				if (++c >= per) {
					var h = Math.max (2, maxA * H * 0.9);
					liveCtx.fillRect ((k++) * (W / cols), (H - h) / 2, W / cols - 1, h);
					c = 0; maxA = 0;
				}
			}
		}
	}
	var liveRing = [];

		el ('record').onclick = function () {
			if (cloneBusy) return;
			if (this.classList.contains ('recording')) app.fireEvent ('RecordStop');
		else {
			liveRing = [];
			app.fireEvent ('RecordStart');
		}
	};
		app.listenFor ('RecordStarted', function () {
			el ('record').textContent = 'Stop';
			el ('record').classList.add ('recording');
			el ('record').title = '';
			el ('record').disabled = true; // no accidental instant stop
		setTimeout (function () { el ('record').disabled = false; }, 800);
			el ('clone-confirm').disabled = true;
			el ('tab-file').disabled = true;
			setReportMessage ('', 'Read a couple of sentences naturally — aim for 6–10 seconds.');
		});
	app.listenFor ('RecordProgress', function (p) {
		el ('rec-time').textContent = p.seconds.toFixed (1) + 's';
		var pct = Math.min (100, p.seconds / 15 * 100);
		el ('rec-bar').style.width = pct + '%';
		el ('rec-bar').className = p.seconds < 6 ? 'short' : (p.seconds <= 10 ? 'good' : 'long');
		drawLive (p.latest);
		if (p.seconds >= 25) app.fireEvent ('RecordStop');
	});
		app.listenFor ('RecordDone', function (r) {
			el ('record').textContent = 'Start recording';
				el ('record').classList.remove ('recording');
				el ('tab-file').disabled = false;
				if (!isOpen ()) return;
				app.resampleRaw (r.audio, r.sampleRate).then (function (audio) {
					if (!isOpen () || cloneBusy) return;
					if (audio.length < 6 * SR24) {
						clipAudio = null;
						clipSource = '';
						pending = null;
						el ('clip-preview').style.display = 'none';
						el ('clone-confirm').disabled = true;
						el ('record').title = 'Need at least 6 seconds of audio to clone a voice.';
						setReportMessage ('err', 'Need more audio — record at least 6 seconds.');
						return;
					}
					el ('record').title = '';
					var leveled = app.levelMobileMicForClone ? app.levelMobileMicForClone (audio) : { audio: audio, gain: 1 };
					var label = leveled.gain > 1 ? 'Microphone recording — mobile level normalized' : 'Microphone recording';
					showClip (leveled.audio, label, 'mic');
				});
			});
		app.listenFor ('RecordError', function (msg) {
			if (!isOpen ()) return;
			setReportMessage ('err', 'Microphone unavailable: ' + msg);
		});

		el ('clone-confirm').onclick = function () {
			if (cloneBusy || !pending) return;
				var label = (el ('clone-name').value.trim () || ('My voice ' + (app.state.voices.length))).slice (0, 24);
				var key = 'clone-' + Date.now ().toString (36);
				activeCloneKey = key;
				setReportMessage ('busy', 'Cloning - first time loads the voice encoder (~40MB)');
				setCloneBusy (true);
			if (cloneReadyListener) app.stopListening ('VoiceReady', cloneReadyListener);
			var doneOnce = cloneReadyListener = app.listenFor ('VoiceReady', function (v) {
					if (v.key !== key) return;
					app.stopListening ('VoiceReady', doneOnce);
					cloneReadyListener = null;
					activeCloneKey = null;
					setCloneBusy (false);
					app.state.activeVoice = key;
				app.fireEvent ('VoiceSelected', key);
				close ();
			});
			app.fireEvent ('CloneRequest', { key: key, label: label, audio: pending });
		};

			app.listenFor ('EngineError', function (err) {
				if (!cloneBusy || !isOpen () || !err ||
					(err.during !== 'clone' && err.during !== 'engine') ||
					(err.key && err.key !== activeCloneKey)) return;
				if (cloneReadyListener) {
					app.stopListening ('VoiceReady', cloneReadyListener);
					cloneReadyListener = null;
				}
				activeCloneKey = null;
				setCloneBusy (false);
				setReportMessage ('err', 'Could not clone this voice (' + (err.message || err) + ')');
			});

			app.listenFor ('EngineProgress', function (p) {
				if (el ('clone-modal').classList.contains ('show')) {
					setReportMessage (cloneBusy ? 'busy' : '', p.label);
				}
			});
})();

})(window, document, window.PKTTS);
