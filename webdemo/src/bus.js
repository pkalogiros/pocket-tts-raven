// PKTTS — the app core: an event bus in the AudioMass style.
// Everything talks through fireEvent / listenFor; no module imports UI code.
(function (w) {
	'use strict';

	var listeners = {};

	w.PKTTS = {
		fireEvent: function (name, arg1, arg2) {
			var list = listeners[name];
			if (!list) return;
			var ret;
			for (var i = 0; i < list.length; ++i) {
				var r = list[i] (arg1, arg2);
				if (r !== undefined) ret = r;
			}
			return ret;
		},

		listenFor: function (name, fn) {
			(listeners[name] = listeners[name] || []).push (fn);
			return fn;
		},

		stopListening: function (name, fn) {
			var list = listeners[name];
			if (!list) return;
			var i = list.indexOf (fn);
			if (i >= 0) list.splice (i, 1);
		},

		ui: {},      // populated by app.js modules
		state: {
			ready: false,
			speaking: false,
			voices: [],       // {key, label, builtin}
			voicePayloads: {}, // cloned voice payloads keyed by voice key
			activeVoice: null,
			lastAudio: null   // Float32Array 24k mono of the last generation
		}
	};
})(window);
