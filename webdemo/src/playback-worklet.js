// Streaming playback worklet: a ring buffer fed by postMessage chunks.
// port.postMessage(Float32Array) appends; {end:true} marks stream end;
// {flush:true} drops everything. Posts {played: n} progress and {ended: true}.
class PKTTSPlayer extends AudioWorkletProcessor {
	constructor () {
		super ();
		this.chunks = [];
		this.offset = 0;
		this.ended = false;
		this.done = false;
		this.played = 0;
		this.port.onmessage = (e) => {
			var d = e.data;
			if (!d) return;
			// flush and end may arrive in ONE message ({flush:true, end:true}
			// is how Stop drains): handle both, in that order.
			if (d.flush) { this.chunks = []; this.offset = 0; this.ended = false; this.done = false; this.played = 0; }
			if (d.end) this.ended = true;
			if (d.length) this.chunks.push (d);
		};
	}

	process (ins, outs) {
		var out = outs[0][0];
		var i = 0;
		var active = this.chunks.length > 0;
		while (i < out.length && this.chunks.length) {
			var head = this.chunks[0];
			var take = Math.min (out.length - i, head.length - this.offset);
			out.set (head.subarray (this.offset, this.offset + take), i);
			i += take;
			this.offset += take;
			if (this.offset >= head.length) { this.chunks.shift (); this.offset = 0; }
		}
		for (; i < out.length; ++i) out[i] = 0;
		if (this.ended && !this.chunks.length && !this.done) {
			this.done = true;
			this.port.postMessage ({ ended: true });
		}
		// Only advance/report progress while actually playing samples;
		// an idle worklet must be silent (no postMessage churn under modals).
		if (active) {
			this.played += out.length;
			if ((this.played & 8191) < 128) this.port.postMessage ({ played: this.played });
		}
		return true;
	}
}
registerProcessor ('pktts-player', PKTTSPlayer);
