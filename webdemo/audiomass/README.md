# AudioMass integration point

PocketTTS-RAVEN does not vendor AudioMass. This directory is a drop-in mount
point for a local AudioMass checkout or build:

- https://audiomass.co
- https://github.com/pkalogiros/audiomass

The web demo opens `audiomass/index.html?skipintro=1` in a same-origin iframe.
To enable the "Edit in AudioMass" buttons, place an AudioMass build here so
this file exists:

```text
webdemo/audiomass/index.html
```

The expected bridge is:

1. AudioMass posts `{ pkttsEditorReady: true }` to the parent window after
   `PKAudioEditor` and `PKAudioEditor.engine.LoadFile` are ready.
2. PocketTTS-RAVEN posts `{ pkttsWav: ArrayBuffer, name: "pocket-tts.wav" }`
   to the iframe.
3. AudioMass validates that `PKAudioEditor.engine.LoadFile` exists, then loads
   the received WAV.
4. AudioMass posts `{ pkttsLoaded: true }` to the parent after the file is
   loaded.

Example bridge code inside AudioMass:

```js
window.parent.postMessage({ pkttsEditorReady: true }, window.location.origin);

window.addEventListener("message", function (ev) {
  var data = ev.data || {};
  if (!data.pkttsWav) return;

  var editor = window.PKAudioEditor;
  var loadFile = editor && editor.engine && editor.engine.LoadFile;
  if (typeof loadFile !== "function") {
    console.warn("[pktts] AudioMass bridge: PKAudioEditor.engine.LoadFile is not ready");
    return;
  }

  var name = data.name || "pocket-tts.wav";
  var file = new File([data.pkttsWav], name, { type: "audio/wav" });
  loadFile.call(editor.engine, file);
  window.parent.postMessage({ pkttsLoaded: true }, window.location.origin);
});
```

The iframe must be served from the same origin as `webdemo/index.html`; the
demo reads edited audio back through `PKAudioEditor.engine.wavesurfer` when the
clone flow uses the "Use this audio" button.
