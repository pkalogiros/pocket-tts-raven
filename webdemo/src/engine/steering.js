export const EMOTIONS = ['neutral', 'angry', 'disgust', 'fear', 'happy', 'sad'];

export function controls(enabled, { emotion = 'neutral', intensity = 1, seed } = {}) {
  if (!EMOTIONS.includes(emotion) || !Number.isFinite(intensity) || intensity < 0 || intensity > 1.2)
    throw new Error('Expected a supported emotion and intensity from 0 to 1.2');
  if (!enabled && emotion !== 'neutral' && intensity > 0)
    throw new Error('Enable emotion steering before selecting an emotion');
  if (seed !== undefined && (!Number.isSafeInteger(seed) || seed < 0))
    throw new Error('Seed must be an integer from 0 to 9007199254740991');
  return { emotion, intensity, seed };
}

export function parseVectors(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 12 || bytes[0] !== 147 || new TextDecoder().decode(bytes.subarray(1, 6)) !== 'NUMPY')
    throw new Error('Expected a NumPy .npy vector file');
  const view = new DataView(buffer);
  const version = bytes[6], start = version === 1 ? 10 : 12;
  if (version !== 1 && version !== 2) throw new Error('Unsupported NPY version');
  const length = version === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  if (start + length > bytes.length) throw new Error('Truncated NPY header');
  const header = new TextDecoder().decode(bytes.subarray(start, start + length));
  if (!/['"]descr['"]\s*:\s*['"]<f4['"]/.test(header) ||
      !/['"]fortran_order['"]\s*:\s*False/.test(header) ||
      !/['"]shape['"]\s*:\s*\(\s*6\s*,\s*1024\s*,?\s*\)/.test(header) ||
      bytes.length - start - length !== 6 * 1024 * 4)
    throw new Error('Expected C-order float32 vectors with shape [6, 1024]');
  const rows = new Float32Array(buffer.slice(start + length));
  if (!rows.every(Number.isFinite)) throw new Error('Vectors must contain only finite values');
  rows.fill(0, 0, 1024);
  return rows;
}

export function applyShift(destination, rows, emotion, intensity) {
  const index = EMOTIONS.indexOf(emotion);
  if (index === 0 || intensity === 0) { destination.fill(0); return; }
  for (let i = 0; i < 1024; ++i) destination[i] = rows[index * 1024 + i] * intensity;
}
