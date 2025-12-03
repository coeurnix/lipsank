# Lipsank

Lipsank is a tiny, single-header C/C++ library that extracts simple real-time mouth parameters from raw mono PCM16 audio. It is fully heuristic (no ML model) and portable enough to run on desktops, mobile, or WebAssembly builds.

# Lipsank Quickstart

Lipsank extracts mouth shape parameters from audio for lip-sync animation. Feed it audio samples, get back `open`, `wide`, `round`, `tension`, and `intensity` values (all 0-1).

## C/C++

Single-header library. Drop `lipsank.h` into your project.

```c
#define LIPSANK_IMPLEMENTATION
#include "lipsank.h"

void on_mouth_params(void *user, float time, const LipSyncMouthParams *p) {
    // p->open, p->wide, p->round, p->tension, p->intensity
    update_your_avatar(p->open, p->wide, p->round);
}

int main() {
    LipSyncContext ctx;
    lipsync_init(&ctx, NULL, on_mouth_params, NULL);  // NULL config = sensible defaults
    
    // Feed audio as you get it (16-bit PCM, mono)
    while (has_audio()) {
        int16_t samples[256];
        read_audio(samples, 256);
        lipsync_feed_pcm16(&ctx, samples, 256);  // Callback fires automatically
    }
    
    lipsync_dispose(&ctx);
}
```

That's it. The callback fires roughly every 10ms with new mouth parameters.

**Floats instead of int16?** Use `lipsync_feed_float()` with samples in -1 to 1 range.

**Custom settings?** Start with `lipsync_default_config(sample_rate)` and tweak from there.

---

## Web (WASM)

Be sure `libsync.wasm` and `lipsync.js` are in your folder and load with `<script src="lipsync.js"></script>`, then:

```javascript
// After Module loads...

// 1. Init with your sample rate
Module.ccall('lipsync_wasm_init', 'number', ['number'], [44100]);

// 2. Decode audio however you want (Web Audio API works great)
const audioCtx = new AudioContext();
const buffer = await audioCtx.decodeAudioData(arrayBuffer);
const samples = buffer.getChannelData(0);  // mono float32 array

// 3. Feed samples in chunks
const CHUNK = 4410;  // 100ms at 44.1kHz
const wasmBuf = Module.ccall('lipsync_wasm_alloc_float_buffer', 'number', ['number'], [CHUNK]);

for (let i = 0; i < samples.length; i += CHUNK) {
    const chunk = samples.slice(i, i + CHUNK);
    Module.HEAPF32.set(chunk, wasmBuf / 4);
    Module.ccall('lipsync_wasm_feed_float', 'number', ['number', 'number'], [wasmBuf, chunk.length]);
}

Module.ccall('lipsync_wasm_free_buffer', null, ['number'], [wasmBuf]);

// 4. Read results (time, open, wide, round, tension, intensity per frame)
const outBuf = Module.ccall('lipsync_wasm_alloc_float_buffer', 'number', ['number'], [6000]);
const count = Module.ccall('lipsync_wasm_read_batch', 'number', ['number', 'number'], [outBuf, 1000]);

const results = [];
for (let i = 0; i < count; i++) {
    const off = (outBuf / 4) + i * 6;
    results.push({
        time: Module.HEAPF32[off],
        open: Module.HEAPF32[off + 1],
        wide: Module.HEAPF32[off + 2],
        round: Module.HEAPF32[off + 3],
        tension: Module.HEAPF32[off + 4],
        intensity: Module.HEAPF32[off + 5]
    });
}

Module.ccall('lipsync_wasm_free_buffer', null, ['number'], [outBuf]);

// 5. Animate your face using results[] while the audio plays (e.g., update every 100ms)
```

---

## What the parameters mean

| Param | What it does |
|-------|--------------|
| `open` | Jaw drop. High for "ah", low for "ee" |
| `wide` | Smile/stretch. High for "ee", low for "oo" |
| `round` | Lip pucker. High for "oo", low for "ee" |
| `tension` | Lip tightness. High for "f", "m", "b" |
| `intensity` | Loudness. Use for emphasis or blink triggers |

Map these to your avatar's blendshapes. Most rigs have something like `jawOpen`, `mouthSmile`, `mouthPucker` - you'll figure out the mapping pretty quick.

---

## Tips

- Loaded audio but no callbacks? Check your sample rate matches the audio.
- Animation looks jittery? You probably don't need to do anything - v2.0 has built-in smoothing. If you're adding your own smoothing on top, try removing it.
- Mouth moves during silence? Lower `gate_threshold_db` in the config (default is -40).
- Want faster response? Decrease `attack_ms`. Want smoother motion? Increase `release_ms`.
