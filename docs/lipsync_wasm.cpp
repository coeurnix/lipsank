// lipsync_wasm.c - WebAssembly wrapper for Lipsank lip-sync library
// Compile with Emscripten: emcc lipsync_wasm.c -o lipsync.js ...

#include <emscripten/emscripten.h>
#include <stdlib.h>
#include <string.h>

#define LIPSANK_IMPLEMENTATION
#include "lipsank.h"

// Global context (single instance for simplicity)
static LipSyncContext g_ctx;
static int g_initialized = 0;

// Circular buffer to store results for JS to poll
#define MAX_RESULTS 4096
typedef struct {
  float time;
  float open;
  float wide;
  float round;
  float tension;
  float intensity;
} LipSyncResult;

static LipSyncResult g_results[MAX_RESULTS];
static int g_result_write = 0;
static int g_result_read = 0;
static int g_result_count = 0;

// Callback that stores results in the ring buffer
static void wasm_callback(void *user_data, float time_seconds,
                          const LipSyncMouthParams *params) {
  (void)user_data;

  if (g_result_count < MAX_RESULTS) {
    LipSyncResult *r = &g_results[g_result_write];
    r->time = time_seconds;
    r->open = params->open;
    r->wide = params->wide;
    r->round = params->round;
    r->tension = params->tension;
    r->intensity = params->intensity;

    g_result_write = (g_result_write + 1) % MAX_RESULTS;
    g_result_count++;
  }
}

// Exported C functions (prevent C++ name mangling)
#ifdef __cplusplus
extern "C" {
#endif

// Initialize the lip-sync engine
EMSCRIPTEN_KEEPALIVE
int lipsync_wasm_init(int sample_rate) {
  if (g_initialized) {
    lipsync_dispose(&g_ctx);
  }

  LipSyncConfig cfg = lipsync_default_config(sample_rate);
  lipsync_init(&g_ctx, &cfg, wasm_callback, NULL);

  g_result_write = 0;
  g_result_read = 0;
  g_result_count = 0;
  g_initialized = 1;

  return lipsync_ready(&g_ctx) ? 1 : 0;
}

// Reset the engine (for processing a new file)
EMSCRIPTEN_KEEPALIVE
void lipsync_wasm_reset(void) {
  if (g_initialized) {
    lipsync_reset(&g_ctx);
  }
  g_result_write = 0;
  g_result_read = 0;
  g_result_count = 0;
}

// Feed float audio samples (range -1 to 1)
// Returns number of new results available
EMSCRIPTEN_KEEPALIVE
int lipsync_wasm_feed_float(float *samples, int count) {
  if (!g_initialized)
    return 0;

  int before = g_result_count;
  lipsync_feed_float(&g_ctx, samples, (size_t)count);
  return g_result_count - before;
}

// Feed int16 audio samples
EMSCRIPTEN_KEEPALIVE
int lipsync_wasm_feed_int16(int16_t *samples, int count) {
  if (!g_initialized)
    return 0;

  int before = g_result_count;
  lipsync_feed_pcm16(&g_ctx, samples, (size_t)count);
  return g_result_count - before;
}

// Get number of available results
EMSCRIPTEN_KEEPALIVE
int lipsync_wasm_results_available(void) { return g_result_count; }

// Read a result (returns pointer to static buffer with 6 floats)
// Returns NULL if no results available
static float g_read_buffer[6];

EMSCRIPTEN_KEEPALIVE
float *lipsync_wasm_read_result(void) {
  if (g_result_count == 0)
    return NULL;

  LipSyncResult *r = &g_results[g_result_read];
  g_read_buffer[0] = r->time;
  g_read_buffer[1] = r->open;
  g_read_buffer[2] = r->wide;
  g_read_buffer[3] = r->round;
  g_read_buffer[4] = r->tension;
  g_read_buffer[5] = r->intensity;

  g_result_read = (g_result_read + 1) % MAX_RESULTS;
  g_result_count--;

  return g_read_buffer;
}

// Batch read: copy up to max_count results into provided buffer
// Buffer should have space for max_count * 6 floats
// Returns number of results copied
EMSCRIPTEN_KEEPALIVE
int lipsync_wasm_read_batch(float *out_buffer, int max_count) {
  int copied = 0;
  while (g_result_count > 0 && copied < max_count) {
    LipSyncResult *r = &g_results[g_result_read];
    out_buffer[copied * 6 + 0] = r->time;
    out_buffer[copied * 6 + 1] = r->open;
    out_buffer[copied * 6 + 2] = r->wide;
    out_buffer[copied * 6 + 3] = r->round;
    out_buffer[copied * 6 + 4] = r->tension;
    out_buffer[copied * 6 + 5] = r->intensity;

    g_result_read = (g_result_read + 1) % MAX_RESULTS;
    g_result_count--;
    copied++;
  }
  return copied;
}

// Cleanup
EMSCRIPTEN_KEEPALIVE
void lipsync_wasm_dispose(void) {
  if (g_initialized) {
    lipsync_dispose(&g_ctx);
    g_initialized = 0;
  }
}

// Get configuration info
EMSCRIPTEN_KEEPALIVE
float lipsync_wasm_get_frame_time(void) {
  if (!g_initialized)
    return 0.01f;
  return (float)g_ctx.cfg.frame_hop_samples / (float)g_ctx.cfg.sample_rate_hz;
}

// Allocate memory for audio buffer (for JS to write into)
EMSCRIPTEN_KEEPALIVE
float *lipsync_wasm_alloc_float_buffer(int count) {
  return (float *)malloc(count * sizeof(float));
}

EMSCRIPTEN_KEEPALIVE
void lipsync_wasm_free_buffer(void *ptr) { free(ptr); }

#ifdef __cplusplus
}
#endif