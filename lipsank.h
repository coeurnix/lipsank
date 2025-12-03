// Lipsank - lightweight heuristic lip-sync parameter extractor (v2.0)
// Single-header library. Define LIPSANK_IMPLEMENTATION in one translation unit
// to compile the implementation.

#ifndef LIPSANK_H
#define LIPSANK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

// -- Public API
// ----------------------------------------------------------------

typedef struct LipSyncMouthParams {
  float open;      // jaw / mouth opening 0..1
  float wide;      // horizontal stretch / smile 0..1
  float round;     // lip rounding / pursing 0..1
  float tension;   // lip tension 0..1 - for labial consonants
  float intensity; // emphasis / loudness 0..1
} LipSyncMouthParams;

typedef struct LipSyncConfig {
  int sample_rate_hz;     // e.g., 16000 or 44100
  int frame_hop_samples;  // hop size between analysis frames
  int frame_size_samples; // window size for analysis
  int lookahead_frames;   // number of future frames to peek (0..60)

  // Smoothing parameters
  float attack_ms;  // time to reach target when increasing (default 35)
  float release_ms; // time to reach target when decreasing (default 80)
  float intensity_smooth_ms; // smoothing for intensity
  float spring_damping;      // 1.0 = critically damped (no overshoot), <1 =
                             // underdamped

  // Noise gate
  float gate_threshold_db;  // level below which mouth closes (default -40)
  float gate_hysteresis_db; // hysteresis band (default 6)

  float initial_noise_db; // assumed floor, e.g., -55 dB
} LipSyncConfig;

typedef struct LipSyncContext LipSyncContext;

typedef void (*LipSyncCallback)(void *user_data, float time_seconds,
                                const LipSyncMouthParams *params);

// Default configuration (roughly 25 ms window, 10 ms hop @16 kHz)
LipSyncConfig lipsync_default_config(int sample_rate_hz);

// User supplies the context storage.
void lipsync_init(LipSyncContext *ctx, const LipSyncConfig *config,
                  LipSyncCallback cb, void *user_data);

void lipsync_reset(LipSyncContext *ctx);

// Feed 16-bit mono PCM samples. Accepts arbitrary chunk sizes.
void lipsync_feed_pcm16(LipSyncContext *ctx, const int16_t *samples,
                        size_t count);

// Feed 32-bit float mono samples (range -1..1)
void lipsync_feed_float(LipSyncContext *ctx, const float *samples,
                        size_t count);

// Optional helper to check if the context has been initialized successfully.
int lipsync_ready(const LipSyncContext *ctx);

// Release internal buffers. The user still owns the context storage itself.
void lipsync_dispose(LipSyncContext *ctx);

#ifdef __cplusplus
} // extern "C"
#endif

// ---------------------------- Implementation
// ----------------------------------

#ifdef LIPSANK_IMPLEMENTATION

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef LIPSANK_MALLOC
#define LIPSANK_MALLOC(sz) malloc(sz)
#endif

#ifndef LIPSANK_FREE
#define LIPSANK_FREE(p) free(p)
#endif

#ifndef LIPSANK_STATIC
#define LIPSANK_STATIC static
#endif

#ifndef LIPSANK_INLINE
#if defined(_MSC_VER)
#define LIPSANK_INLINE __forceinline
#else
#define LIPSANK_INLINE inline
#endif
#endif

#ifndef LIPSANK_PI
#define LIPSANK_PI 3.14159265358979323846f
#endif

#define LIPSANK_MAX_LOOKAHEAD 64
#define LIPSANK_NUM_BANDS 5

// -- Biquad filter for proper band separation ---------------------------------

typedef struct {
  float b0, b1, b2; // feedforward coefficients
  float a1, a2;     // feedback coefficients (a0 normalized to 1)
  float z1, z2;     // state variables
} LipsankBiquad;

// -- Second-order smoother (critically-damped spring) -------------------------
// This provides smooth, jitter-free animation with controllable response time

typedef struct {
  float position; // current value
  float velocity; // rate of change
  float target;   // target value
  float omega;    // natural frequency
  float zeta;     // damping ratio (1.0 = critically damped)
} LipsankSpring;

// -- Feature frame with richer spectral information ---------------------------

typedef struct {
  float rms;                      // root mean square amplitude
  float bands[LIPSANK_NUM_BANDS]; // energy in frequency bands
  float spectral_centroid;        // weighted average frequency
  float spectral_flux;            // change in spectrum (onset detection)
  float zcr;                      // zero crossing rate
  float voiced_ratio;             // estimate of voicing (periodic vs noise)
  float time_seconds;
} LipSyncFeatureFrame;

// -- Ring buffers -------------------------------------------------------------

typedef struct {
  float *data;
  size_t capacity;
  size_t head;
  size_t count;
} LipsankSampleRing;

typedef struct {
  LipSyncFeatureFrame *data;
  size_t capacity;
  size_t head;
  size_t count;
} LipsankFeatureRing;

// -- Main context structure ---------------------------------------------------

struct LipSyncContext {
  LipSyncConfig cfg;
  LipSyncCallback callback;
  void *user_data;

  LipsankSampleRing sample_ring;
  LipsankFeatureRing feature_ring;

  // Biquad filters for band separation (5 bands for formant-like analysis)
  LipsankBiquad band_filters[LIPSANK_NUM_BANDS];

  // Previous band energies for spectral flux
  float prev_bands[LIPSANK_NUM_BANDS];

  // Spring smoothers for each output parameter
  LipsankSpring spring_open;
  LipsankSpring spring_wide;
  LipsankSpring spring_round;
  LipsankSpring spring_tension;
  LipsankSpring spring_intensity;

  // Envelope followers
  float env_fast;    // fast envelope for transients
  float env_slow;    // slow envelope for normalization
  float noise_floor; // adaptive noise floor
  float peak_hold;   // recent peak for normalization

  // Noise gate state
  int gate_open;    // 1 = gate open (speaking), 0 = closed
  float gate_level; // current gate level (smoothed)

  // Processing state
  unsigned long long frames_seen;
  float dt; // time step between frames
  float alpha_env_fast;
  float alpha_env_slow;
  float alpha_noise;
  float alpha_peak_decay;

  int initialized;
};

// -- Utility functions --------------------------------------------------------

LIPSANK_STATIC LIPSANK_INLINE float lipsank_clampf(float v, float lo,
                                                   float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

LIPSANK_STATIC LIPSANK_INLINE float lipsank_minf(float a, float b) {
  return a < b ? a : b;
}

LIPSANK_STATIC LIPSANK_INLINE float lipsank_maxf(float a, float b) {
  return a > b ? a : b;
}

LIPSANK_STATIC LIPSANK_INLINE float lipsank_absf(float x) {
  return x < 0 ? -x : x;
}

LIPSANK_STATIC LIPSANK_INLINE float lipsank_lerpf(float a, float b, float t) {
  return a + t * (b - a);
}

LIPSANK_STATIC LIPSANK_INLINE float lipsank_smoothstep(float x) {
  x = lipsank_clampf(x, 0.0f, 1.0f);
  return x * x * (3.0f - 2.0f * x);
}

LIPSANK_STATIC LIPSANK_INLINE float lipsank_db_to_amp(float db) {
  return powf(10.0f, db / 20.0f);
}

LIPSANK_STATIC LIPSANK_INLINE float lipsank_amp_to_db(float amp) {
  const float eps = 1e-10f;
  return 20.0f * log10f(amp > eps ? amp : eps);
}

// -- Ring buffer operations ---------------------------------------------------

LIPSANK_STATIC void lipsank_sample_ring_free(LipsankSampleRing *r) {
  if (r->data)
    LIPSANK_FREE(r->data);
  r->data = NULL;
  r->capacity = r->head = r->count = 0;
}

LIPSANK_STATIC void lipsank_feature_ring_free(LipsankFeatureRing *r) {
  if (r->data)
    LIPSANK_FREE(r->data);
  r->data = NULL;
  r->capacity = r->head = r->count = 0;
}

LIPSANK_STATIC int lipsank_sample_ring_push(LipsankSampleRing *r, float value) {
  if (!r->data || r->count >= r->capacity)
    return 0;
  size_t idx = (r->head + r->count) % r->capacity;
  r->data[idx] = value;
  r->count++;
  return 1;
}

LIPSANK_STATIC float lipsank_sample_ring_get(const LipsankSampleRing *r,
                                             size_t idx) {
  size_t pos = (r->head + idx) % r->capacity;
  return r->data[pos];
}

LIPSANK_STATIC void lipsank_sample_ring_pop(LipsankSampleRing *r,
                                            size_t count) {
  if (count > r->count)
    count = r->count;
  r->head = (r->head + count) % r->capacity;
  r->count -= count;
}

LIPSANK_STATIC int lipsank_feature_ring_push(LipsankFeatureRing *r,
                                             const LipSyncFeatureFrame *f) {
  if (!r->data || r->count >= r->capacity)
    return 0;
  size_t idx = (r->head + r->count) % r->capacity;
  r->data[idx] = *f;
  r->count++;
  return 1;
}

LIPSANK_STATIC LipSyncFeatureFrame *
lipsank_feature_ring_peek(LipsankFeatureRing *r, size_t idx) {
  if (idx >= r->count)
    return NULL;
  size_t pos = (r->head + idx) % r->capacity;
  return &r->data[pos];
}

LIPSANK_STATIC void lipsank_feature_ring_pop_front(LipsankFeatureRing *r) {
  if (r->count == 0)
    return;
  r->head = (r->head + 1) % r->capacity;
  r->count--;
}

// -- Biquad filter implementation ---------------------------------------------

LIPSANK_STATIC void lipsank_biquad_init_bandpass(LipsankBiquad *bq,
                                                 float sample_rate,
                                                 float center_freq, float q) {
  // Design a 2nd-order bandpass filter using bilinear transform
  float w0 = 2.0f * LIPSANK_PI * center_freq / sample_rate;
  float cos_w0 = cosf(w0);
  float sin_w0 = sinf(w0);
  float alpha = sin_w0 / (2.0f * q);

  float a0 = 1.0f + alpha;
  bq->b0 = (alpha) / a0;
  bq->b1 = 0.0f;
  bq->b2 = (-alpha) / a0;
  bq->a1 = (-2.0f * cos_w0) / a0;
  bq->a2 = (1.0f - alpha) / a0;
  bq->z1 = bq->z2 = 0.0f;
}

LIPSANK_STATIC void lipsank_biquad_init_lowpass(LipsankBiquad *bq,
                                                float sample_rate, float cutoff,
                                                float q) {
  float w0 = 2.0f * LIPSANK_PI * cutoff / sample_rate;
  float cos_w0 = cosf(w0);
  float sin_w0 = sinf(w0);
  float alpha = sin_w0 / (2.0f * q);

  float a0 = 1.0f + alpha;
  bq->b0 = ((1.0f - cos_w0) / 2.0f) / a0;
  bq->b1 = (1.0f - cos_w0) / a0;
  bq->b2 = ((1.0f - cos_w0) / 2.0f) / a0;
  bq->a1 = (-2.0f * cos_w0) / a0;
  bq->a2 = (1.0f - alpha) / a0;
  bq->z1 = bq->z2 = 0.0f;
}

LIPSANK_STATIC void lipsank_biquad_init_highpass(LipsankBiquad *bq,
                                                 float sample_rate,
                                                 float cutoff, float q) {
  float w0 = 2.0f * LIPSANK_PI * cutoff / sample_rate;
  float cos_w0 = cosf(w0);
  float sin_w0 = sinf(w0);
  float alpha = sin_w0 / (2.0f * q);

  float a0 = 1.0f + alpha;
  bq->b0 = ((1.0f + cos_w0) / 2.0f) / a0;
  bq->b1 = (-(1.0f + cos_w0)) / a0;
  bq->b2 = ((1.0f + cos_w0) / 2.0f) / a0;
  bq->a1 = (-2.0f * cos_w0) / a0;
  bq->a2 = (1.0f - alpha) / a0;
  bq->z1 = bq->z2 = 0.0f;
}

LIPSANK_STATIC float lipsank_biquad_process(LipsankBiquad *bq, float x) {
  float y = bq->b0 * x + bq->z1;
  bq->z1 = bq->b1 * x - bq->a1 * y + bq->z2;
  bq->z2 = bq->b2 * x - bq->a2 * y;
  return y;
}

LIPSANK_STATIC void lipsank_biquad_reset(LipsankBiquad *bq) {
  bq->z1 = bq->z2 = 0.0f;
}

// -- Spring smoother implementation -------------------------------------------
// Uses a critically-damped second-order system for smooth, natural motion

LIPSANK_STATIC void lipsank_spring_init(LipsankSpring *s, float initial_value,
                                        float response_time_ms, float damping) {
  s->position = initial_value;
  s->velocity = 0.0f;
  s->target = initial_value;
  // Convert response time to natural frequency
  // For critically damped system, response time ≈ 4/omega
  s->omega = (response_time_ms > 0.0f) ? (4000.0f / response_time_ms) : 100.0f;
  s->zeta = lipsank_clampf(damping, 0.5f, 2.0f);
}

LIPSANK_STATIC void lipsank_spring_set_response(LipsankSpring *s,
                                                float response_time_ms) {
  s->omega = (response_time_ms > 0.0f) ? (4000.0f / response_time_ms) : 100.0f;
}

LIPSANK_STATIC float lipsank_spring_update(LipsankSpring *s, float target,
                                           float dt) {
  s->target = target;

  // Second-order system: x'' + 2*zeta*omega*x' + omega^2*x = omega^2*target
  // Using semi-implicit Euler for stability
  float omega = s->omega;
  float zeta = s->zeta;

  float error = s->target - s->position;
  float accel = omega * omega * error - 2.0f * zeta * omega * s->velocity;

  s->velocity += accel * dt;
  s->position += s->velocity * dt;

  // Prevent overshooting for critically/over-damped systems
  if (zeta >= 1.0f) {
    if ((s->velocity > 0.0f && s->position > s->target) ||
        (s->velocity < 0.0f && s->position < s->target)) {
      s->position = s->target;
      s->velocity = 0.0f;
    }
  }

  return s->position;
}

LIPSANK_STATIC float lipsank_spring_update_asymmetric(LipsankSpring *s,
                                                      float target, float dt,
                                                      float attack_omega,
                                                      float release_omega) {
  // Use different response times for increasing vs decreasing
  float omega = (target > s->position) ? attack_omega : release_omega;
  float zeta = s->zeta;

  float error = target - s->position;
  float accel = omega * omega * error - 2.0f * zeta * omega * s->velocity;

  s->velocity += accel * dt;
  s->position += s->velocity * dt;

  // Prevent overshooting
  if (zeta >= 1.0f) {
    if ((s->velocity > 0.0f && s->position > target) ||
        (s->velocity < 0.0f && s->position < target)) {
      s->position = target;
      s->velocity = 0.0f;
    }
  }

  s->target = target;
  return s->position;
}

// -- Feature extraction -------------------------------------------------------

LIPSANK_STATIC void lipsank_compute_features(LipSyncContext *ctx,
                                             LipSyncFeatureFrame *out) {
  const size_t N = (size_t)ctx->cfg.frame_size_samples;

  if (N == 0 || ctx->sample_ring.count < N) {
    memset(out, 0, sizeof(*out));
    return;
  }

  // Apply Hann window and compute features in one pass
  float energy = 0.0f;
  float zcr = 0.0f;
  float band_energy[LIPSANK_NUM_BANDS] = {0};
  float weighted_freq = 0.0f;

  // Reset biquad states for consistent frame analysis
  // (We use fresh filter passes per frame for better frequency isolation)
  LipsankBiquad temp_filters[LIPSANK_NUM_BANDS];
  memcpy(temp_filters, ctx->band_filters, sizeof(temp_filters));

  float prev_sample = 0.0f;
  int first = 1;

  // Center frequencies for the 5 bands (designed around speech formants)
  static const float band_centers[LIPSANK_NUM_BANDS] = {
      250.0f,  // F1 low (close vowels like 'i', 'u')
      500.0f,  // F1 mid (neutral vowels)
      1000.0f, // F1 high / F2 low (open vowels like 'a')
      2000.0f, // F2 mid (front vowels, consonant transitions)
      4000.0f  // High frequency (fricatives, sibilants)
  };

  for (size_t i = 0; i < N; ++i) {
    // Get sample with Hann window
    float raw = lipsank_sample_ring_get(&ctx->sample_ring, i);
    float window =
        0.5f * (1.0f - cosf(2.0f * LIPSANK_PI * (float)i / (float)(N - 1)));
    float x = raw * window;

    // Zero-crossing rate
    if (!first) {
      if ((prev_sample >= 0.0f && x < 0.0f) ||
          (prev_sample < 0.0f && x >= 0.0f)) {
        zcr += 1.0f;
      }
    }
    prev_sample = x;
    first = 0;

    // Total energy
    energy += x * x;

    // Band energies through biquad filters
    for (int b = 0; b < LIPSANK_NUM_BANDS; ++b) {
      float filtered = lipsank_biquad_process(&temp_filters[b], x);
      band_energy[b] += filtered * filtered;
    }
  }

  const float inv_n = 1.0f / (float)N;
  out->rms = sqrtf(energy * inv_n);
  out->zcr = (N > 1) ? (zcr / (float)(N - 1)) : 0.0f;

  // Normalize band energies and compute spectral centroid
  float band_total = 0.0f;
  for (int b = 0; b < LIPSANK_NUM_BANDS; ++b) {
    out->bands[b] = sqrtf(band_energy[b] * inv_n);
    band_total += out->bands[b];
    weighted_freq += out->bands[b] * band_centers[b];
  }

  out->spectral_centroid =
      (band_total > 1e-6f) ? (weighted_freq / band_total) : 1000.0f;

  // Spectral flux (change from previous frame)
  float flux = 0.0f;
  for (int b = 0; b < LIPSANK_NUM_BANDS; ++b) {
    float diff = out->bands[b] - ctx->prev_bands[b];
    if (diff > 0.0f)
      flux += diff * diff; // Only consider increases (onsets)
    ctx->prev_bands[b] = out->bands[b];
  }
  out->spectral_flux = sqrtf(flux);

  // Voiced ratio estimate (low ZCR + energy in low bands = voiced)
  float low_band_ratio = (band_total > 1e-6f)
                             ? ((out->bands[0] + out->bands[1]) / band_total)
                             : 0.5f;
  float zcr_voiced = 1.0f - lipsank_clampf(out->zcr * 3.0f, 0.0f, 1.0f);
  out->voiced_ratio = 0.5f * (low_band_ratio + zcr_voiced);

  // Timestamp
  const float center = (float)(ctx->frames_seen * ctx->cfg.frame_hop_samples) +
                       (float)(ctx->cfg.frame_size_samples * 0.5f);
  out->time_seconds = center / (float)ctx->cfg.sample_rate_hz;
}

// -- Trajectory planning using lookahead --------------------------------------

typedef struct {
  float open;
  float wide;
  float round;
  float tension;
  float weight; // confidence/energy weight
} LipsankPhonemeTarget;

LIPSANK_STATIC LipsankPhonemeTarget lipsank_compute_phoneme_target(
    const LipSyncFeatureFrame *frame, float noise_floor, float env_slow) {

  LipsankPhonemeTarget target = {0};
  const float eps = 1e-6f;

  // Compute normalized energy
  float norm_energy =
      (frame->rms - noise_floor) / (env_slow - noise_floor + eps);
  norm_energy = lipsank_clampf(norm_energy, 0.0f, 2.0f);

  // Band ratios for phoneme classification
  float band_sum = eps;
  for (int b = 0; b < LIPSANK_NUM_BANDS; ++b) {
    band_sum += frame->bands[b];
  }

  float low_ratio = (frame->bands[0] + frame->bands[1]) / band_sum; // <750 Hz
  float mid_ratio =
      (frame->bands[2] + frame->bands[3]) / band_sum; // 750-3000 Hz
  float high_ratio = frame->bands[4] / band_sum;      // >3000 Hz

  // Spectral centroid normalized (500-4000 Hz range)
  float centroid_norm = (frame->spectral_centroid - 500.0f) / 3500.0f;
  centroid_norm = lipsank_clampf(centroid_norm, 0.0f, 1.0f);

  // --- Bilabial consonant detection ('m', 'b', 'p', 'n') ---
  // These require closed or nearly-closed mouth
  // Characteristics:
  // - Nasals ('m', 'n'): voiced, very low spectral centroid, energy in low
  // bands only
  // - Plosives ('p', 'b'): very low energy (closure) or burst

  float bilabial_score = 0.0f;

  // Nasal detection: low ZCR (voiced) + very high low-band dominance + low
  // centroid
  float nasal_indicator =
      low_ratio * (1.0f - frame->zcr) * (1.0f - centroid_norm);
  if (nasal_indicator > 0.3f && norm_energy > 0.15f && norm_energy < 1.0f) {
    // Likely a nasal consonant - energy present but muffled
    bilabial_score =
        lipsank_clampf((nasal_indicator - 0.3f) * 2.5f, 0.0f, 0.8f);
  }

  // Plosive closure detection: very low energy
  if (norm_energy < 0.15f) {
    // Could be a stop consonant closure phase
    bilabial_score = lipsank_maxf(bilabial_score, 0.6f);
  }

  // --- Mouth opening ---
  // Opens more for loud sounds and open vowels (high F1 / low spectral
  // centroid)
  float open_base = powf(norm_energy, 0.55f); // Non-linear compression

  // Open vowels ('a', 'o') have energy in mid bands
  float vowel_open_boost = mid_ratio * 0.35f;

  // Reduce opening for high-frequency sounds (fricatives)
  float fricative_reduction = high_ratio * 0.25f;

  // Strong reduction for bilabial consonants
  float bilabial_reduction = bilabial_score * 0.9f;

  target.open = lipsank_clampf(open_base + vowel_open_boost -
                                   fricative_reduction - bilabial_reduction,
                               0.0f, 1.0f);

  // Further reduce open for detected bilabials
  if (bilabial_score > 0.4f) {
    target.open *= (1.0f - bilabial_score * 0.7f);
  }

  // --- Mouth width (smile) ---
  // Wide for front vowels ('i', 'e') - high F2 / high spectral centroid
  // Narrow for back vowels ('o', 'u') - low F2
  float wide_base = 0.25f + 0.4f * centroid_norm;

  // High ZCR often indicates wide mouth (unvoiced fricatives like 's')
  float zcr_wide = lipsank_clampf(frame->zcr * 2.0f, 0.0f, 1.0f) * 0.3f;

  // Bilabials are not wide
  float bilabial_wide_reduction = bilabial_score * 0.4f;

  target.wide = lipsank_clampf(wide_base + zcr_wide - low_ratio * 0.25f -
                                   bilabial_wide_reduction,
                               0.0f, 1.0f);

  // --- Lip rounding ---
  // Round for back vowels ('o', 'u'), labial consonants ('w')
  // Low spectral centroid + energy in low bands
  float round_base = 0.15f + low_ratio * 0.45f;

  // Decrease rounding for high-frequency content
  float round_decrease = (centroid_norm * 0.35f + high_ratio * 0.25f);

  target.round = lipsank_clampf(round_base - round_decrease, 0.0f, 1.0f);

  // --- Lip tension ---
  // High for labial consonants ('m', 'b', 'p', 'f', 'v')
  float tension_base = 0.05f;

  // Bilabials have high lip tension (lips pressed together)
  float bilabial_tension = bilabial_score * 0.8f;

  // Onsets and releases often involve lip tension
  float flux_tension = lipsank_clampf(frame->spectral_flux * 2.5f, 0.0f, 0.4f);

  // Fricatives ('f', 'v', 's') have high ZCR
  float fricative_tension = lipsank_clampf(frame->zcr * 1.2f, 0.0f, 0.35f);

  // Reduce tension for clearly open vowels
  float open_reduction = lipsank_clampf(target.open - 0.4f, 0.0f, 0.6f) * 0.4f;

  target.tension =
      lipsank_clampf(tension_base + bilabial_tension + flux_tension +
                         fricative_tension - open_reduction,
                     0.0f, 1.0f);

  // Weight based on energy (low weight for quiet frames, but not zero for
  // bilabial detection)
  target.weight =
      lipsank_clampf(norm_energy * 2.0f + bilabial_score * 0.3f, 0.0f, 1.0f);

  return target;
}

LIPSANK_STATIC LipsankPhonemeTarget
lipsank_plan_trajectory(LipSyncContext *ctx) {
  // Compute weighted average of current and future frames for smooth trajectory
  LipsankPhonemeTarget result = {0};
  float total_weight = 0.0f;

  const size_t lookahead = (size_t)ctx->cfg.lookahead_frames;
  const size_t frame_count =
      lipsank_minf(ctx->feature_ring.count, lookahead + 1);

  if (frame_count == 0) {
    result.weight = 0.0f;
    return result;
  }

  // Find peak energy in lookahead window for anticipation
  float peak_energy = 0.0f;
  for (size_t i = 0; i < frame_count; ++i) {
    LipSyncFeatureFrame *f = lipsank_feature_ring_peek(&ctx->feature_ring, i);
    if (f && f->rms > peak_energy) {
      peak_energy = f->rms;
    }
  }

  // Weight frames with temporal decay and energy-based importance
  for (size_t i = 0; i < frame_count; ++i) {
    LipSyncFeatureFrame *frame =
        lipsank_feature_ring_peek(&ctx->feature_ring, i);
    if (!frame)
      continue;

    LipsankPhonemeTarget target =
        lipsank_compute_phoneme_target(frame, ctx->noise_floor, ctx->env_slow);

    // Temporal weight: current frame most important, future frames less
    float temporal_weight = 1.0f / (1.0f + (float)i * 0.5f);

    // Energy weight: give more weight to loud frames (they're more important)
    float energy_weight = 0.5f + 0.5f * (frame->rms / (peak_energy + 1e-6f));

    float w = temporal_weight * energy_weight * target.weight;

    result.open += target.open * w;
    result.wide += target.wide * w;
    result.round += target.round * w;
    result.tension += target.tension * w;
    total_weight += w;
  }

  if (total_weight > 0.0f) {
    result.open /= total_weight;
    result.wide /= total_weight;
    result.round /= total_weight;
    result.tension /= total_weight;
  }

  result.weight = total_weight;

  // --- Anticipation for plosives ---
  // If we see a big energy jump coming, prepare for it
  LipSyncFeatureFrame *current =
      lipsank_feature_ring_peek(&ctx->feature_ring, 0);
  if (current && frame_count > 1) {
    float current_energy = current->rms;

    // Check if a significant onset is coming
    if (peak_energy > current_energy * 2.5f &&
        current_energy < ctx->env_slow * 0.3f) {
      // Pre-plosive: close mouth slightly, increase tension
      result.open *= 0.5f;
      result.tension = lipsank_clampf(result.tension + 0.3f, 0.0f, 1.0f);
    }
  }

  return result;
}

// -- Parameter mapping with gate and smoothing --------------------------------

LIPSANK_STATIC LipSyncMouthParams lipsank_map_params(LipSyncContext *ctx) {
  LipSyncMouthParams out = {0};

  if (ctx->feature_ring.count == 0) {
    // Update springs even with no data (decay to rest)
    out.open = lipsank_spring_update(&ctx->spring_open, 0.0f, ctx->dt);
    out.wide = lipsank_spring_update(&ctx->spring_wide, 0.3f,
                                     ctx->dt); // Neutral is slightly wide
    out.round = lipsank_spring_update(&ctx->spring_round, 0.0f, ctx->dt);
    out.tension = lipsank_spring_update(&ctx->spring_tension, 0.0f, ctx->dt);
    out.intensity =
        lipsank_spring_update(&ctx->spring_intensity, 0.0f, ctx->dt);
    return out;
  }

  LipSyncFeatureFrame *cur = lipsank_feature_ring_peek(&ctx->feature_ring, 0);
  if (!cur)
    return out;

  // --- Update envelope followers ---
  ctx->env_fast += ctx->alpha_env_fast * (cur->rms - ctx->env_fast);
  ctx->env_slow += ctx->alpha_env_slow * (cur->rms - ctx->env_slow);

  // Update peak hold with decay
  if (cur->rms > ctx->peak_hold) {
    ctx->peak_hold = cur->rms;
  } else {
    ctx->peak_hold *= (1.0f - ctx->alpha_peak_decay);
  }

  // Update noise floor (track minimum with slow adaptation)
  if (cur->rms < ctx->noise_floor) {
    ctx->noise_floor = ctx->noise_floor * 0.99f + cur->rms * 0.01f;
  } else {
    ctx->noise_floor *= (1.0f - ctx->alpha_noise);
  }

  // --- Noise gate with hysteresis ---
  float gate_thresh_amp = lipsank_db_to_amp(ctx->cfg.gate_threshold_db);
  float gate_hyst_amp = lipsank_db_to_amp(ctx->cfg.gate_threshold_db +
                                          ctx->cfg.gate_hysteresis_db);

  if (!ctx->gate_open) {
    // Gate is closed, need energy above threshold + hysteresis to open
    if (cur->rms > gate_hyst_amp) {
      ctx->gate_open = 1;
    }
  } else {
    // Gate is open, stays open until energy drops below threshold
    if (cur->rms < gate_thresh_amp) {
      ctx->gate_open = 0;
    }
  }

  // Smooth gate level for gradual open/close
  float gate_target = ctx->gate_open ? 1.0f : 0.0f;
  ctx->gate_level += 0.15f * (gate_target - ctx->gate_level); // ~30ms smoothing

  // --- Compute trajectory-planned targets ---
  LipsankPhonemeTarget target = lipsank_plan_trajectory(ctx);

  // Apply gate
  target.open *= ctx->gate_level;
  target.tension *= ctx->gate_level;

  // When gate is closing, move toward neutral pose
  if (ctx->gate_level < 0.5f) {
    float blend = ctx->gate_level * 2.0f; // 0-1 over gate closing
    target.wide = lipsank_lerpf(0.35f, target.wide, blend);
    target.round = lipsank_lerpf(0.0f, target.round, blend);
  }

  // --- Compute intensity ---
  const float eps = 1e-6f;
  float intensity_target =
      (cur->rms - ctx->noise_floor) / (ctx->peak_hold - ctx->noise_floor + eps);
  intensity_target = lipsank_clampf(intensity_target, 0.0f, 1.0f);
  intensity_target *= ctx->gate_level;

  // --- Apply asymmetric spring smoothing ---
  float attack_omega =
      (ctx->cfg.attack_ms > 0.0f) ? (4000.0f / ctx->cfg.attack_ms) : 100.0f;
  float release_omega =
      (ctx->cfg.release_ms > 0.0f) ? (4000.0f / ctx->cfg.release_ms) : 50.0f;

  out.open = lipsank_spring_update_asymmetric(
      &ctx->spring_open, target.open, ctx->dt, attack_omega, release_omega);
  out.wide = lipsank_spring_update_asymmetric(&ctx->spring_wide, target.wide,
                                              ctx->dt, attack_omega * 0.7f,
                                              release_omega * 0.7f);
  out.round = lipsank_spring_update_asymmetric(&ctx->spring_round, target.round,
                                               ctx->dt, attack_omega * 0.8f,
                                               release_omega);
  out.tension = lipsank_spring_update_asymmetric(
      &ctx->spring_tension, target.tension, ctx->dt, attack_omega * 1.2f,
      release_omega * 0.6f);
  out.intensity =
      lipsank_spring_update(&ctx->spring_intensity, intensity_target, ctx->dt);

  // Final clamp
  out.open = lipsank_clampf(out.open, 0.0f, 1.0f);
  out.wide = lipsank_clampf(out.wide, 0.0f, 1.0f);
  out.round = lipsank_clampf(out.round, 0.0f, 1.0f);
  out.tension = lipsank_clampf(out.tension, 0.0f, 1.0f);
  out.intensity = lipsank_clampf(out.intensity, 0.0f, 1.0f);

  return out;
}

// -- Config defaults ----------------------------------------------------------

LipSyncConfig lipsync_default_config(int sample_rate_hz) {
  LipSyncConfig cfg;
  cfg.sample_rate_hz = (sample_rate_hz > 0) ? sample_rate_hz : 16000;
  cfg.frame_hop_samples = (int)(cfg.sample_rate_hz * 0.010f);  // 10 ms hop
  cfg.frame_size_samples = (int)(cfg.sample_rate_hz * 0.025f); // 25 ms window
  cfg.lookahead_frames = 4; // 40 ms lookahead

  cfg.attack_ms = 35.0f;  // Fast attack for responsiveness
  cfg.release_ms = 80.0f; // Slower release for natural decay
  cfg.intensity_smooth_ms = 50.0f;
  cfg.spring_damping = 1.0f; // Critically damped (no overshoot)

  cfg.gate_threshold_db = -40.0f;
  cfg.gate_hysteresis_db = 6.0f;

  cfg.initial_noise_db = -55.0f;

  return cfg;
}

// -- Lifecycle ----------------------------------------------------------------

void lipsync_init(LipSyncContext *ctx, const LipSyncConfig *config,
                  LipSyncCallback cb, void *user_data) {
  if (!ctx)
    return;
  memset(ctx, 0, sizeof(*ctx));

  LipSyncConfig cfg = config ? *config : lipsync_default_config(16000);

  // Validate and fix config
  if (cfg.sample_rate_hz <= 0)
    cfg.sample_rate_hz = 16000;
  if (cfg.frame_hop_samples <= 0)
    cfg.frame_hop_samples = (int)(cfg.sample_rate_hz * 0.010f);
  if (cfg.frame_size_samples <= cfg.frame_hop_samples)
    cfg.frame_size_samples = cfg.frame_hop_samples * 2;
  if (cfg.lookahead_frames < 0)
    cfg.lookahead_frames = 0;
  if (cfg.lookahead_frames > LIPSANK_MAX_LOOKAHEAD)
    cfg.lookahead_frames = LIPSANK_MAX_LOOKAHEAD;
  if (cfg.attack_ms <= 0.0f)
    cfg.attack_ms = 35.0f;
  if (cfg.release_ms <= 0.0f)
    cfg.release_ms = 80.0f;
  if (cfg.intensity_smooth_ms <= 0.0f)
    cfg.intensity_smooth_ms = 50.0f;
  if (cfg.spring_damping <= 0.0f)
    cfg.spring_damping = 1.0f;
  if (cfg.gate_threshold_db > 0.0f)
    cfg.gate_threshold_db = -fabsf(cfg.gate_threshold_db);
  if (cfg.gate_hysteresis_db <= 0.0f)
    cfg.gate_hysteresis_db = 6.0f;
  if (cfg.initial_noise_db > 0.0f)
    cfg.initial_noise_db = -fabsf(cfg.initial_noise_db);

  ctx->cfg = cfg;
  ctx->callback = cb;
  ctx->user_data = user_data;

  // Compute time step
  ctx->dt = (float)cfg.frame_hop_samples / (float)cfg.sample_rate_hz;

  // Allocate sample ring buffer
  const size_t ring_needed =
      (size_t)cfg.frame_size_samples +
      (size_t)(cfg.lookahead_frames + 4) * (size_t)cfg.frame_hop_samples +
      (size_t)cfg.frame_size_samples;
  ctx->sample_ring.data = (float *)LIPSANK_MALLOC(sizeof(float) * ring_needed);
  ctx->sample_ring.capacity = ring_needed;
  ctx->sample_ring.head = ctx->sample_ring.count = 0;

  // Allocate feature ring buffer
  const size_t feat_cap = (size_t)cfg.lookahead_frames + 8;
  ctx->feature_ring.data = (LipSyncFeatureFrame *)LIPSANK_MALLOC(
      sizeof(LipSyncFeatureFrame) * feat_cap);
  ctx->feature_ring.capacity = feat_cap;
  ctx->feature_ring.head = ctx->feature_ring.count = 0;

  // Initialize biquad filters for 5 frequency bands
  // Band centers chosen to capture speech formant regions
  static const float centers[LIPSANK_NUM_BANDS] = {250.0f, 500.0f, 1000.0f,
                                                   2000.0f, 4000.0f};
  static const float qs[LIPSANK_NUM_BANDS] = {1.5f, 1.5f, 1.2f, 1.2f, 1.0f};

  for (int b = 0; b < LIPSANK_NUM_BANDS; ++b) {
    float center = centers[b];
    // Clamp center frequency to valid range for sample rate
    float max_freq = (float)cfg.sample_rate_hz * 0.45f;
    if (center > max_freq)
      center = max_freq;
    lipsank_biquad_init_bandpass(&ctx->band_filters[b],
                                 (float)cfg.sample_rate_hz, center, qs[b]);
    ctx->prev_bands[b] = 0.0f;
  }

  // Initialize spring smoothers
  float avg_response = (cfg.attack_ms + cfg.release_ms) * 0.5f;
  lipsank_spring_init(&ctx->spring_open, 0.0f, avg_response,
                      cfg.spring_damping);
  lipsank_spring_init(&ctx->spring_wide, 0.35f, avg_response,
                      cfg.spring_damping); // Neutral is slightly wide
  lipsank_spring_init(&ctx->spring_round, 0.0f, avg_response,
                      cfg.spring_damping);
  lipsank_spring_init(&ctx->spring_tension, 0.0f, avg_response * 0.7f,
                      cfg.spring_damping);
  lipsank_spring_init(&ctx->spring_intensity, 0.0f, cfg.intensity_smooth_ms,
                      cfg.spring_damping);

  // Initialize envelope followers
  const float initial_amp = lipsank_db_to_amp(cfg.initial_noise_db);
  ctx->env_fast = initial_amp * 3.0f;
  ctx->env_slow = initial_amp * 5.0f;
  ctx->noise_floor = initial_amp;
  ctx->peak_hold = initial_amp * 5.0f;

  // Compute alpha coefficients for envelope followers
  // alpha = 1 - exp(-dt/tau)
  ctx->alpha_env_fast = 1.0f - expf(-ctx->dt / 0.015f);   // 15 ms
  ctx->alpha_env_slow = 1.0f - expf(-ctx->dt / 0.300f);   // 300 ms
  ctx->alpha_noise = 1.0f - expf(-ctx->dt / 2.0f);        // 2 s
  ctx->alpha_peak_decay = 1.0f - expf(-ctx->dt / 0.500f); // 500 ms

  // Gate state
  ctx->gate_open = 0;
  ctx->gate_level = 0.0f;

  ctx->frames_seen = 0;
  ctx->initialized = (ctx->sample_ring.data && ctx->feature_ring.data) ? 1 : 0;
}

void lipsync_reset(LipSyncContext *ctx) {
  if (!ctx || !ctx->initialized)
    return;

  ctx->sample_ring.head = ctx->sample_ring.count = 0;
  ctx->feature_ring.head = ctx->feature_ring.count = 0;
  ctx->frames_seen = 0;

  // Reset springs to neutral
  ctx->spring_open.position = ctx->spring_open.target = 0.0f;
  ctx->spring_open.velocity = 0.0f;
  ctx->spring_wide.position = ctx->spring_wide.target = 0.35f;
  ctx->spring_wide.velocity = 0.0f;
  ctx->spring_round.position = ctx->spring_round.target = 0.0f;
  ctx->spring_round.velocity = 0.0f;
  ctx->spring_tension.position = ctx->spring_tension.target = 0.0f;
  ctx->spring_tension.velocity = 0.0f;
  ctx->spring_intensity.position = ctx->spring_intensity.target = 0.0f;
  ctx->spring_intensity.velocity = 0.0f;

  // Reset envelopes
  const float initial_amp = lipsank_db_to_amp(ctx->cfg.initial_noise_db);
  ctx->env_fast = initial_amp * 3.0f;
  ctx->env_slow = initial_amp * 5.0f;
  ctx->noise_floor = initial_amp;
  ctx->peak_hold = initial_amp * 5.0f;

  // Reset gate
  ctx->gate_open = 0;
  ctx->gate_level = 0.0f;

  // Reset biquad states
  for (int b = 0; b < LIPSANK_NUM_BANDS; ++b) {
    lipsank_biquad_reset(&ctx->band_filters[b]);
    ctx->prev_bands[b] = 0.0f;
  }
}

int lipsync_ready(const LipSyncContext *ctx) { return ctx && ctx->initialized; }

// -- Processing ---------------------------------------------------------------

LIPSANK_STATIC void lipsank_process_ready_frames(LipSyncContext *ctx) {
  while (ctx->feature_ring.count > (size_t)ctx->cfg.lookahead_frames) {
    LipSyncMouthParams params = lipsank_map_params(ctx);
    LipSyncFeatureFrame *cur = lipsank_feature_ring_peek(&ctx->feature_ring, 0);
    if (ctx->callback && cur) {
      ctx->callback(ctx->user_data, cur->time_seconds, &params);
    }
    lipsank_feature_ring_pop_front(&ctx->feature_ring);
  }
}

LIPSANK_STATIC void lipsank_feed_sample(LipSyncContext *ctx, float sample) {
  if (ctx->sample_ring.count >= ctx->sample_ring.capacity) {
    lipsank_sample_ring_pop(&ctx->sample_ring, ctx->cfg.frame_hop_samples);
  }
  lipsank_sample_ring_push(&ctx->sample_ring, sample);
}

void lipsync_feed_pcm16(LipSyncContext *ctx, const int16_t *samples,
                        size_t count) {
  if (!ctx || !ctx->initialized || !samples)
    return;

  const float inv_scale = 1.0f / 32768.0f;
  for (size_t i = 0; i < count; ++i) {
    lipsank_feed_sample(ctx, (float)samples[i] * inv_scale);
  }

  // Process complete frames
  while (ctx->sample_ring.count >= (size_t)ctx->cfg.frame_size_samples) {
    LipSyncFeatureFrame feat;
    lipsank_compute_features(ctx, &feat);
    lipsank_feature_ring_push(&ctx->feature_ring, &feat);
    ctx->frames_seen++;
    lipsank_sample_ring_pop(&ctx->sample_ring,
                            (size_t)ctx->cfg.frame_hop_samples);
  }

  lipsank_process_ready_frames(ctx);
}

void lipsync_feed_float(LipSyncContext *ctx, const float *samples,
                        size_t count) {
  if (!ctx || !ctx->initialized || !samples)
    return;

  for (size_t i = 0; i < count; ++i) {
    // Clamp input to valid range
    float s = samples[i];
    if (s > 1.0f)
      s = 1.0f;
    if (s < -1.0f)
      s = -1.0f;
    lipsank_feed_sample(ctx, s);
  }

  // Process complete frames
  while (ctx->sample_ring.count >= (size_t)ctx->cfg.frame_size_samples) {
    LipSyncFeatureFrame feat;
    lipsank_compute_features(ctx, &feat);
    lipsank_feature_ring_push(&ctx->feature_ring, &feat);
    ctx->frames_seen++;
    lipsank_sample_ring_pop(&ctx->sample_ring,
                            (size_t)ctx->cfg.frame_hop_samples);
  }

  lipsank_process_ready_frames(ctx);
}

// -- Cleanup ------------------------------------------------------------------

void lipsync_dispose(LipSyncContext *ctx) {
  if (!ctx)
    return;
  lipsank_sample_ring_free(&ctx->sample_ring);
  lipsank_feature_ring_free(&ctx->feature_ring);
  ctx->initialized = 0;
}

#endif // LIPSANK_IMPLEMENTATION

#endif // LIPSANK_H