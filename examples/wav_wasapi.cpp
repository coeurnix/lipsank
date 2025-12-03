#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif

#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <mmreg.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <cstring>
#include <thread>
#include <vector>

#define LIPSANK_IMPLEMENTATION
#include "../lipsank.h"

static LipSyncMouthParams g_last = {0, 0, 0, 0};
static float g_last_time = 0.0f;

static void on_mouth(void *, float t, const LipSyncMouthParams *p) {
    g_last = *p;
    g_last_time = t;
}

struct WavData {
    int sample_rate = 0;
    int channels = 0;
    std::vector<int16_t> samples; // mono after possible down-mix
};

static bool load_wav(const std::filesystem::path &path, WavData &out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "Failed to open WAV file.\n");
        return false;
    }

    auto read_u32 = [&](uint32_t &v) { f.read(reinterpret_cast<char *>(&v), sizeof(v)); };
    auto read_u16 = [&](uint16_t &v) { f.read(reinterpret_cast<char *>(&v), sizeof(v)); };

    char riff[4], wave[4];
    uint32_t riff_size = 0;
    f.read(riff, 4);
    read_u32(riff_size);
    f.read(wave, 4);
    if (std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE") {
        std::fprintf(stderr, "Not a WAV file.\n");
        return false;
    }

    uint16_t fmt_code = 0, channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0;
    bool have_fmt = false, have_data = false;
    std::vector<char> data_bytes;

    while (f && (!have_fmt || !have_data)) {
        char chunk_id[4];
        uint32_t chunk_size = 0;
        f.read(chunk_id, 4);
        if (!f.read(reinterpret_cast<char *>(&chunk_size), sizeof(chunk_size))) break;
        std::string id(chunk_id, 4);
        if (id == "fmt ") {
            have_fmt = true;
            read_u16(fmt_code);
            read_u16(channels);
            read_u32(sample_rate);
            uint32_t byte_rate = 0;
            read_u32(byte_rate);
            uint16_t block_align = 0;
            read_u16(block_align);
            read_u16(bits_per_sample);
            if (chunk_size > 16) {
                f.seekg(chunk_size - 16, std::ios::cur);
            }
        } else if (id == "data") {
            have_data = true;
            data_bytes.resize(chunk_size);
            if (!f.read(data_bytes.data(), chunk_size)) {
                std::fprintf(stderr, "Unexpected EOF in data chunk.\n");
                return false;
            }
        } else {
            f.seekg(chunk_size, std::ios::cur);
        }
    }

    if (!have_fmt || !have_data) {
        std::fprintf(stderr, "Incomplete WAV (missing fmt or data).\n");
        return false;
    }
    if (fmt_code != 1 || bits_per_sample != 16) {
        std::fprintf(stderr, "Only PCM16 WAV files are supported.\n");
        return false;
    }
    if (channels < 1 || channels > 2) {
        std::fprintf(stderr, "Only mono or stereo WAV files are supported.\n");
        return false;
    }

    const size_t frames = data_bytes.size() / (bits_per_sample / 8) / channels;
    out.samples.resize(frames);
    const int16_t *raw = reinterpret_cast<const int16_t *>(data_bytes.data());
    if (channels == 1) {
        std::copy(raw, raw + frames, out.samples.begin());
    } else {
        for (size_t i = 0; i < frames; ++i) {
            int32_t l = raw[i * 2 + 0];
            int32_t r = raw[i * 2 + 1];
            out.samples[i] = static_cast<int16_t>((l + r) / 2);
        }
    }
    out.channels = 1;
    out.sample_rate = static_cast<int>(sample_rate);
    return true;
}

static void ansi_header() {
    std::printf("\033[2J\033[H"); // clear + home
    std::puts("Lipsank WASAPI demo");
    std::puts("Press Ctrl+C to quit.");
}

static void print_meter() {
    std::printf("\033[H"); // home
    std::printf("t=%6.2f  open=%0.2f  wide=%0.2f  round=%0.2f  intensity=%0.2f\n",
                g_last_time, g_last.open, g_last.wide, g_last.round, g_last.intensity);
    std::fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: wav_wasapi.exe <mono_pcm16.wav>\n");
        return 1;
    }

    std::filesystem::path wav_path = argv[1];
    WavData wav;
    if (!load_wav(wav_path, wav)) return 1;

    LipSyncContext lips;
    LipSyncConfig cfg = lipsync_default_config(wav.sample_rate);
    cfg.lookahead_frames = 2;
    lipsync_init(&lips, &cfg, on_mouth, nullptr);
    if (!lipsync_ready(&lips)) {
        std::fprintf(stderr, "Failed to initialize Lipsank.\n");
        return 1;
    }

    ansi_header();

    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr)) {
        std::fprintf(stderr, "CoInitializeEx failed: 0x%08lx\n", hr);
        return 1;
    }

    IMMDeviceEnumerator *enumerator = nullptr;
    IMMDevice *device = nullptr;
    IAudioClient *client = nullptr;
    IAudioRenderClient *render = nullptr;
    bool client_started = false;

    WAVEFORMATEX wf = {};
    wf.wFormatTag = WAVE_FORMAT_PCM;
    wf.nChannels = 1;
    wf.nSamplesPerSec = wav.sample_rate;
    wf.wBitsPerSample = 16;
    wf.nBlockAlign = wf.nChannels * wf.wBitsPerSample / 8;
    wf.nAvgBytesPerSec = wf.nSamplesPerSec * wf.nBlockAlign;
    const REFERENCE_TIME buffer_duration = 1000000; // 100 ms
    UINT32 buffer_frames = 0;
    size_t cursor = 0;
    const size_t total_frames = wav.samples.size();
    auto sleep_short = [] { std::this_thread::sleep_for(std::chrono::milliseconds(4)); };

    do {
        hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                              __uuidof(IMMDeviceEnumerator),
                              reinterpret_cast<void **>(&enumerator));
        if (FAILED(hr)) {
            std::fprintf(stderr, "Cannot create device enumerator: 0x%08lx\n", hr);
            break;
        }
        hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
        if (FAILED(hr)) {
            std::fprintf(stderr, "Cannot get default audio endpoint: 0x%08lx\n", hr);
            break;
        }
        hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void **)&client);
        if (FAILED(hr)) {
            std::fprintf(stderr, "Failed to activate audio client: 0x%08lx\n", hr);
            break;
        }
        hr = client->IsFormatSupported(AUDCLNT_SHAREMODE_SHARED, &wf, nullptr);
        if (FAILED(hr)) {
            std::fprintf(stderr, "Requested format not directly supported; attempting auto-convert.\n");
        }
        hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
                                buffer_duration, 0, &wf, nullptr);
        if (FAILED(hr)) {
            std::fprintf(stderr, "AudioClient Initialize failed: 0x%08lx\n", hr);
            break;
        }
        hr = client->GetBufferSize(&buffer_frames);
        if (FAILED(hr)) {
            std::fprintf(stderr, "GetBufferSize failed: 0x%08lx\n", hr);
            break;
        }
        hr = client->GetService(__uuidof(IAudioRenderClient), (void **)&render);
        if (FAILED(hr)) {
            std::fprintf(stderr, "GetService(RenderClient) failed: 0x%08lx\n", hr);
            break;
        }
        hr = client->Start();
        if (FAILED(hr)) {
            std::fprintf(stderr, "AudioClient Start failed: 0x%08lx\n", hr);
            break;
        }
        client_started = true;

        while (cursor < total_frames) {
            UINT32 padding = 0;
            hr = client->GetCurrentPadding(&padding);
            if (FAILED(hr)) break;
            UINT32 avail = buffer_frames > padding ? (buffer_frames - padding) : 0;
            if (avail == 0) {
                sleep_short();
                continue;
            }
            UINT32 frames_to_write = static_cast<UINT32>(
                std::min<size_t>(avail, total_frames - cursor));
            BYTE *data = nullptr;
            hr = render->GetBuffer(frames_to_write, &data);
            if (FAILED(hr)) break;

            memcpy(data, &wav.samples[cursor], frames_to_write * sizeof(int16_t));
            lipsync_feed_pcm16(&lips, &wav.samples[cursor], frames_to_write);
            cursor += frames_to_write;
            render->ReleaseBuffer(frames_to_write, 0);
            print_meter();
        }

        // Flush lookahead by feeding a tiny slice of silence.
        if (cfg.lookahead_frames > 0) {
            std::vector<int16_t> silence(cfg.lookahead_frames * cfg.frame_hop_samples, 0);
            lipsync_feed_pcm16(&lips, silence.data(), silence.size());
            print_meter();
        }

        // Let the tail play out briefly.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

    } while (false);

    if (client_started && client) client->Stop();
    if (render) render->Release();
    if (client) client->Release();
    if (device) device->Release();
    if (enumerator) enumerator->Release();
    CoUninitialize();
    std::puts("\nDone.");
    return 0;
}
