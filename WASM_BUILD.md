# WebAssembly Build Instructions

This document explains how to build the WebAssembly version of Lipsank using the integrated CMake build system.

## Prerequisites

1. **Emscripten SDK**: Install and activate the Emscripten SDK
   ```bash
   git clone https://github.com/emscripten-core/emsdk.git
   cd emsdk
   ./emsdk install latest
   ./emsdk activate latest
   source ./emsdk_env.sh  # On Windows: emsdk_env.bat
   ```

2. **Verify Installation**: Make sure `emcc` is on your PATH
   ```bash
   emcc --version
   ```

## Building the WebAssembly Module

### Using CMake

1. **Configure with WASM support enabled**:
   ```bash
   cmake -DLIPSANK_BUILD_WASM=ON -B build
   ```

2. **Build the project**:
   ```bash
   cmake --build build
   ```

   This will build all targets including the WebAssembly module.

### Building Only the WASM Module

If you want to build just the WebAssembly module:
```bash
cmake --build build --target lipsank_wasm
```

## Output Files

The WebAssembly build will generate the following files in the `docs/` directory:
- `lipsync.js` - JavaScript glue code
- `lipsync.wasm` - WebAssembly binary

These files are designed to work with the existing `docs/index.html` demo page.

## Running the Demo

After building the WebAssembly module:

1. Navigate to the docs directory:
   ```bash
   cd docs
   ```

2. Start a local web server:
   ```bash
   # Using Python
   python3 -m http.server 8080
   
   # Or using Node.js
   npx serve .
   ```

3. Open your browser and navigate to `http://localhost:8080`

## Troubleshooting

### "emcc not found" error
Make sure Emscripten SDK is installed and activated, and that `emcc` is on your PATH.

### Build fails with other errors
1. Ensure you have the latest Emscripten SDK
2. Check that all source files are present:
   - `docs/lipsync_wasm.cpp`
   - `lipsank.h`
3. Verify write permissions to the `docs/` directory

### WebAssembly doesn't load in browser
1. Check browser console for errors
2. Ensure the web server is serving files with correct MIME types
3. Make sure both `lipsync.js` and `lipsync.wasm` are accessible
