cmake_minimum_required(VERSION 3.12)
project(lipsank LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# WebAssembly Build Support (Emscripten)
# ============================================================================
option(LIPSANK_BUILD_WASM "Build WebAssembly version for web demo" OFF)

if(LIPSANK_BUILD_WASM)
    # Check if Emscripten compiler is available
    find_program(EMCC_EXECUTABLE emcc)
    if(NOT EMCC_EXECUTABLE)
        message(FATAL_ERROR "Emscripten emcc not found. Please install and activate Emscripten SDK:\n"
            "  git clone https://github.com/emscripten-core/emsdk.git\n"
            "  cd emsdk\n"
            "  ./emsdk install latest\n"
            "  ./emsdk activate latest\n"
            "  source ./emsdk_env.sh")
    endif()
    
    message(STATUS "Found Emscripten: ${EMCC_EXECUTABLE}")
    
    # Get Emscripten version
    execute_process(
        COMMAND ${EMCC_EXECUTABLE} --version
        OUTPUT_VARIABLE EMCC_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Using Emscripten: ${EMCC_VERSION}")
    
    # Create docs directory if it doesn't exist
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs)
    
    # Set up exported functions and runtime methods using single quotes for Windows compatibility
    set(EXPORTED_FUNCTIONS "['_lipsync_wasm_init','_lipsync_wasm_reset','_lipsync_wasm_feed_float','_lipsync_wasm_feed_int16','_lipsync_wasm_results_available','_lipsync_wasm_read_result','_lipsync_wasm_read_batch','_lipsync_wasm_dispose','_lipsync_wasm_get_frame_time','_lipsync_wasm_alloc_float_buffer','_lipsync_wasm_free_buffer','_malloc','_free']")
    set(EXPORTED_RUNTIME_METHODS "['ccall','cwrap','HEAPF32']")
    
    # Custom command to build WASM module
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/docs/lipsync.js ${CMAKE_CURRENT_SOURCE_DIR}/docs/lipsync.wasm
        COMMAND ${EMCC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/docs/lipsync_wasm.cpp
            -I${CMAKE_CURRENT_SOURCE_DIR}
            -o ${CMAKE_CURRENT_SOURCE_DIR}/docs/lipsync.js
            -O3
            -s WASM=1
            -s MODULARIZE=0
            -s EXPORT_ES6=0
            -s EXPORTED_FUNCTIONS=${EXPORTED_FUNCTIONS}
            -s EXPORTED_RUNTIME_METHODS=${EXPORTED_RUNTIME_METHODS}
            -s ALLOW_MEMORY_GROWTH=1
            -s INITIAL_MEMORY=16777216
            -s MAXIMUM_MEMORY=268435456
            -s NO_EXIT_RUNTIME=1
            -s ASSERTIONS=0
            -s SAFE_HEAP=0
            --no-entry
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/docs/lipsync_wasm.cpp ${CMAKE_CURRENT_SOURCE_DIR}/lipsank.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building WebAssembly module with Emscripten"
    )
    
    # Custom target for WASM build
    add_custom_target(lipsank_wasm ALL
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/docs/lipsync.js ${CMAKE_CURRENT_SOURCE_DIR}/docs/lipsync.wasm
        COMMENT "WebAssembly build complete"
    )
    
    # Set output files as generated to avoid conflicts
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/lipsync.js
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/lipsync.wasm
        PROPERTIES GENERATED TRUE
    )
    
    message(STATUS "WebAssembly target configured. Output will be in docs/")
endif()

if(WIN32)
    add_executable(wav_wasapi examples/wav_wasapi.cpp)
    target_include_directories(wav_wasapi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(MSVC)
        target_compile_definitions(wav_wasapi PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    target_link_libraries(wav_wasapi PRIVATE Ole32 Mmdevapi Avrt Uuid)
endif()
