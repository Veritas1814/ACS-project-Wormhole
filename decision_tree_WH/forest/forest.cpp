// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <decision_forest.h>

using namespace tt;
using namespace tt::tt_metal;

int main() {
    // Example float weights
    std::vector<float> forest = {1.0f, 2.0f, 3.0f, 4.0f};
    uint32_t forest_size = forest.size();

    /* Device and program setup */
    IDevice* device = CreateDevice(0);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    CoreRange cores({0, 0}, {0, 0}); // use 1 core for test

    // Create DRAM buffer for forest weights
    size_t forest_bytes = forest_size * sizeof(float);
    auto forest_buffer = CreateBuffer(InterleavedBufferConfig{
        .device = device,
        .size = forest_bytes,
        .page_size = forest_bytes,
        .buffer_type = BufferType::DRAM
    });

    // Create DRAM buffer for result
    auto result_buffer = CreateBuffer(InterleavedBufferConfig{
        .device = device,
        .size = forest_bytes,
        .page_size = forest_bytes,
        .buffer_type = BufferType::DRAM
    });

    // Write forest data to DRAM
    EnqueueWriteBuffer(cq, forest_buffer, forest, false);

    // Create circular buffer in L1
    constexpr uint32_t cb_index = CBIndex::c_0;
    auto cb = CreateCircularBuffer(program, cores,
        CircularBufferConfig(forest_size, {{cb_index, DataFormat::Float32}})
        .set_page_size(cb_index, forest_size));

    // Attach data movement kernels
    auto reader_kernel = CreateKernel(program, "kernels/dataflow/forest_reader.cpp", cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto writer_kernel = CreateKernel(program, "kernels/dataflow/forest_writer.cpp", cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Set runtime args for core
    CoreCoord core = {0, 0};
    SetRuntimeArgs(program, reader_kernel, core, {forest_buffer->address(), forest_size});
    SetRuntimeArgs(program, writer_kernel, core, {result_buffer->address(), forest_size});

    // Launch program
    EnqueueProgram(cq, program, false);
    Finish(cq);

    // Read back result
    std::vector<float> result(forest_size);
    EnqueueReadBuffer(cq, result_buffer, result, true);

    // Print round-trip values
    for (size_t i = 0; i < forest_size; ++i)
        printf("Result[%zu] = %.2f\n", i, result[i]);

    CloseDevice(device);
    return 0;
}
