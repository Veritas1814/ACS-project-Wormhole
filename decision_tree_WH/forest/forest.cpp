// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <decision_tree_final.h>

using namespace tt;
using namespace tt::tt_metal;
void readCSV(const std::string& filename, std::vector<float>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            data.push_back(std::stod(value));
        }
    }
}
int main() {
    // Example float weights
    std::vector<float> samples;
    std::cout << "pryvit" << std::endl;
    readCSV("/root/c150661229a53d9c021900f2235cc3a1/ACS-project-Wormhole/decision_tree_WH/forest/data/iris_test.csv", samples);
    std::cout << "pryvit1" << std::endl;

    uint32_t n_samples = 20;
    uint32_t sample_vec_size =samples.size()* sizeof(float);
    DecisionTreeFinal tree; 
    std::cout << sample_vec_size << " pryvit2" << std::endl;

    tree.loadFromJson("/root/c150661229a53d9c021900f2235cc3a1/ACS-project-Wormhole/decision_tree_WH/forest/data/tree.json");
    std::cout << "pryvit3" << std::endl;

    std::vector<float> forest = tree.getFlatVector();
    // std::vector<float> forest(1024,1);//will be defined by function
    uint32_t n_trees = 1;
    uint32_t tree_size = forest.size();
    uint32_t forest_size = n_trees*tree_size; 
    size_t forest_bytes = forest_size * sizeof(float);

    printf("Creating device...\n");
    IDevice* device = CreateDevice(0);
    printf("Device created\n");
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    constexpr CoreCoord core = {0, 0}; 

    // Create DRAM config for forest weights
    tt_metal::InterleavedBufferConfig dram_forest_config{
        .device = device,
        .size = forest_size,
        .page_size = tree_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    // Create DRAM config for samples
    tt_metal::InterleavedBufferConfig dram_sample_config{
        .device = device,
        .size = n_samples * sample_vec_size* sizeof(float),
        .page_size = sample_vec_size* sizeof(float),
        .buffer_type = tt_metal::BufferType::DRAM};

    // Create DRAM config for results
    tt_metal::InterleavedBufferConfig dram_res_config{
        .device = device,
        .size = n_trees*n_samples*sizeof(float),
        .page_size = n_samples*sizeof(float),
        .buffer_type = tt_metal::BufferType::DRAM};
    
    //Creating Buffers
    std::shared_ptr<tt::tt_metal::Buffer> sample_dram_buffer = CreateBuffer(dram_sample_config);
    std::shared_ptr<tt::tt_metal::Buffer> forest_dram_buffer = CreateBuffer(dram_forest_config);
    std::shared_ptr<tt::tt_metal::Buffer> dram_res_buffer = CreateBuffer(dram_res_config);
    
    //Banks
    uint32_t forest_bank_id = 0;
    uint32_t sample_bank_id = 0;
    uint32_t res_bank_id = 0;

    constexpr uint32_t forest_cb_index = CBIndex::c_0;
    constexpr uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_forest_config =
        CircularBufferConfig(num_input_tiles * tree_size, {{forest_cb_index, tt::DataFormat::Float32}})
            .set_page_size(forest_cb_index, tree_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_forest_config);

    constexpr uint32_t sample_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_sample_config =
        CircularBufferConfig(num_input_tiles * n_samples*sample_vec_size, {{sample_cb_index, tt::DataFormat::Float32}})
            .set_page_size(sample_cb_index, sample_vec_size);
    CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_sample_config);

    constexpr uint32_t output_cb_index = CBIndex::c_16;
    constexpr uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles *n_trees* n_samples*sizeof(float), {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, n_samples*sizeof(float));
    CBHandle cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    // Attach data movement kernels
    auto reader_kernel = CreateKernel(program,
        "/root/c150661229a53d9c021900f2235cc3a1/ACS-project-Wormhole/decision_tree_WH/forest/kernels/dataflow/forest_reader.cpp", 
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer_kernel = CreateKernel(program, 
        "/root/c150661229a53d9c021900f2235cc3a1/ACS-project-Wormhole/decision_tree_WH/forest/kernels/dataflow/forest_writer.cpp", 
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto compute_single_core_kernel_id = tt_metal::CreateKernel(
        program,
        "/root/c150661229a53d9c021900f2235cc3a1/ACS-project-Wormhole/decision_tree_WH/forest/kernels/compute/forest_compute.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args ={tree_size,n_trees,sample_vec_size,n_samples}});

    // Set runtime args for core
    SetRuntimeArgs(program, reader_kernel, core, 
        {forest_dram_buffer->address(),sample_dram_buffer->address(), n_trees,tree_size,n_samples,sample_vec_size});
    SetRuntimeArgs(program, writer_kernel, core, 
        {dram_res_buffer->address(), n_trees,n_samples});
    
    EnqueueWriteBuffer(cq, forest_dram_buffer, forest, false);
    EnqueueWriteBuffer(cq, sample_dram_buffer, samples, false);

    // Launch program
    printf("Launching program...\n");
    EnqueueProgram(cq, program, false);
    Finish(cq);
    printf("Host: Program finished running.\n");

    // Read back result
    std::vector<float> result(n_trees*n_samples);
    EnqueueReadBuffer(cq, dram_res_buffer, result, true);

    // Print predictions of voting
    std::vector<float> final_predictions;
    for (size_t i = 0; i < n_samples; ++i){
        std::unordered_map<float, int> count;
        for (int t = 0; t < n_trees; ++t) {
            float val = result[t * n_samples + i];
            count[val]++;
        }
        float most_common_val = 0.0;
        int max_freq = 0;
        for (const auto& pair : count) {
            if (pair.second > max_freq) {
                max_freq = pair.second;
                most_common_val = pair.first;
            }
        }

        final_predictions.push_back(most_common_val);
    }
    std::cout << "Final predictions: ";
    for (float val : final_predictions) {
        std::cout << val << " ";
    }
    std::cout << "\n";
    CloseDevice(device);
    return 0;
}




// int main() {
    // std::vector<float> forest(1024,1);
    // uint32_t forest_size = forest.size();
    // uint32_t tree_size = 64;
    // uint32_t n_trees = 16
    // printf("Creating device...\n");
    // IDevice* device = CreateDevice(0);
    // printf("Device created\n");
    // CommandQueue& cq = device->command_queue();
    // Program program = CreateProgram();

    // CoreCoord start_core = {0, 0};
    // CoreCoord end_core = {0, 7};
    // CoreRange cores(start_core, end_core);
    // uint32_t num_cores = cores.size();

    // size_t forest_bytes = forest_size * sizeof(float);
    // auto forest_buffer = CreateBuffer(InterleavedBufferConfig{
    //     .device = device,
    //     .size = forest_bytes,
    //     .page_size = tree_size*sizeof(float),
    //     .buffer_type = BufferType::DRAM
    // });
    // auto reader_kernel = CreateKernel(program, "/root/c150661229a53d9c021900f2235cc3a1/ACS-project-Wormhole/decision_tree_WH/forest/kernels/dataflow/forest_reader.cpp", cores,
    //     DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    // auto writer_kernel = CreateKernel(program, "/root/c150661229a53d9c021900f2235cc3a1/ACS-project-Wormhole/decision_tree_WH/forest/kernels/dataflow/forest_writer.cpp", cores,
    //     DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    // // Create DRAM buffer for result
    // auto result_buffer = CreateBuffer(InterleavedBufferConfig{
    //     .device = device,
    //     .size = forest_bytes,
    //     .page_size = tree_size*sizeof(float),
    //     .buffer_type = BufferType::DRAM
    // });
    // uint32_t forest_bank_id = 0;
    // uint32_t sample_bank_id = 0;
    // uint32_t res_bank_id = 0;
    // // Write forest data to DRAM
    // EnqueueWriteBuffer(cq, forest_buffer, forest, false);

    // // Create circular buffer in L1
    // constexpr uint32_t cb_index = CBIndex::c_0;
    // auto cb = CreateCircularBuffer(program, core,
    //     CircularBufferConfig(forest_size, {{cb_index, DataFormat::Float32}})
    //     .set_page_size(cb_index, forest_size));
    // tree_size_converted=tree_size*sizeof(float)
    // uint32_t tree_offset =0;
    // for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
    //     CoreCoord core = {0, core_idx};
    //     SetRuntimeArgs(program, reader_kernel, core, {forest_buffer->address(), forest_size,tree_offset,n_trees,tree_size_converted});
    //     SetRuntimeArgs(program, writer_kernel, core, {result_buffer->address(), forest_size,tree_offset,n_trees,tree_size_converted});
    //     tree_offset+=1;
    //     }
    // printf("Launching program...\n");
    // EnqueueProgram(cq, program, false);
    // Finish(cq);
    // printf("Host: Program finished running.\n");
    // std::vector<float> result(forest_size);
    // EnqueueReadBuffer(cq, result_buffer, result, true);

    // // Print round-trip values
    // for (size_t i = 0; i < forest_size; ++i)
    //     printf("Result[%zu] = %.2f\n", i, result[i]);

    // CloseDevice(device);
    // return 0;
// }