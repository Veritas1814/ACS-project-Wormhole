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
    uint32_t sample_vec_size =samples.size() / n_samples;
    DecisionTreeFinal tree; 
    std::cout << sample_vec_size << " pryvit2" << std::endl;

    tree.loadFromJson("/root/c150661229a53d9c021900f2235cc3a1/ACS-project-Wormhole/decision_tree_WH/forest/data/tree.json");
    std::cout << "pryvit3" << std::endl;

    std::vector<float> forest = tree.getFlatVector();
    uint32_t n_trees = 1;
    uint32_t tree_size = forest.size();

    // size_t forest_bytes = forest_size ;
    std::vector<float> features = tree.getFeatures();
    std::vector<float> values = tree.getValues();
    std::vector<float> threshold = tree.getTreshold();
    uint32_t feature_size1 =features.size();
    uint32_t value_size1 =values.size();
    uint32_t threshold_size1 =threshold.size();

    uint32_t features_size = n_trees*feature_size1; 
    uint32_t values_size = n_trees*value_size1; 
    uint32_t thresholds_size = n_trees*threshold_size1;

    printf("Creating device...\n");
    IDevice* device = CreateDevice(0);
    printf("Device created\n");
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    constexpr CoreCoord core = {0, 0}; 

    // Create DRAM config for features
    tt_metal::InterleavedBufferConfig dram_feature_config{
        .device = device,
        .size = features_size,
        .page_size = feature_size1,
        .buffer_type = tt_metal::BufferType::DRAM};
    // Create DRAM config for values
    tt_metal::InterleavedBufferConfig dram_value_config{
        .device = device,
        .size = values_size,
        .page_size = value_size1,
        .buffer_type = tt_metal::BufferType::DRAM};
    // Create DRAM config for threshold
    tt_metal::InterleavedBufferConfig dram_threshold_config{
        .device = device,
        .size = thresholds_size,
        .page_size = threshold_size1,
        .buffer_type = tt_metal::BufferType::DRAM};

    // Create DRAM config for samples
    tt_metal::InterleavedBufferConfig dram_sample_config{
        .device = device,
        .size = n_samples * sample_vec_size,
        .page_size = sample_vec_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    // Create DRAM config for results
    tt_metal::InterleavedBufferConfig dram_res_config{
        .device = device,
        .size = n_trees*n_samples,
        .page_size = n_samples,
        .buffer_type = tt_metal::BufferType::DRAM};
    
    //Creating Buffers
    std::shared_ptr<tt::tt_metal::Buffer> sample_dram_buffer = CreateBuffer(dram_sample_config);
    std::shared_ptr<tt::tt_metal::Buffer> feature_dram_buffer = CreateBuffer(dram_feature_config);
    std::shared_ptr<tt::tt_metal::Buffer> value_dram_buffer = CreateBuffer(dram_value_config);
    std::shared_ptr<tt::tt_metal::Buffer> threshold_dram_buffer = CreateBuffer(dram_threshold_config);
    std::shared_ptr<tt::tt_metal::Buffer> dram_res_buffer = CreateBuffer(dram_res_config);
    
    //Banks
    uint32_t feature_bank_id = 0;
    uint32_t value_bank_id = 0;
    uint32_t threshold_bank_id = 0;
    uint32_t sample_bank_id = 0;
    uint32_t res_bank_id = 0;

    constexpr uint32_t feature_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t value_cb_index = tt::CBIndex::c_1;
    constexpr uint32_t threshold_cb_index = tt::CBIndex::c_2;
    constexpr uint32_t sample_cb_index = tt::CBIndex::c_3;
    constexpr uint32_t output_cb_index = tt::CBIndex::c_4;

    constexpr uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_feature_config =
        CircularBufferConfig(num_input_tiles * features_size, {{feature_cb_index, tt::DataFormat::Float32}})
            .set_page_size(feature_cb_index, feature_size1);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_feature_config);

     CircularBufferConfig cb_value_config =
        CircularBufferConfig(num_input_tiles * values_size, {{value_cb_index, tt::DataFormat::Float32}})
            .set_page_size(value_cb_index, value_size1);
    CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_value_config);

     CircularBufferConfig cb_threshold_config =
        CircularBufferConfig(num_input_tiles * thresholds_size, {{threshold_cb_index, tt::DataFormat::Float32}})
            .set_page_size(threshold_cb_index, threshold_size1);
    CBHandle cb_src2 = tt_metal::CreateCircularBuffer(program, core, cb_threshold_config);
    
    CircularBufferConfig cb_sample_config =
        CircularBufferConfig(num_input_tiles * n_samples*sample_vec_size, {{sample_cb_index, tt::DataFormat::Float32}})
            .set_page_size(sample_cb_index, sample_vec_size);
    CBHandle cb_src3 = tt_metal::CreateCircularBuffer(program, core, cb_sample_config);

    constexpr uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles *n_trees* n_samples, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, n_samples);
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
        tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args ={n_trees,features_size,values_size,thresholds_size,feature_size1,value_size1,threshold_size1,n_samples,sample_vec_size}});

    // Set runtime args for core
    SetRuntimeArgs(program, reader_kernel, core, 
        {feature_dram_buffer->address(),value_dram_buffer->address(),threshold_dram_buffer->address(),sample_dram_buffer->address(), n_trees,features_size,values_size,thresholds_size,feature_size1,value_size1,threshold_size1,n_samples,sample_vec_size, feature_bank_id, value_bank_id, threshold_bank_id, sample_bank_id});
    SetRuntimeArgs(program, writer_kernel, core, 
        {dram_res_buffer->address(), n_trees,n_samples});
    
    EnqueueWriteBuffer(cq, feature_dram_buffer, features, false);
    EnqueueWriteBuffer(cq, value_dram_buffer, values, false);
    EnqueueWriteBuffer(cq, threshold_dram_buffer, threshold, false);
    EnqueueWriteBuffer(cq, sample_dram_buffer, samples, false);

    // Launch program
    printf("Launching program...\n");
    EnqueueProgram(cq, program, false);
    Finish(cq);
    printf("Host: Program finished running.\n");

    // Read back result
    std::vector<float> result(n_trees*n_samples);
    EnqueueReadBuffer(cq, dram_res_buffer, result, true);

    // // Print predictions of voting
    // std::vector<float> final_predictions;
    // for (size_t i = 0; i < n_samples; ++i){
    //     std::unordered_map<float, int> count;
    //     for (int t = 0; t < n_trees; ++t) {
    //         float val = result[t * n_samples + i];
    //         count[val]++;
    //     }
    //     float most_common_val = 0.0;
    //     int max_freq = 0;
    //     for (const auto& pair : count) {
    //         if (pair.second > max_freq) {
    //             max_freq = pair.second;
    //             most_common_val = pair.first;
    //         }
    //     }

    //     final_predictions.push_back(most_common_val);
    // }
    std::cout << "Final predictions: ";
    // for (float val : final_predictions) {
    //     std::cout << val << " ";
    // }
    // std::cout << "\n";
    CloseDevice(device);  
    return 0;
}
