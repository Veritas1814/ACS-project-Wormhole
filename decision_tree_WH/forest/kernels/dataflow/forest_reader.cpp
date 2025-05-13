#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Args: DRAM address (byte), number of floats
    DPRINT_DATA0(DPRINT << "Hello, Master, I am running a void data movement kernel on NOC 0." << ENDL());
    uint32_t feature_dram_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t value_dram_buffer_addr = get_arg_val<uint32_t>(1);
    uint32_t threshold_dram_buffer_addr = get_arg_val<uint32_t>(2);
    uint32_t sample_dram_buffer_addr = get_arg_val<uint32_t>(3);

    uint32_t n_trees = get_arg_val<uint32_t>(4);
    uint32_t features_size = get_arg_val<uint32_t>(5);
    uint32_t values_size = get_arg_val<uint32_t>(6);
    uint32_t thresholds_size = get_arg_val<uint32_t>(7);

    // uint32_t features_size1 = get_arg_val<uint32_t>(8);
    // uint32_t values_size1 = get_arg_val<uint32_t>(9);
    // uint32_t thresholds_size1 = get_arg_val<uint32_t>(10);

    uint32_t n_samples = get_arg_val<uint32_t>(11);
    uint32_t sample_vec_size = get_arg_val<uint32_t>(12);

    uint32_t feature_bank_id = get_arg_val<uint32_t>(13); 
    uint32_t value_bank_id = get_arg_val<uint32_t>(14);
    uint32_t threshold_bank_id = get_arg_val<uint32_t>(15);
    uint32_t sample_bank_id = get_arg_val<uint32_t>(16);

    constexpr uint32_t feature_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t value_cb_index = tt::CBIndex::c_1;
    constexpr uint32_t threshold_cb_index = tt::CBIndex::c_2;
    constexpr uint32_t sample_cb_index = tt::CBIndex::c_3;

    // Get tile size in bytes (1 float per ublock = 4 bytes if FP32)
    uint32_t ublock_bytes_features = get_tile_size(feature_cb_index);  // usually 4 for float32
    uint32_t l1_write_addr_features = get_write_ptr(feature_cb_index);

    uint32_t ublock_bytes_values = get_tile_size(value_cb_index);  // usually 4 for float32
    uint32_t l1_write_addr_values = get_write_ptr(value_cb_index);

    uint32_t ublock_bytes_treshold = get_tile_size(threshold_cb_index);  // usually 4 for float32
    uint32_t l1_write_addr_treshold = get_write_ptr(threshold_cb_index);

    uint32_t ublock_bytes_saples = get_tile_size(sample_cb_index);  
    uint32_t l1_write_addr_samples = get_write_ptr(sample_cb_index);
    
    DPRINT_DATA1(DPRINT << "Hello, Master, I am running in reade1." << ENDL());
    uint32_t feature_offset =feature_dram_buffer_addr;
    uint32_t value_offset =value_dram_buffer_addr;
    uint32_t treshold_offset =threshold_dram_buffer_addr;
    uint32_t sample_offset = sample_dram_buffer_addr;
    for (uint32_t i=0; i<n_trees; i++){
        uint64_t feature_noc_addr = get_noc_addr_from_bank_id<true>(feature_bank_id, feature_offset);
        uint64_t value_noc_addr = get_noc_addr_from_bank_id<true>(value_bank_id, value_offset);
        uint64_t treshold_noc_addr = get_noc_addr_from_bank_id<true>(threshold_bank_id, treshold_offset);

        cb_reserve_back(feature_cb_index, 1);
        noc_async_read(feature_noc_addr, l1_write_addr_features, ublock_bytes_features);
        noc_async_read_barrier();
        cb_push_back(feature_cb_index, 1); 
        feature_offset += ublock_bytes_features;

        cb_reserve_back(value_cb_index, 1);
        noc_async_read(value_noc_addr, l1_write_addr_values, ublock_bytes_values);
        noc_async_read_barrier();
        cb_push_back(value_cb_index, 1);
        value_offset += ublock_bytes_values;

        cb_reserve_back(threshold_cb_index, 1);
        noc_async_read(treshold_noc_addr, l1_write_addr_treshold, ublock_bytes_treshold);
        noc_async_read_barrier();
        cb_push_back(threshold_cb_index, 1);
        treshold_offset += ublock_bytes_treshold;

        for (uint32_t j=0; j<n_samples;j++){
            uint64_t sample_noc_addr = get_noc_addr_from_bank_id<true>(sample_bank_id, sample_offset);
            cb_reserve_back(sample_cb_index, 1);
            noc_async_read(sample_noc_addr, l1_write_addr_samples, ublock_bytes_saples);
            noc_async_read_barrier();
            cb_push_back(sample_cb_index, 1); 
            sample_offset+=ublock_bytes_saples;
        }
    }

    DPRINT_DATA0(DPRINT << "Hello, Master, I am running a void data movement kernel on NOC 0." << ENDL());
}