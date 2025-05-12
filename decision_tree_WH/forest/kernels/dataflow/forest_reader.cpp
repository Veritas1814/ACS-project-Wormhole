#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Args: DRAM address (byte), number of floats
    DPRINT_DATA0(DPRINT << "Hello, Master, I am running a void data movement kernel on NOC 0." << ENDL());
    uint32_t forest_dram_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t sample_dram_buffer_addr = get_arg_val<uint32_t>(1);
    uint32_t n_trees = get_arg_val<uint32_t>(2);
    uint32_t tree_size = get_arg_val<uint32_t>(3);
    uint32_t n_samples = get_arg_val<uint32_t>(4);
    uint32_t sample_vec_size = get_arg_val<uint32_t>(4);
    constexpr uint32_t forest_bank_id = 0; 
    constexpr uint32_t sample_bank_id = 0;     
    constexpr uint32_t forest_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t sample_cb_index = CBIndex::c_1;
    // Get tile size in bytes (1 float per ublock = 4 bytes if FP32)
    uint32_t ublock_bytes_forest = get_tile_size(forest_cb_index);  // usually 4 for float32
    uint32_t l1_write_addr_forest = get_write_ptr(forest_cb_index);

    uint32_t ublock_bytes_saples = get_tile_size(sample_cb_index);  
    uint32_t l1_write_addr_samples = get_write_ptr(sample_cb_index)
    
    DPRINT_DATA0(DPRINT << "Hello, Master, I am running a void data movement kernel on NOC 0." << ENDL());
    uint32_t dram_offset =forest_dram_buffer_addr;
    uint32_t sample_offset = sample_dram_buffer_addr;
    for (uint32_t i=0; i<n_trees; i++){
        uint64_t forest_noc_addr = get_noc_addr_from_bank_id<true>(forest_bank_id, dram_offset);
        cb_reserve_back(forest_cb_index, 1);
        noc_async_read(forest_noc_addr, l1_write_addr_forest, tree_size);
        noc_async_read_barrier();
        cb_push_back(forest_cb_index, 1); 
        dram_offset += tree_size;
        for (uint32_t j=0; j<n_samples;j++){
            uint64_t sample_noc_addr = get_noc_addr_from_bank_id<true>(sample_bank_id, sample_offset);
            cb_reserve_back(forest_cb_index, 1);
            noc_async_read(forest_noc_addr, l1_write_addr_samples, sample_vec_size);
            noc_async_read_barrier();
            cb_push_back(forest_cb_index, 1); 
            sample_offset+=sample_vec_size 
        }
    }

    DPRINT_DATA0(DPRINT << "Hello, Master, I am running a void data movement kernel on NOC 0." << ENDL());
}