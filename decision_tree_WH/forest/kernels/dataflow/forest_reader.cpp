// // SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
// //
// // SPDX-License-Identifier: Apache-2.0

// #include <stdint.h>
// #include "dataflow_api.h"

// // void kernel_main() {

// //     uint32_t forest_dram_buffer_addr = get_arg_val<uint32_t>(0);
// //     uint32_t tree_size = get_arg_val<uint32_t>(1);
// //     uint32_t n_trees = get_arg_val<uint32_t>(2);

// //     const InterleavedAddrGen<src_is_dram> s0 = {.bank_base_address = forest_dram_buffer_addr, .page_size = data_size_bytes};
// //     // const InterleavedAddrGen<pad_is_dram> s1 = {.bank_base_address = pad_addr, .page_size = data_size_bytes};
// //     // uint64_t src1_noc_addr = get_noc_addr_from_bank_id<true>(src1_bank_id, src1_addr);

// //     constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
// //     // constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

// //     // single-tile ublocks
// //     // uint32_t ublock_size_bytes_0 = get_tile_size(cb_id_in0);
// //     // uint32_t ublock_size_bytes_1 = get_tile_size(cb_id_in1);

// //     uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
// //     // uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
// //     uint32_t dram_offset =forest_dram_buffer_addr;
// //     for (uint32_t i=0; i<n_trees; i++){
// //         uint64_t forest_noc_addr = get_noc_addr_from_bank_id<true>(forest_bank_id, dram_offset);
// //         cb_reserve_back(cb_id_in0, 1);
// //         noc_async_read(forest_noc_addr, l1_write_addr_in0, tree_size);
// //         noc_async_read_barrier();
// //         cb_push_back(cb_id_in0, 1); 
// //         dram_offset += tree_size;
// //     }
// //     // cb_reserve_back(cb_id_in1, 1);
// //     // noc_async_read(src1_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);
// //     // noc_async_read_barrier();
// //     // cb_push_back(cb_id_in1, 1);
// // }

// void kernel_main() {
//     // === Load arguments ===
//     uint32_t forest_dram_buffer_addr = get_arg_val<uint32_t>(0);  // Base address of forest in DRAM
//     uint32_t tree_size = get_arg_val<uint32_t>(1);                // Size of each tree in bytes
//     uint32_t n_trees = get_arg_val<uint32_t>(2);                  // Number of trees to load

//     // === Configuration ===
//     constexpr uint32_t forest_bank_id = 0;         // Set your actual DRAM bank ID here
//     constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;

//     // === Get local L1 address to write to ===
//     uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

//     // === Load each tree from DRAM ===
//     for (uint32_t i = 0; i < n_trees; i++) {
//         uint32_t dram_offset = forest_dram_buffer_addr + i * tree_size;
//         uint64_t forest_noc_addr = get_noc_addr_from_bank_id<true>(forest_bank_id, dram_offset);

//         cb_reserve_back(cb_id_in0, 1);
//         noc_async_read(forest_noc_addr, l1_write_addr, tree_size);
//         noc_async_read_barrier();
//         cb_push_back(cb_id_in0, 1);
//     }
// }

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Args: DRAM address (byte), number of floats
    DPRINT_DATA0(DPRINT << "Hello, Master, I am running a void data movement kernel on NOC 0." << ENDL());

    uint32_t forest_dram_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t tree_size = get_arg_val<uint32_t>(1);
    uint32_t bank_id = 0;
    constexpr uint32_t cb_id = tt::CBIndex::c_0;

    // Get tile size in bytes (1 float per ublock = 4 bytes if FP32)
    uint32_t ublock_bytes = get_tile_size(cb_id);  // usually 4 for float32
    uint32_t l1_write_addr = get_write_ptr(cb_id);

    uint64_t noc_addr = get_noc_addr_from_bank_id<true>(bank_id,forest_dram_buffer_addr);
    DPRINT_DATA0(DPRINT << "Hello, Master, I am running a void data movement kernel on NOC 0." << ENDL());
    cb_reserve_back(cb_id, 1); 
    noc_async_read(noc_addr, l1_write_addr, tree_size);
    noc_async_read_barrier();
    cb_push_back(cb_id, 1);
    DPRINT_DATA0(DPRINT << "Hello, Master, I am running a void data movement kernel on NOC 0." << ENDL());
}