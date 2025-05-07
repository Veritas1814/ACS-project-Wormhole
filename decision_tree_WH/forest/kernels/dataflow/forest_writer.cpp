// #include <stdint.h>
// #include <assert.h>
// #include "dataflow_api.h"


// // void forest_writer() {
// //     uint32_t output_base_addr = get_arg_val<uint32_t>(0);
// //     uint32_t tree_size        = get_arg_val<uint32_t>(1);
// //     uint32_t num_trees        = get_arg_val<uint32_t>(2);
// //     assert(num_trees > 0);
// //     constexpr uint32_t CB_OUTPUT = tt::CBIndex::c_16;
// //     assert(tree_size <= get_tile_size(CB_OUTPUT));
// //     uint32_t l1_read_ptr = get_read_ptr(CB_OUTPUT);
// //     constexpr uint32_t NUM_BANKS = tt::DRAM_BANK_COUNT;
// //     uint32_t bank_id = 0;

// //     for (uint32_t idx = 0; idx < num_trees; idx += 2) {
// //         uint32_t count = (idx + 2 <= num_trees ? 2 : 1);
// //         cb_reserve_front(CB_OUTPUT, count);
// //         for (uint32_t j = 0; j < count; ++j) {
// //             uint32_t offset = output_base_addr + (idx + j) * tree_size;
// //             uint64_t addr   = get_noc_addr_from_bank_id<true>((bank_id + j) % NUM_BANKS, offset);
// //             noc_async_read(l1_read_ptr + j * tree_size, addr, tree_size);
// //         }
// //         noc_async_read_barrier();
// //         cb_push_front(CB_OUTPUT, count);
// //         bank_id = (bank_id + count) % NUM_BANKS;
// //     }
// // }

// void kernel_main() {
//     uint32_t output_base_addr = get_arg_val<uint32_t>(0);
//     uint32_t tree_size        = get_arg_val<uint32_t>(1);
//     uint32_t num_trees        = get_arg_val<uint32_t>(2);
//     constexpr uint32_t CB_OUTPUT = tt::CBIndex::c_16;
//     constexpr uint32_t NUM_BANKS = 8;

//     uint32_t l1_read_ptr = get_read_ptr(CB_OUTPUT);
//     uint32_t bank_id = 0;

//     for (uint32_t idx = 0; idx < num_trees; idx++) {
//         cb_wait_front(CB_OUTPUT, 1);

//         uint32_t offset = output_base_addr + idx * tree_size;
//         uint64_t addr   = get_noc_addr_from_bank_id<true>(bank_id, offset);

//         noc_async_write(l1_read_ptr, addr, tree_size);
//         noc_async_write_barrier();
//         cb_pop_front(CB_OUTPUT, 1);

//         bank_id = (bank_id + 1) % NUM_BANKS;
//     }
// }
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_floats = get_arg_val<uint32_t>(2);  // new: how many floats to write

    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(dst_bank_id, dst_addr);

    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out);

    for (uint32_t i = 0; i < num_floats; ++i) {
        cb_wait_front(cb_id_out, 1);
        noc_async_write(l1_read_addr, dst_noc_addr + i * ublock_size_bytes, ublock_size_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, 1);
    }
}
