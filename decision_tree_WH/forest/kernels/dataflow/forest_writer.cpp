#include "dataflow_api.h"
#include "debug/dprint.h"
void kernel_main() {
    uint32_t result_adr = get_arg_val<uint32_t>(0);
    uint32_t n_trees = get_arg_val<uint32_t>(1);
    uint32_t n_samples = get_arg_val<uint32_t>(2);
    uint32_t res_bank_id = 0;

    constexpr uint32_t cb_id_out = tt::CBIndex::c_4;
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out);
    DPRINT_DATA1(DPRINT << "Hello, Master, I am running in writer1." << ENDL());
    uint32_t result_offset = result_adr;
    for (uint32_t i=0; i<n_samples; i++){
        uint64_t res_noc_addr = get_noc_addr_from_bank_id<true>(res_bank_id, result_offset);
        cb_wait_front(cb_id_out, 1);
        noc_async_write(l1_read_addr, res_noc_addr, ublock_size_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, 1);
        result_offset+=sizeof(float);
    }
    DPRINT_DATA0(DPRINT << "Hello, Master, I am running in writer0." << ENDL());
}
