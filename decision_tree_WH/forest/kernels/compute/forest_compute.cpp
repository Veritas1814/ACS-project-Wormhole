
// #include <cstdint>
// #include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api.h"

using std::uint32_t;
namespace NAMESPACE {
    void MAIN {
        constexpr auto cb_tree   = tt::CBIndex::c_0;  
        constexpr auto cb_sample = tt::CBIndex::c_1;   
        constexpr auto cb_out    = tt::CBIndex::c_5;
        
        uint32_t tree_size = get_compile_time_arg_val(0);
        uint32_t n_trees = get_compile_time_arg_val(1);
        uint32_t sample_vec_size = get_compile_time_arg_val(2);
        uint32_t n_samples = get_compile_time_arg_val(3);
        
        cb_wait_front(cb_tree,   3 * n_trees);
        cb_wait_front(cb_sample, n_samples * sample_vec_size);
        cb_reserve_back(cb_out, n_samples);

        cb_wait_front(cb_tree, 1);
        volatile uint32_t* cb_tree_addr;
        cb_get_tile(cb_tree, 0, &cb_tree_addr);


        volatile tt_l1_ptr float* tree = reinterpret_cast<volatile tt_l1_ptr float*>(cb_tree_addr);        
        volatile float* feat_arr  = tree;
        volatile float* th_arr    = tree + n_trees;
        volatile float* val_arr   = tree + 2 * n_trees;

        for (uint32_t s = 0; s < n_samples; ++s) {
            cb_wait_front(cb_sample, 1);
            volatile uint32_t* cb_sample_addr;
            // cb_get_tile(cb_sample, 0, &cb_sample_addr);
            volatile float* sample = reinterpret_cast<volatile float*>(cb_sample_addr);
            
            int32_t node = 0;
            while (true) {
                int32_t feat = static_cast<int32_t>(feat_arr[node]);
                if (feat < 0) 
                    break;
                float x  = sample[feat];
                float th = th_arr[node];
                node = (x >= th) ? (2 * node + 2)
                                    : (2 * node + 1);
            }
            int32_t cls = static_cast<int32_t>(val_arr[node]);
            cb_wait_front(cb_out, 1);
            volatile uint32_t* cb_out_addr;
            // cb_get_tile(cb_out, 0, &cb_out_addr);
            volatile int32_t* out_ptr = reinterpret_cast<volatile int32_t*>(cb_out_addr);
            *out_ptr = cls;

            cb_push_back(cb_out, 1);
            cb_pop_front(cb_sample, 4);
        }
        // cb_release_tile(cb_tree);
        // cb_release_tile(cb_sample);
        // cb_release_tile(cb_out);

        cb_pop_front(cb_tree, 3 * n_trees);
    }
}