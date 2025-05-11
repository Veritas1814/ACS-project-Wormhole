#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"


namespace DecisionTreeKernel {
    constexpr auto cb_tree   = tt::CBIndex::c_0;  
    constexpr auto cb_sample = tt::CBIndex::c_1;   
    constexpr auto cb_out    = tt::CBIndex::c_16;

    void MAIN {
        uint32_t tree_size = get_arg_val<uint32_t>(0);
        uint32_t n_trees = get_arg_val<uint32_t>(1);
        uint32_t sample_vec_size = get_arg_val<uint32_t>(2);
        uint32_t n_samples = get_arg_val<uint32_t>(3);

        cb_wait_front(cb_tree,   3 * n_trees);
        cb_wait_front(cb_sample, n_samples * sample_vec_size);

        cb_reserve_back(cb_out, n_samples);

        const float* tree_data = reinterpret_cast<const float*>(get_read_ptr(cb_tree));
        const float* feat_arr  = tree_data;
        const float* th_arr    = tree_data + n_trees;
        const float* val_arr   = tree_data + 2 * n_trees;

        for (uint32_t s = 0; s < n_samples; ++s) {
            const float* sample = reinterpret_cast<const float*>(get_read_ptr(cb_sample));

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
            int32_t* out_ptr = reinterpret_cast<int32_t*>(get_write_ptr(cb_out));
            *out_ptr = cls;

            cb_push_back(cb_out, 1);
            cb_pop_front(cb_sample, 4);
        }

        cb_pop_front(cb_tree, 3 * n_trees);
    }
}