
// #include <cstdint>
// #include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api.h"
#include "debug/dprint_tensix.h"

using std::uint32_t;
namespace NAMESPACE {
    void MAIN {
        
        constexpr uint32_t feature_cb_index = tt::CBIndex::c_0;
        constexpr uint32_t value_cb_index = tt::CBIndex::c_1;
        constexpr uint32_t threshold_cb_index = tt::CBIndex::c_2;
        constexpr uint32_t sample_cb_index = tt::CBIndex::c_3;  
        constexpr auto cb_out    = tt::CBIndex::c_4;
        
        uint32_t n_trees = get_compile_time_arg_val(0);
        uint32_t features_size = get_compile_time_arg_val(1);
        uint32_t values_size = get_compile_time_arg_val(2);
        uint32_t thresholds_size = get_compile_time_arg_val(3);
        uint32_t feature_size1 = get_compile_time_arg_val(4);
        uint32_t value_size1 = get_compile_time_arg_val(5);
        uint32_t threshold_size1 = get_compile_time_arg_val(6);
        uint32_t n_samples = get_compile_time_arg_val(7);
        uint32_t sample_vec_size = get_compile_time_arg_val(8);

        // cb_reserve_back(cb_out, n_samples);

        for (uint32_t s = 0; s < n_samples; ++s) {
            cb_wait_front(sample_cb_index, 1);
            cb_wait_front(feature_cb_index, 1);
            cb_wait_front(value_cb_index, 1);
            cb_wait_front(threshold_cb_index, 1);

            volatile uint32_t* cb_sample_addr;
            cb_get_tile(sample_cb_index, 0, &cb_sample_addr);
            volatile uint32_t* cb_feature_addr;
            cb_get_tile(feature_cb_index, 0, &cb_feature_addr);
            volatile uint32_t* cb_value_addr;
            cb_get_tile(value_cb_index, 0, &cb_value_addr);
            volatile uint32_t* cb_treshold_addr;
            cb_get_tile(threshold_cb_index, 0, &cb_treshold_addr);

            int32_t node = 0;
            while (true) {
                int32_t feat = static_cast<int32_t>(cb_feature_addr[node]);
                if (feat < 0)
                    break;
                float x  = cb_sample_addr[feat];
                float th = cb_treshold_addr[node];
                node = (x >= th) ? (2 * node + 2)
                                : (2 * node + 1);
            }

            int32_t cls = static_cast<int32_t>(cb_value_addr[node]);

            cb_wait_front(cb_out, 1);
            volatile uint32_t* cb_out_addr;
            cb_get_tile(cb_out, 0, &cb_out_addr);
            *cb_out_addr = cls;
            cb_release_tile(cb_out);
            cb_push_back(cb_out, 1);

            cb_pop_front(sample_cb_index, 1);
            cb_pop_front(feature_cb_index, 1);
            cb_pop_front(value_cb_index, 1);
            cb_pop_front(threshold_cb_index, 1);

            cb_release_tile(sample_cb_index);
            cb_release_tile(feature_cb_index);
            cb_release_tile(value_cb_index);
            cb_release_tile(threshold_cb_index);
        }

        
    }
}