/*******************************************************************************
* Copyright 2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_SYCL_SYCL_PRIMITIVE_CONF_HPP
#define GPU_SYCL_SYCL_PRIMITIVE_CONF_HPP

#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct sycl_binary_conf_t {
    sycl_md_t src0_md;
    sycl_md_t src1_md;
    sycl_md_t dst_md;

    alg_kind_t alg_kind;
    bool do_scale_src0;
    bool do_scale_src1;
    int broadcast_dims[sycl_md_t::max_dims];
    int ndims;
    bool is_tensor_op;

    int block_size;
    int wg_size;

    sycl_post_ops_t post_ops;
};

struct sycl_layer_normalization_conf_t {
    prop_kind_t prop_kind;
    sycl_md_t data_md;
    sycl_md_t diff_data_md;
    sycl_md_t data_scaleshift_md;
    sycl_md_t diff_data_scaleshift_md;
    sycl_md_t scale;
    sycl_md_t shift;
    sycl_md_t stat_md;
    sycl_md_t stat_d;
    sycl_md_t var_md;
    sycl_md_t dst_md;
    sycl_md_t diff_dst_md;
    dim_t wk_size;
    bool is_fwd;
    bool src_def;
    bool dst_def;
    size_t n_thr;
    size_t n_thr2;
    size_t diff_shift_off;
    bool rt_scaling = false;
    int scale_count;
    float oscale = 1.0f;
    dim_t N;
    dim_t C;
    bool use_ss;
    bool use_scale;
    bool use_shift;
    dim_t wei_shift_off;
    bool calculate_stats;
    bool calculate_diff_stats;
    bool save_stats;
    int shift_off;
    bool zero_dims;
    int ss_off;
    float layer_norm_epsilon;
    unsigned flags;
    int ndims;
    int block_size;
    int wg_size;
};
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_binary_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_layer_normalization_conf_t);

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
