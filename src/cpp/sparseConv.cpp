#include "sparseConv.h"

/**
 * calculate after length of 1-D convolution, given some configs.
 */
inline int conv_length(int length, int kernel, int padding, int stride, int dilation = 1) {
    return (length + 2*padding - (dilation * (kernel - 1) + 1)) / stride + 1;
}

/**
 * support both LongInt / Float / double type
 * Refer:
 * https://github.com/pytorch/pytorch/blob/v1.11.0/aten/src/ATen/native/Im2Col.cpp#L14
 * https://github.com/pytorch/pytorch/blob/v1.11.0/aten/src/ATen/native/im2col.h#L14
 * @param input 
 *  - @param (channel_last=false) -> [batch, channel, height, width]
 *  - @param (channel_last=true) -> [batch, height, width, channel]
 */
template<typename T>
torch::Tensor unfold_template(
    const torch::Tensor& input, 
    size_hw kernel, 
    size_hw padding, 
    size_hw stride, 
    size_hw dilation, 
    bool channel_last=false
) {
    int batch_size = input.size(0);
    int in_channels = channel_last ? input.size(3) : input.size(1);
    int in_height = channel_last ? input.size(1) : input.size(2);
    int in_width = channel_last ? input.size(2) : input.size(3);
    int out_height = conv_length(in_height, kernel.h, padding.h, stride.h, dilation.h);
    int out_width = conv_length(in_width, kernel.w, padding.w, stride.w, dilation.w);
    int out_channels = in_channels * kernel.h * kernel.w;

    vector<int64_t> out_shape = channel_last ? 
        { batch_size, out_height * out_width, out_channels } :
        { batch_size, out_channels, out_height * out_width };
    auto output = torch::zeros(out_shape, input.options());
    
    // make sure contiguous, then use pointer
    auto in_ptr = data_ptr<T>(input);
    auto out_ptr = data_ptr<T>(output);

    if (channel_last) { // todo...
        parallel_for(batch_size, true, [&](auto b_i) {
            for (const auto h_i : c10::irange(out_height)) {
                int64_t h_in = h_i * stride.h - padding.h + h_offset * dilation.h;
                
                for (const auto w_i : c10::irange(out_width)) {
                    int64_t w_in = w_i * stride.w - padding.w + w_offset * dilation.w;

                    for (const auto col_i : c10::irange(out_channels)) {
                        auto w_offset = col_i % kernel.w;
                        auto h_offset = (col_i / kernel.w) % kernel.h;
                        auto c_in = col_i / kernel.h / kernel.w;

                        out_ptr[((b_i * out_height + h_i) * out_width + w_i) * out_channels + col_i] =
                            (h_in >= 0 && w_in >= 0 && h_in < in_height && w_in < in_width)
                            ? in_ptr[((b_i * in_channels + c_in) * in_height + h_in) * in_width + w_in]
                            : static_cast<T>(0);
                    }
                }
            }
        });
    }
    else {
        parallel_for(batch_size, true, [&](auto b_i) {
            for (const auto col_i : c10::irange(out_channels)) {
                auto w_offset = col_i % kernel.w;
                auto h_offset = (col_i / kernel.w) % kernel.h;
                auto c_in = col_i / kernel.h / kernel.w;

                for (const auto h_i : c10::irange(out_height)) {
                    int64_t h_in = h_i * stride.h - padding.h + h_offset * dilation.h;
                    
                    for (const auto w_i : c10::irange(out_width)) {
                        int64_t w_in = w_i * stride.w - padding.w + w_offset * dilation.w;

                        out_ptr[((b_i * out_channels + col_i) * out_height + h_i) * out_width + w_i] =
                            (h_in >= 0 && w_in >= 0 && h_in < in_height && w_in < in_width)
                            ? in_ptr[((b_i * in_channels + c_in) * in_height + h_in) * in_width + w_in]
                            : static_cast<T>(0);
                    }
                }
            }
        });
    }

    return output;
}

template
torch::Tensor unfold_template<float>(const torch::Tensor& input, size_hw kernel, size_hw padding, size_hw stride, size_hw dilation);
template
torch::Tensor unfold_template<double>(const torch::Tensor& input, size_hw kernel, size_hw padding, size_hw stride, size_hw dilation);
template
torch::Tensor unfold_template<int64_t>(const torch::Tensor& input, size_hw kernel, size_hw padding, size_hw stride, size_hw dilation);


/**
 * @param input indices/values: [batch, channel, height, width]
 * @param weight 
 *  indices: [out_channel, in_connects], 
 *  value: [out_channel, in_connects, kernel_height, kernel_width]
 * @param stride [height, width]
 * @param padding [height, width]
 * @return indices/values: [batch, sub_channels, height, width]
 *  {output, unfold_cols}
 */
template<typename T>
vector<SparseTensor> sparseConv2d_forward_template(
    SparseTensor& input,
    SparseTensor& weight,
    vector<int64_t> stride,
    vector<int64_t> padding
) {
    TORCH_CHECK(input.dtype() == weight.dtype()); // only use input.dtype for out.dtype
    TORCH_CHECK(input.sparse_dim() == 1, "input should be channel-first layout format.");
    TORCH_CHECK(input.indices().dim() == 4, "input.indices should be dim = 4");
    TORCH_CHECK(input.values().dim() == 4, "input.values should be dim = 4");
    TORCH_CHECK(weight.indices().dim() == 2, "weight.indices should be dim = 2");
    TORCH_CHECK(weight.values().dim() == 4, "weight.values should be dim = 4");
    TORCH_CHECK(stride.size() == 2, "stride must be length = 2");
    TORCH_CHECK(padding.size() == 2, "padding must be length = 2");
    int batch_size = input._size(0);
    // int in_connects = input._size(1);
    int in_height = input._size(2);
    int in_width = input._size(3);
    vector<int64_t> kernel = {weight._size(2), weight._size(3)};
    vector<int64_t> dilation = {1, 1};
    int out_height = conv_length(in_height, kernel[0], padding[0], stride[0], dilation[0]);
    int out_width = conv_length(in_width, kernel[1], padding[1], stride[1], dilation[1]);
    auto out_channels = weight._size(0);
    auto in_connects = weight._size(1);
    // make sure contiguous, then use pointer
    auto weight_vPtr = values_ptr<T>(weight);

    // ..._blocks [batch, sub_channels*k_h*k_w, blocks]
    auto indices_blocks = unfold_template<int64_t>(input.indices(), kernel, padding, stride, dilation);
    auto values_blocks = unfold_template<T>(input.values(), kernel, padding, stride, dilation);
    auto block_channels = indices_blocks.size(1);
    auto n_blocks = indices_blocks.size(2);
    // make sure contiguous, then use pointer
    auto indices_bPtr = data_ptr<int64_t>(indices_blocks);
    auto values_bPtr = values_blocks.contiguous().template data_ptr<T>();
    
    // calculate longest sub_channels of output
    int max_channels = 0;
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
        for (const auto bt : c10::irange(start, end)) { // batch
            for (const auto bk : c10::irange(n_blocks)) { // block
                
                unordered_set<int> out_ids;
                for (const auto ch : c10::irange(block_channels)) { // channel
                    auto channel_id = indices_bPtr[(bt * block_channels + ch) * n_blocks + bk];
                    // auto channel_id = indices_bPtr[(bt * n_blocks + bk) * block_channels + ch];
                    auto connects_ids = weight.c_id2offset(channel_id);
                    for (const auto& link_offset : connects_ids) {
                        out_ids.insert(link_offset / in_connects);
                    }
                }
                // todo...bug: max_channels may have multiple threads writing problem.
                max_channels = max(max_channels, (int)out_ids.size());
            }
        }
    });

    auto out = create_sparse_IdVal_options<T>(
        {batch_size, max_channels, out_height, out_width}, 
        input.values().options());
    
    // sparse matmul (determined by input)
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
        for (const auto bt : c10::irange(start, end)) { // batch
            for (const auto bk : c10::irange(n_blocks)) { // block
                
                // dotproduct each block with all related weight neurons.
                unordered_map<int64_t, T> out_map; // out_channel_id -> value
                for (const auto ch : c10::irange(block_channels)) { // channel
                    auto ptr = (bt * block_channels + ch) * n_blocks + bk;
                    // auto ptr = (bt * n_blocks + bk) * block_channels + ch;
                    auto channel_id = indices_bPtr[ptr];
                    auto val = values_bPtr[ptr];
                    auto k_w = ch % kernel[1];
                    auto k_h = ch % (kernel[1]*kernel[0]) / kernel[1];
                    auto out_ids = weight.c_id2offset(channel_id);
                    // retrieve all links, conneting the input point.
                    for (const auto& link_offset : out_ids) {
                        auto out_id = link_offset / in_connects;
                        out_map[out_id] += val * weight_vPtr[(link_offset * kernel[0] + k_h) * kernel[1] + k_w];
                    }
                }

                // summary each block
                int n = 0;
                for (const auto& o : out_map) {
                    out.id_ptr[(bt * max_channels + n) * n_blocks + bk] = o.first;
                    out.val_ptr[(bt * max_channels + n) * n_blocks + bk] = o.second;
                    n++;
                }
            }
        }
    });

    return {
        SparseTensor(out.indices, out.values, input.sparse_dim(), out_channels),
        SparseTensor(indices_blocks, values_blocks, 1, input.range()) // id still in [0, in_channels]
    };
}

template vector<SparseTensor> sparseConv2d_forward_template<float>(
    SparseTensor& input, SparseTensor& weight, vector<int64_t> stride, vector<int64_t> padding);
template vector<SparseTensor> sparseConv2d_forward_template<double>(
    SparseTensor& input, SparseTensor& weight, vector<int64_t> stride, vector<int64_t> padding);


/**
 * @param input indices/values: [batch, channel, height, width]
 * @param weight 
 *  indices: [out_channel, in_connects], 
 *  value: [out_channel, in_connects, kernel_height, kernel_width]
 * @param stride [height, width]
 * @param padding [height, width]
 * @return indices/values: [batch, sub_channels, height, width]
 *  {output, unfold_cols}
 */
template<typename T>
vector<SparseTensor> sparseConv2d_forward_template_v1(
    SparseTensor& input,
    SparseTensor& weight,
    vector<int64_t> stride,
    vector<int64_t> padding
) {
    TORCH_CHECK(input.dtype() == weight.dtype()); // only use input.dtype for out.dtype
    TORCH_CHECK(input.indices().dim() == 4, "input.indices/values should be dim = 4");
    TORCH_CHECK(input.sparse_dim() == 3, "input should be channel-last layout format.");
    TORCH_CHECK(input.values().dim() == 4, "input.values should be dim = 4");
    TORCH_CHECK(weight.indices().dim() == 2, "weight.indices should be dim = 2");
    TORCH_CHECK(weight.values().dim() == 4, "weight.values should be dim = 4");
    TORCH_CHECK(stride.size() == 2, "stride must be length = 2");
    TORCH_CHECK(padding.size() == 2, "padding must be length = 2");
    // TORCH_CHECK(input.is_coalesced(), "input must be coalesced");
    unfold_channel_last(input.indices());
    unfold_channel_last(input.values());
    // unfolded_cols [batch, num_blocks, kernel_h, kernel_w, in_sub_channels (sparse)]
    
    // find max_out_channels
    auto out_channels_stats = torch::zeros({ batch_size * num_blocks }, torch::dtype(torch::kI64));
    auto out_channels_stats_ptr = data_ptr<int64_t>(out_channels_stats);
    
    unfolded_merged [batch, num_blocks, kernel_h * kernel_w * in_sub_channels (sparse)]
    sizes_for(unfolded_merged, true, [&]() {
        // todo...
    });

    

    // create out tensor: [batch, num_block, kernel_h, kernel_w, max_out_channels(sparse)]
    auto max_out_channels = out_channels_stats.max().item<int64_t>();
    auto out = create_sparse_IdVal_options();

    parrallel_for(batch_size * num_blocks, true, [&](auto i) {
        // find all out_channels_ids required

        // todo...
        
    });


    return {
        SparseTensor(out.indices, out.values, input.sparse_dim(), out_channels),
        SparseTensor(indices_blocks, values_blocks, 1, input.range()) // id still in [0, in_channels]
    };
}


/**
 * @param grad [batch, sub_out_channels, out_height, out_width]
 * @param unfolded_inputs [batch, (sub_in_channels_hw) sub_in_channels * kernel_h * kernel_w, n_blocks (out_height*out_height)]
 *  id in unfolded_inputs is not include kernel_h, kernel_w
 * @param kernel [int, int]
 * @return SparseConv2dWeightGrad: sizes=[out_channels, in_channels, kernel_h, kernel_w]
 */
template<typename T>
torch::Tensor sparseConv2d_backward_template_v0(
    SparseTensor& grad, 
    SparseTensor& unfolded_inputs,
    vector<int64_t>& kernel
) {
    TORCH_CHECK(grad.dtype() == unfolded_inputs.dtype());
    auto grad_shape = grad.indices().sizes().vec();
    auto out_height = grad_shape[2];
    auto out_width = grad_shape[3];
    // auto sub_out_channels = grad._size(1);
    auto in_channels = unfolded_inputs._size(1);
    auto in_channels_range = unfolded_inputs.range();
    vector<int64_t> out_size = {grad.range(), in_channels_range, kernel[0], kernel[1]};
    auto grad_id_ptr = indices_ptr(grad);
    auto grad_val_ptr = values_ptr<T>(grad);
    auto input_id_ptr = indices_ptr(unfolded_inputs);
    auto input_val_ptr = values_ptr<T>(unfolded_inputs);

    unordered_map<int64_t, T> out_map;
    // be careful about the parallel, since they use the same unordered_map
    sizes_for_vec(grad_shape, false, [&](vector<int64_t> locals, const int64_t i) {
        auto b_i = locals[0];
        // auto out_c_offset = locals[1];
        auto h_i = locals[2];
        auto w_i = locals[3];
        auto grad_id = grad_id_ptr[i];
        auto grad_val = grad_val_ptr[i];
        for (auto const in_c_offset : c10::irange(in_channels)) {
            auto offset = ((b_i * in_channels + in_c_offset) * out_height + h_i) * out_width + w_i;
            auto in_c_i = input_id_ptr[offset];
            auto input_val = input_val_ptr[offset];
            // in_c_i not include kernel_size
            auto kh_i = in_c_offset % (kernel[0]*kernel[1]) / kernel[1];
            auto kw_i = in_c_offset % (kernel[0]*kernel[1]) % kernel[1];
            auto out_id = ((grad_id * in_channels_range + in_c_i) * kernel[0] + kh_i) * kernel[1] + kw_i;
            out_map[out_id] += grad_val * input_val;
        }
    });

    int nse = out_map.size();

    auto out = create_sparse_IdVal_options<T>({4, nse}, {nse}, grad.values().options());
    int nse_i = 0;
    vector<int64_t> locals(4, 0);
    for (const auto& item : out_map) {
        global2locals_offset(out_size, item.first, locals);
        for (size_t i = 0; i < locals.size(); i++) {
            out.id_ptr[i*nse + nse_i] = locals[i];
        }
        out.val_ptr[nse_i] = item.second;
        nse_i++;
    }

    return torch::sparse_coo_tensor(
        out.indices, out.values, out_size, grad.values().options().layout(torch::kSparse));
}

template torch::Tensor sparseConv2d_backward_template_v0<float>(
    SparseTensor& grad, SparseTensor& unfolded_inputs, vector<int64_t>& kernel);
template torch::Tensor sparseConv2d_backward_template_v0<double>(
    SparseTensor& grad, SparseTensor& unfolded_inputs, vector<int64_t>& kernel);


/**
 * @param arr a strided array. [index, ...]
 * Return:
 *  index of the first value.
 */
inline int64_t strided_binary_search(int64_t* arr, int64_t stride, int64_t start, int64_t end, int64_t value) {
    // TORCH_CHECK(false, "not valid, according to strided_loop_search.");
    if (end >= start) {
        int64_t mid = start + (end - start) / stride / 2 * stride; // make sure mid is times of stride, from start.
        if (arr[mid] == value) {
            // find the first one
            int64_t i = mid;
            while (i >= 0 && arr[i] == value) i -= stride;
            return i + stride;
        }
        if (arr[mid] > value)
            return strided_binary_search(arr, stride, start, mid - stride, value);
        else
            return strided_binary_search(arr, stride, mid + stride, end, value);
    }
    // not exsits
    return -1;
}

/**
 * simple loop version of searching, for validating `strided_binary_search`.
 */
inline int64_t strided_loop_search(int64_t* arr, int64_t stride, int64_t start, int64_t end, int64_t value) {
    for (int64_t i = start; i < end; i += stride) {
        if (arr[i] == value) return i;
    }
    return -1;
}


/**
 * @param grad [batch, sub_out_channels, out_height, out_width]
 * @param unfolded_inputs [batch, (sub_in_channels_hw) sub_in_channels * kernel_h * kernel_w, n_blocks (out_height*out_height)]
 *  id in unfolded_inputs is not include kernel_h, kernel_w
 * @param weight id: [out_channels, in_sub_channels], values: [..., kernel_height, kernel_width]
 * @param kernel [int, int]
 * @return SparseConv2dWeightGrad: sizes=[out_channels, in_channels, kernel_h, kernel_w]
 */
template<typename T>
SparseWeightGrads conv2d_backward_template(
    SparseTensor& grad, 
    SparseTensor& unfolded_inputs,
    SparseTensor& weight,
    vector<int64_t>& kernel
) {
    TORCH_CHECK(grad.dtype() == unfolded_inputs.dtype(), "grad and unfolded_inputs are not same dtype");
    TORCH_CHECK(unfolded_inputs.indices().dim() == 3 && unfolded_inputs.sparse_dim() == 1);
    TORCH_CHECK(grad.indices().dim() == 4);
    TORCH_CHECK(weight.values().dim() == 4);
    TORCH_CHECK(weight.range() == unfolded_inputs.range());
    auto grad_vals_ptr = values_ptr<T>(grad);
    auto grad_numel = grad.indices().numel();
    auto grad_sizes = grad.indices().sizes().vec();
    auto weight_ids_ptr = indices_ptr(weight);

    const auto sorted_tuple = grad.indices().view({ -1 }).sort();
    auto grad_ids = get<0>(sorted_tuple), grad_map = get<1>(sorted_tuple);
    auto grad_ids_ptr = data_ptr<int64_t>(grad_ids);
    auto grad_map_ptr = data_ptr<int64_t>(grad_map);
    
    // size of output (unique ids)
    auto out_channels_ids = vector<int64_t>();
    auto out_channels_id_start = vector<int64_t>(); // start offset of out_channels, in grad_ids
    for (int i = 0; i < grad_numel; i++) {
        if (i < 1 || grad_ids_ptr[i - 1] != grad_ids_ptr[i]) {
            out_channels_ids.push_back(grad_ids_ptr[i]);
            out_channels_id_start.push_back(i);
        }
    }

    // create output tensor
    auto out = create_sparse_IdVal_options<T>(
        { (int)out_channels_ids.size(), weight._size(1)},
        { (int)out_channels_ids.size(), weight._size(1), weight._size(2), weight._size(3) }, grad.values().options());
    // init out.indices (from part of weight.indices)
    for (size_t i = 0, stride = weight._size(1); i < out_channels_ids.size(); i++) {
        auto start = weight_ids_ptr + out_channels_ids[i] * stride;
        copy(start, start + stride, out.id_ptr + i * stride);
    }

    // start = high_resolution_clock::now();
    // sort unfolded_inputs
    auto sorted_input_ids = unfolded_inputs.indices().sort(1);
    // end = high_resolution_clock::now();
    // cout << "(unfolded_inputs sort): " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;
    
    auto input_val_ptr = values_ptr<T>(unfolded_inputs);
    auto sorted_input_id_ptr = data_ptr<int64_t>(get<0>(sorted_input_ids));
    auto sorted_input_map_ptr = data_ptr<int64_t>(get<1>(sorted_input_ids));
    // calculate out.values
    // int out_offset = 0; // todo... move to paralle_for
    int out_channels_stride = weight._size(1) * weight._size(2) * weight._size(3);
    // auto grad_locals = vector<int64_t>(grad_sizes.size(), 0); // todo... only for serial
    auto input_example_size = unfolded_inputs._size(1) * unfolded_inputs._size(2);
    auto kernel_size = kernel[0] * kernel[1];
    auto in_channels_stride = unfolded_inputs._size(2); // include kernel_h, kernel_w
    auto weight_in_channels = weight._size(1);

    // start = high_resolution_clock::now();
    // multi threads: same out_id must only in one thread
    parallel_for((int)out_channels_ids.size(), true, [&](auto out_id_offset) {
        auto out_channels_id = out_channels_ids[out_id_offset];
        auto out_id_start = out_channels_id_start[out_id_offset];
        auto out_offset = out_channels_stride * out_id_offset;
        int64_t grad_locals[grad_sizes.size()];

        // traversal when same out_channels_ids (contiguous).
        for (int i = out_id_start; out_channels_id == grad_ids_ptr[i] && i < grad_numel; i++) {
            auto offset = grad_map_ptr[i];
            const auto val = grad_vals_ptr[offset];
            global2locals_offset(grad_sizes, offset, grad_locals);
            // traversal each in_connect, from weight.indice
            for (int j = 0; j < weight_in_channels; j++) {
                auto in_channels_id = weight_ids_ptr[out_channels_id * weight_in_channels + j];
                auto start = grad_locals[0] * input_example_size + grad_locals[2] * grad_sizes[3] + grad_locals[3];
                auto end = start + in_channels_stride * unfolded_inputs._size(1);

                auto offset = strided_binary_search(sorted_input_id_ptr, in_channels_stride, start, end, in_channels_id);

                // write to out tensor if non-zero in inputs
                if (offset < 0) continue;
                // each point within [kernel_h, kernel_w]
                for (int k = offset; k < end; k += in_channels_stride) {
                    // channels_offset: [sub_in_channels, kernel_h, kernel_w]
                    auto channels_offset = sorted_input_map_ptr[k];
                    auto kernel_offset = channels_offset % kernel_size;
                    auto in_val = input_val_ptr[start + channels_offset * in_channels_stride];
                    out.val_ptr[out_offset + j * kernel_size + kernel_offset] += in_val * val;
                    if (k + in_channels_stride < end
                        && sorted_input_id_ptr[k] != sorted_input_id_ptr[k + in_channels_stride]) break;
                }
            }
        }
    });
    // end = high_resolution_clock::now();
    // cout << "(body elapse): " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;

    // for (int i = 0; i < grad_numel; i++) {
    //     auto out_channels_id = grad_ids_ptr[i];
    //     auto offset = grad_map_ptr[i];
    //     const auto val = grad_vals_ptr[offset];
    //     global2locals_offset(grad_sizes, offset, grad_locals);
    //     // traversal each in_connect, from weight.indice
    //     for (int j = 0; j < weight_in_channels; j++) {
    //         auto in_channels_id = weight_ids_ptr[out_channels_id * weight_in_channels + j];
    //         auto start = grad_locals[0] * input_example_size + grad_locals[2] * grad_sizes[3] + grad_locals[3];
    //         auto end = start + in_channels_stride * unfolded_inputs._size(1);

    //         auto offset = strided_binary_search(sorted_input_id_ptr, in_channels_stride, start, end, in_channels_id);

    //         // write to out tensor if non-zero in inputs
    //         if (offset < 0) continue;
    //         // each point within [kernel_h, kernel_w]
    //         for (int k = offset; k < end; k += in_channels_stride) {
    //             // channels_offset: [sub_in_channels, kernel_h, kernel_w]
    //             auto channels_offset = sorted_input_map_ptr[k];
    //             auto kernel_offset = channels_offset % kernel_size;
    //             auto in_val = input_val_ptr[start + channels_offset * in_channels_stride];
    //             out.val_ptr[out_offset + j * kernel_size + kernel_offset] += in_val * val;
    //             if (k + in_channels_stride < end
    //                 && sorted_input_id_ptr[k] != sorted_input_id_ptr[k + in_channels_stride]) break;
    //         }
    //     }
    //     // to next out_channels_id
    //     if (i + 1 < grad_numel && grad_ids_ptr[i] != grad_ids_ptr[i + 1]) {
    //         out_offset += out_channels_stride;
    //     }
    // }

    
    return {
        torch::tensor(out_channels_ids),
        SparseTensor(out.indices, out.values, weight.sparse_dim(), weight.range())
    };
}

template SparseWeightGrads conv2d_backward_template<float>(
    SparseTensor& grad, SparseTensor& unfolded_inputs, SparseTensor& weight, vector<int64_t>& kernel);
template SparseWeightGrads conv2d_backward_template<double>(
    SparseTensor& grad, SparseTensor& unfolded_inputs, SparseTensor& weight, vector<int64_t>& kernel);

