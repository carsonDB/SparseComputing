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
 * @param input shape [batch, channel, height, width]
 */
template<typename T>
torch::Tensor unfold(const torch::Tensor& input, size_hw kernel, size_hw padding, size_hw stride, size_hw dilation) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_height = conv_length(in_height, kernel.h, padding.h, stride.h, dilation.h);
    int out_width = conv_length(in_width, kernel.w, padding.w, stride.w, dilation.w);
    int out_channels = in_channels * kernel.h * kernel.w;

    auto output = torch::zeros({ batch_size, out_channels, out_height * out_width }, input.options());
    // make sure contiguous, then use pointer
    auto in_ptr = input.contiguous().data_ptr<T>();
    auto out_ptr = output.data_ptr<T>();
    at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {

        for (const auto b_i : c10::irange(start, end)) {
            
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
                        // out_ptr[((b_i * out_height + h_i) * out_width + w_i) * out_channels + col_i] =
                        //     (h_in >= 0 && w_in >= 0 && h_in < in_height && w_in < in_width)
                        //     ? in_ptr[((b_i * in_channels + c_in) * in_height + h_in) * in_width + w_in]
                        //     : static_cast<T>(0);
                    }
                }
            }
        }
    });

    return output;
}


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
    auto weight_vPtr = weight.values().contiguous().data_ptr<T>();

    // ..._blocks [batch, sub_channels*k_h*k_w, blocks]
    auto indices_blocks = unfold<int64_t>(input.indices(), kernel, padding, stride, dilation);
    auto values_blocks = unfold<T>(input.values(), kernel, padding, stride, dilation);
    auto block_channels = indices_blocks.size(1);
    auto n_blocks = indices_blocks.size(2);
    // make sure contiguous, then use pointer
    auto indices_bPtr = indices_blocks.contiguous().data_ptr<int64_t>();
    auto values_bPtr = values_blocks.contiguous().template data_ptr<T>();
    
    // calculate longest sub_channels of output
    int max_channels = 0;
    // auto start = high_resolution_clock::now();
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
                max_channels = max(max_channels, (int)out_ids.size());
            }
        }
    });
    // auto end = high_resolution_clock::now();
    // cout << "(elapse): " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;

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
 * @param grad [batch, sub_out_channels, out_height, out_width]
 * @param unfolded_inputs [batch, (sub_in_channels_hw) sub_in_channels * kernel_h * kernel_w, n_blocks (out_height*out_height)]
 *  id in unfolded_inputs is not include kernel_h, kernel_w
 * @param kernel [int, int]
 * @return SparseConv2dWeightGrad: sizes=[out_channels, in_channels, kernel_h, kernel_w]
 */
template<typename T>
torch::Tensor sparseConv2d_backward_template(
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
    auto grad_id_ptr = grad.c_indices_ptr();
    auto grad_val_ptr = grad.values().contiguous().data_ptr<T>();
    auto input_id_ptr = unfolded_inputs.c_indices_ptr();
    auto input_val_ptr = unfolded_inputs.values().contiguous().data_ptr<T>();

    unordered_map<int64_t, T> out_map;
    // be careful about the parallel, since they use the same unordered_map
    sizes_for(grad_shape, false, [&](vector<int64_t> locals, const int64_t i) {
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
        global2local_offset(out_size, item.first, locals);
        for (size_t i = 0; i < locals.size(); i++) {
            out.id_ptr[i*nse + nse_i] = locals[i];
        }
        out.val_ptr[nse_i] = item.second;
        nse_i++;
    }

    return torch::sparse_coo_tensor(
        out.indices, out.values, out_size, grad.values().options().layout(torch::kSparse));
}

template torch::Tensor sparseConv2d_backward_template<float>(
    SparseTensor& grad, SparseTensor& unfolded_inputs, vector<int64_t>& kernel);
template torch::Tensor sparseConv2d_backward_template<double>(
    SparseTensor& grad, SparseTensor& unfolded_inputs, vector<int64_t>& kernel);

