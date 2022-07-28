#include "sparseLinear.h"


/**
 * @param input [batch, sub_channels(sparse), height, width]
 * @param weight [out_channels, in_channels]
 * @return dense tensor [batch, out_channels]
 */
torch::Tensor sparseLinear_forward(SparseTensor& input, torch::Tensor& weight) {
    TORCH_CHECK(input.sparse_dim() == 1);
    TORCH_CHECK(input.indices().dim() == 4, "input shape: [batch, sub_channels(sparse), height, width]");
    auto batch_size = input._size(0);
    auto sub_channels = input._size(1);
    auto in_height = input._size(2);
    auto in_width = input._size(3);
    auto input_numel = input.indices().numel();
    auto out_channels = weight.size(0);
    auto in_channels = weight.size(1);
    auto output = torch::zeros({batch_size, out_channels}, input.values().options());
    auto in_id_ptr = input.indices().contiguous().data_ptr<int64_t>();
    auto in_val_ptr = values_ptr<float>(input);
    auto weight_ptr = weight.contiguous().data_ptr<float>();
    auto out_ptr = output.data_ptr<float>();
    
    at::parallel_for(0, input_numel, 0, [&](int64_t start, int64_t end) {
        for (const auto i : c10::irange(start, end)) {
            auto w_i = i % in_width;
            auto h_i = i % (in_height * in_width) / in_width;
            auto b_i = i / (sub_channels * in_height * in_width);
            auto id = in_id_ptr[i];
            auto val = in_val_ptr[i];
            // convert original id -> fatten id
            auto flatten_id = (id * in_height + h_i) * in_width + w_i;
            for (const auto j : c10::irange(out_channels)) {
                out_ptr[b_i * out_channels + j] += weight_ptr[j * in_channels + flatten_id] * val;
            }
        }
    });

    return output;
}


/**
 * @param input [batch, sub_channels(sparse), height, width]
 * @param grad [batch, out_channels]
 * @return [out_channels, sub_in_channels (sparse)]
 */
SparseTensor sparseLinear_backward(SparseTensor& input, torch::Tensor& grad) {
    TORCH_CHECK(input.indices().dim() == 4);
    auto input_numel = input.indices().numel();
    auto sub_channels = input.indices().size(1);
    auto in_height = input.indices().size(2);
    auto in_width = input.indices().size(3);
    auto out_channels = grad.size(1);
    auto in_id_ptr = input.indices().contiguous().data_ptr<int64_t>();
    auto in_val_ptr = values_ptr<float>(input);
    auto grad_ptr = data_ptr<float>(grad);
    // find sub_in_channels
    unordered_set<int64_t> id_set;
    for (const auto i : c10::irange(input_numel)) {
        auto id = in_id_ptr[i];
        auto h_i = i % (in_height * in_width) / in_width;
        auto w_i = i % in_width;
        // sub_channels -> sub_in_channels (include height, width)
        id_set.insert((id * in_height + h_i) * in_width + w_i);
    }
    int64_t sub_in_channels = id_set.size();

    auto out = create_sparse_IdVal({out_channels, sub_in_channels});

    for (const auto o : c10::irange(out_channels)) {
        unordered_map<int64_t, float> out_map; // in_channel_id -> value (out_channel_id = o)
        for (const auto i : c10::irange(input_numel)) {
            auto id = in_id_ptr[i];
            auto val = in_val_ptr[i];
            auto h_i = i % (in_height * in_width) / in_width;
            auto w_i = i % in_width;
            auto b_i = i / (in_height * in_width * sub_channels);
            auto in_channel_id = (id * in_height + h_i) * in_width + w_i;
            out_map[in_channel_id] += val * grad_ptr[b_i * out_channels + o];
        }
        int n = 0;
        for (const auto& item : out_map) {
            out.id_ptr[o * sub_in_channels + n] = item.first;
            out.val_ptr[o * sub_in_channels + n] = item.second;
            n++;
        }
    }

    return SparseTensor(out.indices, out.values, 1, input.range() * in_height * in_width);
}


/**
 * @param input [batch, sub_channels(sparse), height, width]
 * @param grad [batch, out_channels]
 * @return coo_sparse [out_channels, in_channels]
 */
torch::Tensor linear_backward_coo_v0(SparseTensor& input, torch::Tensor& grad) {
    TORCH_CHECK(input.indices().dim() == 4);
    auto input_numel = input.indices().numel();
    auto sub_channels = input.indices().size(1);
    auto in_height = input.indices().size(2);
    auto in_width = input.indices().size(3);
    auto out_channels = grad.size(1);
    auto in_channels = input.range() * in_height * in_width;
    auto in_id_ptr = indices_ptr(input);
    auto in_val_ptr = values_ptr<float>(input);
    auto grad_ptr = grad.contiguous().data_ptr<float>();
    // find sub_in_channels
    unordered_set<int64_t> id_set;
    for (const auto i : c10::irange(input_numel)) {
        auto id = in_id_ptr[i];
        auto h_i = i % (in_height * in_width) / in_width;
        auto w_i = i % in_width;
        // sub_channels -> sub_in_channels (include height, width)
        id_set.insert((id * in_height + h_i) * in_width + w_i);
    }
    int64_t sub_in_channels = id_set.size();

    auto nse = out_channels * sub_in_channels;
    auto out = create_sparse_IdVal({2, nse}, {nse});

    int n = 0;
    for (const auto o : c10::irange(out_channels)) {
        unordered_map<int64_t, float> out_map; // in_channel_id -> value (out_channel_id = o)
        for (const auto i : c10::irange(input_numel)) {
            auto id = in_id_ptr[i];
            auto val = in_val_ptr[i];
            auto h_i = i % (in_height * in_width) / in_width;
            auto w_i = i % in_width;
            auto b_i = i / (in_height * in_width * sub_channels);
            auto in_channel_id = (id * in_height + h_i) * in_width + w_i;
            out_map[in_channel_id] += val * grad_ptr[b_i * out_channels + o];
        }
        for (const auto& item : out_map) {
            // [0, n]
            out.id_ptr[n] = o;
            // [1, n]
            out.id_ptr[nse + n] = item.first;
            // [n]
            out.val_ptr[n] = item.second;
            n++;
        }
    }

    return torch::sparse_coo_tensor(out.indices, out.values, {out_channels, in_channels}, torch::dtype(torch::kF32));
}


/**
 * @param input [batch, sub_channels(sparse)]
 * @param grad [batch, out_channels]
 * @return coo_sparse nse=[out_channels, in_channels], ids=[2, nse], vals=[nse]
 */
template<typename T>
torch::Tensor linear_backward_coo_template(SparseTensor& input, torch::Tensor& grad) {
    TORCH_CHECK(input.indices().dim() == 2);
    TORCH_CHECK(input.sparse_dim() == 1, "input: [batch, sub_channels(sparse)]");
    TORCH_CHECK(input._size(0) == grad.size(0), "input and grad should have same batch_size")
    TORCH_CHECK(input.is_coalesced(), "input must be called coalesce() before");

    // sorted merge (input [batch, channels (sparse)] -> merged [channels (sparse)])
    auto merge_dim = input.sparse_dim();
    auto merge_axis = set<int64_t>{0, 1};
    auto sorted_pair = sorted_merge(input.indices(), merge_axis, merge_dim);
    auto merged_size = sorted_pair.first.size(merge_dim);
    auto sorted_ids_ptr = data_ptr<int64_t>(sorted_pair.first);
    auto sorted_map_ptr = data_ptr<int64_t>(sorted_pair.second);

    // give each entry of input an offset of merged and unique flatten out ids.
    auto in2unique = vector<int64_t>(input.indices().numel(), 0);
    auto out_ids = vector<int64_t>();
    // unique ids (sorted_ids: [1, merged_size])
    for (int i = 0; sorted_map_ptr[i] >= 0 && i < merged_size; i++) {
        in2unique[sorted_map_ptr[i]] = (int)out_ids.size();
        if (i + 1 == merged_size || sorted_ids_ptr[i] != sorted_ids_ptr[i + 1]) {
            out_ids.push_back(sorted_ids_ptr[i]);
        }
    }

    // create output
    int64_t out_channels = grad.size(1);
    int64_t out_ids_len = out_ids.size();
    int64_t nse = out_channels * out_ids_len;
    auto out = create_sparse_IdVal({2, nse}, {nse});
    // fill ids to out.ids
    auto out_channels_ids = torch::arange(out_channels).view({ out_channels, 1 })
        .expand({out_channels, out_ids_len});
    out.indices.index_put_({ 0, "..." }, out_channels_ids.reshape({-1}));
    for (int i = 0; i < out_channels; i++)
        copy(out_ids.begin(), out_ids.end(), out.id_ptr + nse + i * out_ids_len);

    // product of each corresponding points.
    // input: [batch, 1, in_sub_channels];
    // grad: [batch, out_channels, 1];
    auto grad_w = input.values().view({input._size(0), 1, input._size(1)}) *
                  grad.view({grad.size(0), grad.size(1), 1});
    // grad_w: [batch, out_channels, in_sub_channels]

    auto grad_w_ptr = data_ptr<T>(grad_w);
    auto batch_size = grad_w.size(0);
    auto in_sub_channels = grad_w.size(2);
    at::parallel_for(0, out_channels, 0, [&](int64_t start, int64_t end) {
        for (const auto& out_id : c10::irange(start, end)) {

            auto out_channels_start = out_id * in_sub_channels;
            auto val_start = out_id * out_ids_len;
            for (const auto& i : c10::irange(batch_size)) {
                for (const auto& j : c10::irange(in_sub_channels)) {
                    // grad_w: [batch, out_channels, in_sub_channels]
                    auto grad_offset = out_channels_start + i * out_channels * in_sub_channels + j;
                    // in2unique: [batch, in_sub_channels] -> vals: [batch, out_ids_len]
                    auto val_offset = val_start + in2unique[i * in_sub_channels + j];
                    out.val_ptr[val_offset] += grad_w_ptr[grad_offset];
                }
            }
        }
    });

    auto out_size = vector<int64_t>{out_channels, input.range()};
    return torch::sparse_coo_tensor(out.indices, out.values, out_size, grad.options().layout(torch::kSparse));
}


template torch::Tensor linear_backward_coo_template<float>(SparseTensor& input, torch::Tensor& grad);
template torch::Tensor linear_backward_coo_template<double>(SparseTensor& input, torch::Tensor& grad);
