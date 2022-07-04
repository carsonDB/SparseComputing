#include "sparseMM.h"


/**
 * Designed for linear layer
 * @param input [batch, sub_channels(sparse), height, width]
 * @param weight [out_channels, in_channels]
 * @return dense tensor [batch, out_channels]
 */
torch::Tensor sparseDenseMM_forward(SparseTensor& input, torch::Tensor& weight) {
    TORCH_CHECK(input.sparse_dim() == 1);
    auto batch_size = input._size(0);
    auto sub_channels = input._size(1);
    auto in_height = input._size(2);
    auto in_width = input._size(3);
    auto input_numel = input.indices().numel();
    auto out_channels = weight.size(0);
    auto in_channels = weight.size(1);
    auto output = torch::zeros({batch_size, out_channels}, torch::dtype(torch::kF32));
    auto in_id_ptr = input.indices().contiguous().data_ptr<int64_t>();
    auto in_val_ptr = input.values().contiguous().data_ptr<float>();
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
 * Designed for linear layer
 * @param input [batch, sub_channels(sparse), height, width]
 * @param grad [batch, out_channels]
 * @return [out_channels, sub_in_channels (sparse)]
 */
SparseTensor sparseDenseMM_backward(SparseTensor& input, torch::Tensor& grad) {
    TORCH_CHECK(input.indices().dim() == 4);
    auto input_numel = input.indices().numel();
    auto sub_channels = input.indices().size(1);
    auto in_height = input.indices().size(2);
    auto in_width = input.indices().size(3);
    auto out_channels = grad.size(1);
    auto in_id_ptr = input.indices().contiguous().data_ptr<int64_t>();
    auto in_val_ptr = input.values().contiguous().data_ptr<float>();
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
 * Designed for linear layer
 * @param input [batch, sub_channels(sparse), height, width]
 * @param grad [batch, out_channels]
 * @return coo_sparse [out_channels, in_channels]
 */
torch::Tensor sparseDenseMM_backward_coo(SparseTensor& input, torch::Tensor& grad) {
    TORCH_CHECK(input.indices().dim() == 4);
    auto input_numel = input.indices().numel();
    auto sub_channels = input.indices().size(1);
    auto in_height = input.indices().size(2);
    auto in_width = input.indices().size(3);
    auto out_channels = grad.size(1);
    auto in_channels = input.range() * in_height * in_width;
    auto in_id_ptr = input.indices().contiguous().data_ptr<int64_t>();
    auto in_val_ptr = input.values().contiguous().data_ptr<float>();
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
