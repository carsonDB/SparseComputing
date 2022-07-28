#include "utils.h"

/**
 * uniform sample without replacement.
 * From: https://www.educative.io/edpresso/what-is-reservoir-sampling
 * @param n sample from
 * @param k number samples
 * @return torch::Tensor 
 */
torch::Tensor reservoir_sampling(int n, int k) {
    auto input = torch::arange(n, torch::dtype(torch::kI64));
    auto output = torch::empty({k}, torch::kI64);
    // Initializing the output array to the first k elements
    // of the input array.
    output.index_put_({thIndex::Slice(0, k, 1)}, input.index({thIndex::Slice(0, k, 1)}));
    
    // Iterating over k to n-1 
    for(int i = k; i < n; i++){
        // Generating a random number from 0 to i
        auto z = torch::randint(0, i+1, {1}, torch::dtype(torch::kI64)).item<int64_t>();
        // Replacing an element in the  output with an element
        // in the input if the randomly generated number is less
        // than k.
        if(z < k){
            output.index_put_({z}, input.index({i})); // todo...abandon index...
        }
    }

    return output;
}


template<typename T>
struct MergeInfo {
    T* input;
    int64_t length; // length of one sorted array
    int64_t stride; // stride of adjacent elements in (physical address) array
    function<int64_t(int64_t)> addr_map; // map from logical address to physical address.
};


/**
 * start1/2 represents logical address, contiguous from start1 to start2 (represent id of each array).
 * Must be converted to physical address by addr_map, when used in input/output memory.
 * Inplace version, modified from: https://www.geeksforgeeks.org/merge-k-sorted-arrays/, stack overflow otherwise.
 * @param start1 start logical address of first array
 * @param start2 start logical address of last array
 * @param output map indices (offset) to original input.
 * @return valid length of merged array
 */
template<typename T>
int64_t merge_strided_arrays(int64_t* output, int64_t start1, int64_t start2, MergeInfo<T>& info) {
    auto n = info.length;
    auto input = info.input;
    auto stride = info.stride;
    // ignore zeros tailing
    auto is_ordered = [&stride, &input](int64_t start, int64_t i) {
        return i < 1 || input[start + (i - 1) * stride] <= input[start + i * stride];
    };
    // only one array
    if (start1 == start2) {
        auto start_global = info.addr_map(start1);
        int64_t i = 0;
        for(; i < n && is_ordered(start_global, i); i++)
            output[i] = start_global + i * stride;
        return i;
    }
    // merge two sorted arrays (compare by value, and store indice of value into output array)
    else if (start2 - start1 == 1) {
        auto addr1 = info.addr_map(start1);
        auto addr2 = info.addr_map(start2);
        int64_t i = 0, j = 0, k = 0;
        while (i < n && j < n && is_ordered(addr1, i) && is_ordered(addr2, j)) {
            auto z1 = addr1 + i * stride, z2 = addr2 + j * stride;
            if (input[z1] < input[z2]) {
                output[k++] = z1;
                i++;
            }
            else {
                output[k++] = z2;
                j++;
            }
        }
        for (; i < n && is_ordered(addr1, i); i++)
            output[k++] = addr1 + i * stride;
        for (; j < n && is_ordered(addr2, j); j++) 
            output[k++] = addr2 + j * stride; 
        return k;
    }

    // divide and merge each one recursively.
    auto mid = (start1 + start2) / 2;
    // auto n1 = n * (mid - start1 + 1), n2 = n * (start2 - mid);
    // int64_t merged1[n1], merged2[n2];
    auto n1 = merge_strided_arrays<T>(output, start1, mid, info);
    auto n2 = merge_strided_arrays<T>(output + n1, mid + 1, start2, info);

    // merge above two (merged1/2 store global indice of value in input)
    // int64_t i = 0, j = 0, k = 0;
    // while (i < n1 && j < n2)
    //     if (input[merged1[i]] < input[merged2[j]])
    //         output[k++] = merged1[i++];
    //     else
    //         output[k++] = merged2[j++];
    // for (; i < n1; i++)
    //     output[k++] = merged1[i];
    // for (; j < n2; j++) 
    //     output[k++] = merged2[j]; 
    inplace_merge(output, output + n1, output + n1 + n2, 
        [&input](auto a, auto b) { return input[a] < input[b]; });
    
    return n1 + n2;
}

/** Allow variant-length of sorted arrays, with zeros-tailing.
 * @param merge_axis should include sorted_axis.
 * @param sorted_axis sorted arrays are along this.
 * @return pair<merged_values, merged_map>, merged_size=1, all keepdims=True
 * merged_map: map from merged_offset to original offset. (-int_max for default, which should not be used)
 */
pair<torch::Tensor, torch::Tensor> sorted_merge(
    const torch::Tensor& input, 
    set<int64_t>& merge_axis, 
    int64_t sorted_axis
) {
    TORCH_CHECK(input.dtype() == torch::kI64);
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(merge_axis.count(sorted_axis) > 0);
    auto in_shape = input.sizes().vec();
    // get unrelated/merged shape, ignored dim size = 1.
    vector<int64_t> unrelated_shape = in_shape;
    vector<int64_t> merge_shape = in_shape;
    for (int64_t i = 0; i < (int64_t)in_shape.size(); i++) {
        merge_axis.count(i) > 0 ? 
            (unrelated_shape[i] = 1) :
            (merge_shape[i] = 1);
    }

    // create out merged tensor
    const auto merged_size = accumulate(merge_shape.begin(), merge_shape.end(), 1, multiplies<int64_t>());
    vector<int64_t> out_shape = unrelated_shape;
    out_shape[sorted_axis] = merged_size;
    auto out_vals = torch::zeros(out_shape, input.options());
    auto out_map = torch::zeros(out_shape, input.options().dtype(torch::kI64));
    auto out_map_ptr = out_map.data_ptr<int64_t>();

    // traveral each merge case
    auto sorted_length = in_shape[sorted_axis];
    auto n_array = merged_size / sorted_length; // number of sorted arrays, in one merge example.
    auto array_shape = merge_shape;
    array_shape[sorted_axis] = 1; // merge_shape without sorted_axis
    auto in_ptr = input.data_ptr<int64_t>();
    auto out_vals_ptr = out_vals.data_ptr<int64_t>();
    // stride for sorted_axis
    auto sorted_stride = accumulate(
        in_shape.begin() + sorted_axis + 1, in_shape.end(), 1, multiplies<int64_t>());
    // stride for merged (output) dimension 
    auto out_stride = accumulate(out_shape.begin() + sorted_axis + 1, out_shape.end(), 1, multiplies<int64_t>());
    // // record max_len stats
    // const auto unrelated_size = accumulate(unrelated_shape.begin(), unrelated_shape.end(), 1, multiplies<int64_t>());
    // auto max_lens = vector(unrelated_size, 0);

    sizes_for(unrelated_shape, true, [&](auto locals, auto global) {
        auto array_locals = vector<int64_t>(array_shape.size(), 1);
        auto unrelated_start = local2global_offset(in_shape, locals);
        function<int64_t(int64_t)> arrayId2addr = [&](int64_t arrayId) {
            global2locals_offset(array_shape, arrayId, array_locals);
            auto array_offset = local2global_offset(in_shape, array_locals);
            return unrelated_start + array_offset;
        };
        // heap alloc size larger than stack size (int64_t[...])
        // otherwise, stack overflow
        auto output = new int64_t[merged_size]{ -numeric_limits<int64_t>::max() };
        MergeInfo<int64_t> info = { in_ptr, sorted_length, sorted_stride, arrayId2addr };
        
        auto max_len = merge_strided_arrays<int64_t>(output, 0, n_array - 1, info);
        auto unrelated_start_for_output = local2global_offset(out_shape, locals);
        for (int64_t i = 0; i < max_len; i++) {
            auto j = unrelated_start_for_output + i * out_stride;
            // copy contiguous output indices into strided out_map.
            out_map_ptr[j] = output[i];
            // fill out_vals.
            out_vals_ptr[j] = in_ptr[output[i]];
        }
        // remove dynamic alloc memory
        delete output;
    });

    return { out_vals, out_map };
}
