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
