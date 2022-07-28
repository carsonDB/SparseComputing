#include "sparseConv.h"
#include "sparseLinear.h"
#include "sparseOps.h"
#include "sparseTensor.h"
#include "utils.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // sparseConv
    py::class_<size_hw>(m, "size_hw")
        .def(py::init<vector<int64_t> &>())
    ;
    m.def("unfold", &unfold);
    m.def("sparseConv2d_forward", &sparseConv2d_forward, "SparseConv forward");
    m.def("sparseConv2d_backward_v0", &sparseConv2d_backward_v0, "SparseConv backward v0");
    m.def("conv2d_backward", &conv2d_backward);
    // sparseMM
    m.def("sparseLinear_forward", &sparseLinear_forward);
    m.def("sparseLinear_backward", &sparseLinear_backward);
    m.def("linear_backward_coo_v0", &linear_backward_coo_v0);
    m.def("linear_backward_coo", &linear_backward_coo);
    // sparseOps
    m.def("reduce_max", &reduce_max);
    m.def("reduce_min", &reduce_min);
    m.def("reduce_sum", &reduce_sum);
    m.def("reduce_count_nonzero", &reduce_count_nonzero);
    m.def("reduce_prod", &reduce_prod);
    m.def("reduce_max_v0", &reduce_max_v0);
    m.def("reduce_min_v0", &reduce_min_v0);
    m.def("reduce_sum_v0", &reduce_sum_v0);
    m.def("reduce_count_nonzero_v0", &reduce_count_nonzero_v0);
    m.def("reduce_prod_v0", &reduce_prod_v0);

    m.def("elementwise_add_v0", py::overload_cast<SparseTensor&, SparseTensor&>(&elementwise_add_v0));
    m.def("elementwise_add", py::overload_cast<SparseTensor&, SparseTensor&>(&elementwise_add));
    m.def("elementwise_sub_v0", py::overload_cast<SparseTensor&, SparseTensor&>(&elementwise_sub_v0));
    m.def("elementwise_sub", py::overload_cast<SparseTensor&, SparseTensor&>(&elementwise_sub));
    m.def("elementwise_mul_v0", py::overload_cast<SparseTensor&, SparseTensor&>(&elementwise_mul_v0));
    m.def("elementwise_mul", py::overload_cast<SparseTensor&, SparseTensor&>(&elementwise_mul));
    m.def("elementwise_div_v0", py::overload_cast<SparseTensor&, SparseTensor&>(&elementwise_div_v0));
    m.def("elementwise_div", py::overload_cast<SparseTensor&, SparseTensor&>(&elementwise_div));

    m.def("elementwise_add", py::overload_cast<SparseTensor&, torch::Tensor&, bool>(&elementwise_add));
    m.def("elementwise_sub", py::overload_cast<SparseTensor&, torch::Tensor&, bool>(&elementwise_sub));
    m.def("elementwise_mul", py::overload_cast<SparseTensor&, torch::Tensor&, bool>(&elementwise_mul));
    m.def("elementwise_div", py::overload_cast<SparseTensor&, torch::Tensor&, bool>(&elementwise_div));
    m.def("reshape", &reshape);
    // sparseTensor
    py::class_<SparseTensor>(m, "SparseTensor")
        .def(py::init<torch::Tensor&, torch::Tensor&, int64_t, int64_t>())
        .def("indices", &SparseTensor::indices)
        .def("values", &SparseTensor::values)
        .def("range", &SparseTensor::range)
        .def("sparse_dim", &SparseTensor::sparse_dim)
        .def("init_rand_indices", &SparseTensor::init_rand_indices)
        .def("update_with", &SparseTensor::update_with)
        .def("is_coalesced", &SparseTensor::is_coalesced)
        .def("coalesce", &SparseTensor::coalesce)
    ;
    // utils
    m.def("sorted_merge", &sorted_merge);
}