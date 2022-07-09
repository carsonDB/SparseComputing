#include "sparseConv.h"
#include "sparseMM.h"
#include "sparseOps.h"
#include "sparseTensor.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // bind_sparseConv(m);
    py::class_<size_hw>(m, "size_hw")
        .def(py::init<vector<int64_t> &>())
    ;
    m.def("unfold", &unfold<float>, "unfold");
    m.def("sparseConv2d_forward", &sparseConv2d_forward, "SparseConv forward");
    m.def("sparseConv2d_backward", &sparseConv2d_backward, "SparseConv backward");
    // bind_sparseMM(m);
    m.def("sparseDenseMM_forward", &sparseDenseMM_forward, "Sparse dense MM forward");
    m.def("sparseDenseMM_backward", &sparseDenseMM_backward, "Sparse dense MM backward");
    m.def("sparseDenseMM_backward_coo", &sparseDenseMM_backward_coo, "Sparse dense MM backward with coo output");
    // bind_sparseOps(m);
    m.def("reduce_sum", &reduce_sum);
    m.def("reduce_count_nonzero", &reduce_count_nonzero);
    m.def("reduce_prod", &reduce_prod);
    m.def("elementwise_add", py::overload_cast<SparseTensor&, SparseTensor&>(&elementwise_add));
    m.def("elementwise_add", py::overload_cast<SparseTensor&, torch::Tensor&, bool>(&elementwise_add));
    m.def("elementwise_sub", py::overload_cast<SparseTensor&, SparseTensor&>(&elementwise_sub));
    m.def("elementwise_sub", py::overload_cast<SparseTensor&, torch::Tensor&, bool>(&elementwise_sub));
    m.def("elementwise_mul", py::overload_cast<SparseTensor&, SparseTensor&>(&elementwise_mul));
    m.def("elementwise_mul", py::overload_cast<SparseTensor&, torch::Tensor&, bool>(&elementwise_mul));
    m.def("elementwise_div", py::overload_cast<SparseTensor&, SparseTensor&>(&elementwise_div));
    m.def("elementwise_div", py::overload_cast<SparseTensor&, torch::Tensor&, bool>(&elementwise_div));
    // bind_sparseTensor(m);
    py::class_<SparseTensor>(m, "SparseTensor")
        .def(py::init<torch::Tensor&, torch::Tensor&, int64_t, int64_t>())
        .def("indices", &SparseTensor::indices)
        .def("values", &SparseTensor::values)
        .def("range", &SparseTensor::range)
        .def("sparse_dim", &SparseTensor::sparse_dim)
        .def("init_rand_indices", &SparseTensor::init_rand_indices)
        .def("update_with", &SparseTensor::update_with)
        .def("coalesce", &SparseTensor::coalesce)
    ;
}