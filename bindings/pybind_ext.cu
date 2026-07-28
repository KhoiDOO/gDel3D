#include <torch/extension.h>
#include <vector>
#include <cstring>
#include "GpuDelaunay.h"

torch::Tensor tetrahedralize_cuda(const torch::Tensor& points) {
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must be an N x 3 tensor");
    TORCH_CHECK(points.size(0) >= 4, "Need at least 4 points for tetrahedralization");
    
    torch::Tensor points_contig = points.is_cuda() ? points.contiguous().cpu() : points.contiguous();
    int num_points = points_contig.size(0);
    
    Point3HVec pointVec;
    pointVec.resize(num_points);
    
    if (points_contig.dtype() == torch::kFloat32) {
        const float* pts_ptr = points_contig.data_ptr<float>();
        for (int i = 0; i < num_points; ++i) {
            pointVec[i]._p[0] = static_cast<RealType>(pts_ptr[i * 3 + 0]);
            pointVec[i]._p[1] = static_cast<RealType>(pts_ptr[i * 3 + 1]);
            pointVec[i]._p[2] = static_cast<RealType>(pts_ptr[i * 3 + 2]);
        }
    } else if (points_contig.dtype() == torch::kFloat64) {
        const double* pts_ptr = points_contig.data_ptr<double>();
        for (int i = 0; i < num_points; ++i) {
            pointVec[i]._p[0] = static_cast<RealType>(pts_ptr[i * 3 + 0]);
            pointVec[i]._p[1] = static_cast<RealType>(pts_ptr[i * 3 + 1]);
            pointVec[i]._p[2] = static_cast<RealType>(pts_ptr[i * 3 + 2]);
        }
    } else {
        TORCH_CHECK(false, "points must be float32 or float64");
    }
    
    std::vector<int64_t> valid_tets;
    {
        GDelOutput output;
        GDelParams params;
        GpuDel gpuDel(params);
        gpuDel.compute(pointVec, &output);
        
        int num_raw_tets = output.tetVec.size();
        for (int i = 0; i < num_raw_tets; ++i) {
            const Tet& tet = output.tetVec[i];
            if (tet._v[0] < num_points && tet._v[1] < num_points && 
                tet._v[2] < num_points && tet._v[3] < num_points) {
                valid_tets.push_back(static_cast<int64_t>(tet._v[0]));
                valid_tets.push_back(static_cast<int64_t>(tet._v[1]));
                valid_tets.push_back(static_cast<int64_t>(tet._v[2]));
                valid_tets.push_back(static_cast<int64_t>(tet._v[3]));
            }
        }
    }
    
    int num_tets = valid_tets.size() / 4;
    torch::Tensor result = torch::empty({num_tets, 4}, torch::dtype(torch::kInt64).device(torch::kCPU));
    if (num_tets > 0) {
        std::memcpy(result.data_ptr<int64_t>(), valid_tets.data(), valid_tets.size() * sizeof(int64_t));
    }
    return result.to(points.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tetrahedralize_cuda", &tetrahedralize_cuda, "Compute 3D Delaunay tetrahedralization on CUDA using gDel3D");
}
