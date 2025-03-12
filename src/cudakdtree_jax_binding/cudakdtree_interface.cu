#include <cstdint>
#include <string_view>
#include <unordered_map>

#include <iostream>

#include "cukd/builder.h"
#include "cukd/knn.h"

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;


enum class TraversalMode : int32_t {
  stack_free = 0,
  cct,
  stack_free_bounds_tracking,
};

enum class CandidateList : int32_t {
  fixed_list = 0,
  heap,
};

std::ostream& operator<<(std::ostream& os, const TraversalMode mode)
{
    switch(mode)
    {
        case TraversalMode::stack_free:
            os << "stack_free";
            break;
        case TraversalMode::cct:
            os << "cct";
            break;
    }
    return os;
}

template<typename point_t>
struct OrderedPoint {
    point_t position;
    int     idx;
};

template<typename point_t>
struct OrderedPoint_traits : public cukd::default_data_traits<point_t>
{
    using data_t = OrderedPoint<point_t>;
    using point_traits = cukd::point_traits<point_t>;
    using scalar_t = typename point_traits::scalar_t;

    static inline __device__ __host__
    const point_t &get_point(const data_t &data) { return data.position; }

    static inline __device__ __host__
    scalar_t  get_coord(const data_t &data, int dim)
    { return cukd::get_coord(get_point(data), dim); }

    enum { has_explicit_dim = false };
    static inline __device__ int get_dim(const data_t &) { return -1; }
};


using FixedList9T = cukd::FixedCandidateList<9>;
using FixedList25T = cukd::FixedCandidateList<25>;
using Heap25T = cukd::HeapCandidateList<25>;

using DataD2T = OrderedPoint<float2>;
using DataTraitsD2T = OrderedPoint_traits<float2>;

template<TraversalMode M, typename _CandidateListT, typename _DataT, typename _DataTraitsT> struct KDTreeSpec;

// TODO: maybe there's a nicer way to do this
template<typename _CandidateListT, typename _DataT, typename _DataTraitsT>
struct KDTreeSpec<TraversalMode::stack_free_bounds_tracking, _CandidateListT, _DataT, _DataTraitsT>{
    using CandidateListT = _CandidateListT;
    using DataT = _DataT;
    using DataTraitsT = _DataTraitsT;
    using PointT = typename _DataTraitsT::point_t;

    static const TraversalMode transversal_mode = TraversalMode::stack_free_bounds_tracking;

    static inline __device__
    float travelsal_func(CandidateListT &result,
                         PointT query,
                         const cukd::box_t<PointT> world_bounds,
                         const DataT *nodes,
                         const int n_nodes,
                         const PointT *periodic_box_size)
    {
        return cukd::stackFreeBoundsTracking::knn<CandidateListT,DataT,DataTraitsT>(result, query, world_bounds, nodes, n_nodes, periodic_box_size);
    };
};

template<typename _CandidateListT, typename _DataT, typename _DataTraitsT>
struct KDTreeSpec<TraversalMode::cct, _CandidateListT, _DataT, _DataTraitsT>{
    using CandidateListT = _CandidateListT;
    using DataT = _DataT;
    using DataTraitsT = _DataTraitsT;
    using PointT = typename _DataTraitsT::point_t;

    static const TraversalMode transversal_mode = TraversalMode::cct;

    static inline __device__
    float travelsal_func(CandidateListT &result,
                         PointT query,
                         const cukd::box_t<PointT> world_bounds,
                         const DataT *nodes,
                         const int n_nodes,
                         const PointT *periodic_box_size)
    {
        return cukd::cct::knn<CandidateListT,DataT,DataTraitsT>(result, query, world_bounds, nodes, n_nodes);
    };
};

using StackFreeBoundsTrackingD2K9FixedListT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, FixedList9T, DataD2T, DataTraitsD2T>;


template<typename KDTreeSpecT>
__global__ void knn_kernel(int32_t* results,
                           typename KDTreeSpecT::PointT* queries,
                           int n_queries,
                           typename KDTreeSpecT::DataT* nodes,
                           int n_nodes,
                           float max_radius,
                           const cukd::box_t<typename KDTreeSpecT::PointT>* world_bounds,
                           const typename KDTreeSpecT::PointT *periodic_box_size)
{
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= n_queries) return;

    typename KDTreeSpecT::CandidateListT result(max_radius);
    KDTreeSpecT::travelsal_func(result, queries[tid], *world_bounds, nodes, n_nodes, periodic_box_size);

    const int k = KDTreeSpecT::CandidateListT::num_k;
    for(int i=0; i < k; i++){
        const int idx = result.get_pointID(i);
        if(idx >= 0 && idx < n_nodes)
            results[tid*k+i] = nodes[idx].idx;
        else
            results[tid*k+i] = -1;
    };
}

template<typename PointT>
__global__ void copy_positions_and_idx_kernel(OrderedPoint<PointT> *points,
                                              PointT *positions,
                                              int n_points)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= n_points) return;
    points[tid].position = positions[tid];
    points[tid].idx = tid;
}

template<typename KDTreeSpecT>
void cudakdtree_knn_wrapper(cudaStream_t stream,
                            typename KDTreeSpecT::PointT* positions, int n_points, int n_dim,
                            typename KDTreeSpecT::PointT* queries, int n_queries,
                            int32_t* results,
                            const float max_radius,
                            const auto box_size)
{
    using PointT = typename KDTreeSpecT::PointT;
    using DataT = typename KDTreeSpecT::DataT;
    using DataTraitsT = typename KDTreeSpecT::DataTraitsT;

    // Make copy of positions and attach a tag to be able to track the order of the points
    DataT *ordered_points;
    // use cudaMallocAsync instead?
    CUKD_CUDA_CHECK(cudaMallocManaged((void**)&ordered_points,n_points*sizeof(*ordered_points)));
    int bs = 128;
    int nb = cukd::divRoundUp(n_points, bs);
    copy_positions_and_idx_kernel<<<nb, bs, 0, stream>>>(ordered_points, positions, n_points);
    CUKD_CUDA_SYNC_CHECK()

    // Allocate box for world bounds tht will be filled in when building the tree
    cukd::box_t<PointT>* bounds_ptr;
    CUKD_CUDA_CHECK(cudaMallocManaged((void**)&bounds_ptr, sizeof(cukd::box_t<PointT>)));
    // Build the KDTree from the provided points
    cukd::buildTree<DataT, DataTraitsT>(ordered_points, n_points, bounds_ptr, stream);
    CUKD_CUDA_SYNC_CHECK()

    // If a box size is specified, pass that to the traversal code. Else pass a nullptr, which means no periodic boundaries.
    PointT *box_size_ptr = nullptr;
    if(box_size.size() > 0) {
        CUKD_CUDA_CHECK(cudaMallocManaged((void**)&box_size_ptr, sizeof(PointT)));
        for(int i = 0; i < box_size.size(); i++) {
            cukd::set_coord(*box_size_ptr, i, box_size[i]);
        }
    }

    // Perform the kNN search
    bs = 128;
    nb = cukd::divRoundUp(n_queries, bs);
    knn_kernel<KDTreeSpecT><<<nb, bs, 0, stream>>>(
        results, queries, n_queries, ordered_points, n_points,
        max_radius, bounds_ptr, box_size_ptr);
    CUKD_CUDA_SYNC_CHECK()
    
    cudaFree(ordered_points);
    cudaFree(bounds_ptr);
    cudaFree(box_size_ptr);
}


template<typename PointT>
auto get_queries(std::optional<ffi::BufferR2<ffi::F32>> queries, ffi::BufferR2<ffi::F32> points)
{
    if(queries.has_value()) {
        return std::make_pair((*queries).typed_data(), (*queries).dimensions()[0]);
    }
    else {
        return std::make_pair(points.typed_data(), points.dimensions()[0]);
    }
}

#define CALL_CUDA_KDTREE(KDTreeSpecT, PointT)                               \
    cudakdtree_knn_wrapper<KDTreeSpecT>(                                    \
        stream,                                                             \
        reinterpret_cast<PointT*>(points.typed_data()), n_points, n_dim,    \
        reinterpret_cast<PointT*>(tree_queries_data), n_tree_queries,       \
        result_idx->typed_data(),                                           \
        max_radius,                                                         \
        box_size                                                            \
    );

#define INVALID_ARGUMENT_ERROR                                                      \
    ffi::Error::InvalidArgument(                                                    \
        "The combination of dimensions, number of neighbours, transversal mode, "   \
        "and type of candidate list is not supported."                              \
    );


ffi::Error kdtree_call_impl(cudaStream_t stream,
                            ffi::BufferR2<ffi::F32> points,
                            std::optional<ffi::BufferR2<ffi::F32>> queries,
                            TraversalMode traversal_mode,
                            CandidateList candidate_list,
                            const int k,
                            const float max_radius,
                            ffi::Span<const float> box_size,
                            ffi::ResultBufferR2<ffi::S32> result_idx)
{
    const int n_dim = points.dimensions()[1];
    const int n_points = points.dimensions()[0];

    if(box_size.size() > 0 && (traversal_mode != TraversalMode::stack_free_bounds_tracking)) {
        return ffi::Error::InvalidArgument(
            "The transversal mode does not support periodic boundaries."
        );
    }

    // TODO: there is probably a better way to do this with templates and macros
    if(n_dim == 2) {
        using PointT = float2;
        using DataT = OrderedPoint<PointT>;
        using DataTraitsT = OrderedPoint_traits<PointT>;

        auto [tree_queries_data, n_tree_queries] = get_queries<PointT>(queries, points);

        if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::fixed_list && k == 8) {
            using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::FixedCandidateList<8>, DataT, DataTraitsT>;
            CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        }
        else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::fixed_list && k == 9) {
            using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::FixedCandidateList<9>, DataT, DataTraitsT>;
            CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        }
        else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::fixed_list && k == 16) {
            using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::FixedCandidateList<16>, DataT, DataTraitsT>;
            CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        }
        else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::fixed_list && k == 25) {
            using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::FixedCandidateList<25>, DataT, DataTraitsT>;
            CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        }
        // else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::heap && k == 25) {
        //     using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::HeapCandidateList<25>, DataT, DataTraitsT>;
        //     CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        // }
        else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::fixed_list && k == 61) {
            using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::FixedCandidateList<61>, DataT, DataTraitsT>;
            CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        }
        // else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::heap && k == 61) {
        //     using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::HeapCandidateList<61>, DataT, DataTraitsT>;
        //     CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        // }
        else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::fixed_list && k == 128) {
            using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::FixedCandidateList<128>, DataT, DataTraitsT>;
            CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        }
        // else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::heap && k == 128) {
        //     using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::HeapCandidateList<128>, DataT, DataTraitsT>;
        //     CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        // }
        else
            return INVALID_ARGUMENT_ERROR
    }
    else if(n_dim == 3) {
        using PointT = float3;
        using DataT = OrderedPoint<PointT>;
        using DataTraitsT = OrderedPoint_traits<PointT>;

        auto [tree_queries_data, n_tree_queries] = get_queries<PointT>(queries, points);

        if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::fixed_list && k == 8) {
            using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::FixedCandidateList<8>, DataT, DataTraitsT>;
            CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        }
        else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::fixed_list && k == 16) {
            using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::FixedCandidateList<16>, DataT, DataTraitsT>;
            CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        }
        else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::fixed_list && k == 27) {
            using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::FixedCandidateList<27>, DataT, DataTraitsT>;
            CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        }
        // else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::heap && k == 27) {
        //     using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::HeapCandidateList<27>, DataT, DataTraitsT>;
        //     CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        // }
        else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::fixed_list && k == 33) {
            using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::FixedCandidateList<33>, DataT, DataTraitsT>;
            CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        }
        else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::fixed_list && k == 100) {
            using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::FixedCandidateList<100>, DataT, DataTraitsT>;
            CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        }
        // else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::heap && k == 100) {
        //     using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::HeapCandidateList<100>, DataT, DataTraitsT>;
        //     CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        // }
        else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::fixed_list && k == 128) {
            using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::FixedCandidateList<128>, DataT, DataTraitsT>;
            CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        }
        // else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::heap && k == 128) {
        //     using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::HeapCandidateList<128>, DataT, DataTraitsT>;
        //     CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        // }
        // else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::fixed_list && k == 179) {
        //     using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::FixedCandidateList<179>, DataT, DataTraitsT>;
        //     CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        // }
        // else if(traversal_mode == TraversalMode::stack_free_bounds_tracking && candidate_list == CandidateList::heap && k == 179) {
        //     using KDTreeSpecT = KDTreeSpec<TraversalMode::stack_free_bounds_tracking, cukd::HeapCandidateList<179>, DataT, DataTraitsT>;
        //     CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        // }
        else if(traversal_mode == TraversalMode::cct && candidate_list == CandidateList::fixed_list && k == 27) {
            using KDTreeSpecT = KDTreeSpec<TraversalMode::cct, cukd::FixedCandidateList<27>, DataT, DataTraitsT>;
            CALL_CUDA_KDTREE(KDTreeSpecT, PointT)
        }
        else
            return INVALID_ARGUMENT_ERROR
    }
    else
    {
        return ffi::Error::InvalidArgument(
            "The combination of dimensions, number of neighbours, transversal mode, "
            "and type of candidate list is not supported."
        );
    }
    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error(
            XLA_FFI_Error_Code_INTERNAL,
            std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}


XLA_FFI_REGISTER_ENUM_ATTR_DECODING(TraversalMode);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(CandidateList);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kdtree_call, kdtree_call_impl,
                              ffi::Ffi::Bind()
                                    .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                    .Arg<ffi::BufferR2<ffi::F32>>()
                                    .OptionalArg<ffi::BufferR2<ffi::F32>>()
                                    .Attr<TraversalMode>("traversal_mode")
                                    .Attr<CandidateList>("candidate_list")
                                    .Attr<int>("k")
                                    .Attr<float>("max_radius")
                                    .Attr<ffi::Span<const float>>("box_size")
                                    .Ret<ffi::BufferR2<ffi::S32>>(),
                                    {xla::ffi::Traits::kCmdBufferCompatible});


NB_MODULE(_cudakdtree_interface, m) {
    m.def("registrations", []() {
        nb::dict registrations;
        registrations["kdtree_call"] =
            nb::capsule(reinterpret_cast<void *>(kdtree_call));
        return registrations;
    });

    nb::enum_<TraversalMode>(m, "TraversalMode", nb::is_arithmetic())
        .value("stack_free", TraversalMode::stack_free)
        .value("cct", TraversalMode::cct)
        .value("stack_free_bounds_tracking", TraversalMode::stack_free_bounds_tracking);
    
    nb::enum_<CandidateList>(m, "CandidateList", nb::is_arithmetic())
        .value("fixed_list", CandidateList::fixed_list)
        .value("heap", CandidateList::heap);
}