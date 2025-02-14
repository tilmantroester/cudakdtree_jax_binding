#include <cstdint>
#include <string_view>
#include <unordered_map>

#include <iostream>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;


enum class TraversalMode : int32_t {
  stack_free = 0,
  cct,
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

ffi::Error kdtree_call_impl(ffi::BufferR2<ffi::F32> points,
                            std::optional<ffi::BufferR2<ffi::F32>> queries,
                            xla::ffi::RemainingArgs args,
                            TraversalMode traversal_mode,
                            const int k,
                            ffi::Span<const float> box_size,
                            ffi::ResultBufferR2<ffi::S32> result_idx) {
    return ffi::Error::Success();
}


XLA_FFI_REGISTER_ENUM_ATTR_DECODING(TraversalMode);

XLA_FFI_DEFINE_HANDLER_SYMBOL(kdtree_call, kdtree_call_impl,
                              ffi::Ffi::Bind()
                                    .Arg<ffi::BufferR2<ffi::F32>>()
                                    .OptionalArg<ffi::BufferR2<ffi::F32>>()
                                    .RemainingArgs()
                                    .Attr<TraversalMode>("traversal_mode")
                                    .Attr<int>("k")
                                    .Attr<ffi::Span<const float>>("box_size")
                                    .Ret<ffi::BufferR2<ffi::S32>>());


NB_MODULE(_cudakdtree_interface, m) {
    m.def("registrations", []() {
        nb::dict registrations;
        registrations["kdtree_call"] =
            nb::capsule(reinterpret_cast<void *>(kdtree_call));
        return registrations;
    });

    nb::enum_<TraversalMode>(m, "TraversalMode", nb::is_arithmetic())
        .value("stack_free", TraversalMode::stack_free)
        .value("cct", TraversalMode::cct);
}