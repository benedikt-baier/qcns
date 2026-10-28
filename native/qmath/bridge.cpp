// Stable array ABI: no Python, NumPy or QCNS++ object ownership crosses this boundary.
#include "qmath/matrix.h"
#include "qmath/dynamics/single_transformation.h"
#include "qmath/dynamics/controlled_transformation.h"
#include "qmath/dynamics/cnot.h"
#include "qmath/channels/dephasing.h"
#include "qmath/channels/amplitude_damping.h"
#include "qmath/channels/depolarization.h"
#include "qmath/info/reductions.h"
#include <bit>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <omp.h>

#if defined(_WIN32)
#define QCNS_API extern "C" __declspec(dllexport)
#else
#define QCNS_API extern "C" __attribute__((visibility("default")))
#endif

namespace {
thread_local std::string last_error;
static_assert(std::is_trivially_copyable_v<qmath::Complex>);
static_assert(std::is_standard_layout_v<qmath::Complex>);
static_assert(sizeof(qmath::Complex) == 2 * sizeof(double));
static_assert(offsetof(qmath::Complex, re) == 0);
static_assert(offsetof(qmath::Complex, im) == sizeof(double));

struct Threads {
    int previous;
    explicit Threads(int count) : previous(omp_get_max_threads()) { omp_set_num_threads(count); }
    ~Threads() { omp_set_num_threads(previous); }
};
}

QCNS_API int qcns_qmath_abi() noexcept { return 1; }
QCNS_API const char* qcns_qmath_error() noexcept { return last_error.c_str(); }

// Operations: single, controlled, CNOT, dephasing, amplitude damping,
// symmetric Pauli depolarization, single-subsystem partial trace.
// Indices use Python/MSB ordering. Buffers are native-endian complex128, row-major.
QCNS_API int qcns_qmath_transform(int operation, const void* input, std::size_t input_bytes,
                                 void* output, std::size_t output_bytes, std::size_t dim,
                                 std::size_t target, std::size_t control,
                                 const void* gate, std::size_t gate_bytes,
                                 double probability, int threads) noexcept {
    try {
        last_error.clear();
        if (!input || !output || dim < 2 || !std::has_single_bit(dim) || dim > (1u << 15))
            throw std::invalid_argument("Expected a square 1..15 qubit density matrix");
        const auto n = static_cast<std::size_t>(std::countr_zero(dim));
        if (operation < 0 || operation > 6 || target >= n || threads < 1 || threads > 1024)
            throw std::invalid_argument("Invalid operation, target, or thread count");
        if ((operation == 1 || operation == 2) && (control >= n || control == target))
            throw std::invalid_argument("Control and target must be distinct valid indices");
        const auto out_dim = operation == 6 ? dim / 2 : dim;
        if (input_bytes != dim * dim * sizeof(qmath::Complex) ||
            output_bytes != out_dim * out_dim * sizeof(qmath::Complex))
            throw std::invalid_argument("Buffer sizes do not match density matrix dimensions");
        if (operation <= 1 && (!gate || gate_bytes != 4 * sizeof(qmath::Complex)))
            throw std::invalid_argument("Expected a complex128 2x2 gate");
        if (operation >= 3 && operation <= 5 &&
            (!std::isfinite(probability) || probability < 0 || probability > 1))
            throw std::invalid_argument("Noise probability must be in [0,1]");
        Threads limits(threads);
        qmath::Matrix state(dim);
        std::memcpy(state.data_ptr(), input, input_bytes);
        const auto t = n - 1 - target;
        if (operation <= 1) {
            qmath::Matrix local(2);
            std::memcpy(local.data_ptr(), gate, gate_bytes);
            if (operation == 0) qmath::dynamics::single_transform(t, local, state);
            else qmath::dynamics::controlled_transform_no_c(t, n - 1 - control, local, state);
        } else if (operation == 2) {
            qmath::dynamics::apply_cnot(state, n - 1 - control, t);
        } else if (operation == 3) {
            qmath::channels::apply_dephasing(state, t, probability);
        } else if (operation == 4) {
            qmath::channels::apply_amplitude_damping(state, t, probability);
        } else if (operation == 5) {
            qmath::channels::apply_depolarizing_pauli(state, t, probability);
        } else {
            auto reduced = qmath::info::partial_trace(state, t);
            std::memcpy(output, reduced.data_ptr(), output_bytes);
            return 0;
        }
        std::memcpy(output, state.data_ptr(), output_bytes);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
    } catch (...) {
        last_error = "Unknown QMath kernel exception";
    }
    return -1;
}
