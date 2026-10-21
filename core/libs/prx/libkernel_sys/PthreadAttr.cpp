#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libkernel/Pthread/include/Pthread.hpp"

namespace {
constexpr int GuestEinval = 22;
constexpr int GuestErange = 34;
constexpr std::size_t KernelCpusetBytes = 32;
}

extern "C" {

int APS5_VABI pthread_attr_getaffinity_np_nid_postfix(const PthreadAttr* attr, std::size_t cpusetSize, void* cpuset) {
    if (!attr || !*attr) return GuestEinval;
    if (cpusetSize < KernelCpusetBytes) return GuestErange;
    if (!cpuset) throw std::runtime_error(std::string(__func__) + ": null cpuset");
    auto* bytes = static_cast<std::uint8_t*>(cpuset);
    if ((*attr)->_affinityExplicit) {
        std::memset(bytes, 0, KernelCpusetBytes);
        std::memcpy(bytes, &(*attr)->_affinity, sizeof(KernelCpumask));
    } else std::memset(bytes, 0xFF, KernelCpusetBytes);
    std::memset(bytes + KernelCpusetBytes, 0, cpusetSize - KernelCpusetBytes);
    return 0;
}

int APS5_VABI pthread_attr_setaffinity_np_nid_postfix(PthreadAttr* attr, std::size_t cpusetSize, const void* cpuset) {
    if (!attr || !*attr) return GuestEinval;
    if (cpusetSize == 0 || !cpuset) {
        (*attr)->_affinity = DEFAULT_THREAD_AFFINITY;
        (*attr)->_affinityExplicit = false;
        return 0;
    }
    if (cpusetSize < KernelCpusetBytes) return GuestErange;
    const auto* bytes = static_cast<const std::uint8_t*>(cpuset);
    if (std::any_of(bytes + KernelCpusetBytes, bytes + cpusetSize, [](std::uint8_t byte) { return byte != 0; })) return GuestEinval;
    std::memcpy(&(*attr)->_affinity, bytes, sizeof(KernelCpumask));
    (*attr)->_affinityExplicit = true;
    return 0;
}

}
