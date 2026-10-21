#include "SceTypes.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

extern "C" {
int APS5_VABI scePthreadCreate(Pthread* thread, const PthreadAttr* attr, PthreadEntry entry, void* arg, const char* name);
int APS5_VABI scePthreadJoin(Pthread thread, void** retval);
int APS5_VABI scePthreadGetaffinity(Pthread thread, KernelCpumask* mask);
Pthread APS5_VABI scePthreadSelf();
int APS5_VABI scePthreadAttrInit(PthreadAttr* attr);
int APS5_VABI scePthreadAttrDestroy(PthreadAttr* attr);
int APS5_VABI scePthreadAttrSetaffinity(PthreadAttr* attr, KernelCpumask mask);
int APS5_VABI scePthreadAttrGetaffinity(const PthreadAttr* attr, KernelCpumask* mask);
int APS5_VABI pthread_attr_getaffinity_np_nid_postfix(const PthreadAttr* attr, std::size_t cpusetSize, void* cpuset);
int APS5_VABI pthread_attr_setaffinity_np_nid_postfix(PthreadAttr* attr, std::size_t cpusetSize, const void* cpuset);
}

static void Require(bool value) { if (!value) std::abort(); }

static constexpr int GUEST_EINVAL = 22;
static constexpr int GUEST_ERANGE = 34;
static constexpr std::size_t CPUSET_BYTES = 32;

using Cpuset = std::array<std::uint8_t, CPUSET_BYTES + 8>;

static bool Filled(const Cpuset& set, std::size_t begin, std::size_t end, std::uint8_t value) {
    for (std::size_t i = begin; i < end; ++i)
        if (set[i] != value) return false;
    return true;
}

static void* APS5_VABI ReportAffinity(void* arg) {
    Require(scePthreadGetaffinity(scePthreadSelf(), static_cast<KernelCpumask*>(arg)) == 0);
    return nullptr;
}

static void CheckArguments() {
    Cpuset set{};
    PthreadAttr empty = nullptr;
    Require(pthread_attr_getaffinity_np_nid_postfix(nullptr, CPUSET_BYTES, set.data()) == GUEST_EINVAL);
    Require(pthread_attr_getaffinity_np_nid_postfix(&empty, CPUSET_BYTES, set.data()) == GUEST_EINVAL);
    Require(pthread_attr_setaffinity_np_nid_postfix(nullptr, CPUSET_BYTES, set.data()) == GUEST_EINVAL);
    Require(pthread_attr_setaffinity_np_nid_postfix(&empty, CPUSET_BYTES, set.data()) == GUEST_EINVAL);
    PthreadAttr attr = nullptr;
    Require(scePthreadAttrInit(&attr) == 0);
    Require(pthread_attr_getaffinity_np_nid_postfix(&attr, CPUSET_BYTES - 1, set.data()) == GUEST_ERANGE);
    Require(pthread_attr_setaffinity_np_nid_postfix(&attr, CPUSET_BYTES - 1, set.data()) == GUEST_ERANGE);
    set[0] = 1;
    set[CPUSET_BYTES + 3] = 1;
    Require(pthread_attr_setaffinity_np_nid_postfix(&attr, set.size(), set.data()) == GUEST_EINVAL);
    Require(pthread_attr_setaffinity_np_nid_postfix(&attr, 0, set.data()) == 0);
    Require(pthread_attr_setaffinity_np_nid_postfix(&attr, CPUSET_BYTES, nullptr) == 0);
    Require(scePthreadAttrDestroy(&attr) == 0);
}

static void CheckRoundTrip() {
    PthreadAttr attr = nullptr;
    Require(scePthreadAttrInit(&attr) == 0);
    Cpuset set{};
    set.fill(0x5A);
    Require(pthread_attr_getaffinity_np_nid_postfix(&attr, set.size(), set.data()) == 0);
    Require(Filled(set, 0, CPUSET_BYTES, 0xFF) && Filled(set, CPUSET_BYTES, set.size(), 0));

    set.fill(0);
    set[0] = 0x05;
    set[1] = 0x01;
    Require(pthread_attr_setaffinity_np_nid_postfix(&attr, set.size(), set.data()) == 0);
    KernelCpumask mask = 0;
    Require(scePthreadAttrGetaffinity(&attr, &mask) == 0 && mask == 0x105);
    set.fill(0x5A);
    Require(pthread_attr_getaffinity_np_nid_postfix(&attr, CPUSET_BYTES, set.data()) == 0);
    Require(set[0] == 0x05 && set[1] == 0x01 && Filled(set, 2, CPUSET_BYTES, 0) && Filled(set, CPUSET_BYTES, set.size(), 0x5A));

    Require(scePthreadAttrSetaffinity(&attr, 0x3) == 0);
    Require(pthread_attr_getaffinity_np_nid_postfix(&attr, CPUSET_BYTES, set.data()) == 0);
    Require(set[0] == 0x03 && Filled(set, 1, CPUSET_BYTES, 0));

    Require(pthread_attr_setaffinity_np_nid_postfix(&attr, 0, nullptr) == 0);
    Require(pthread_attr_getaffinity_np_nid_postfix(&attr, CPUSET_BYTES, set.data()) == 0);
    Require(Filled(set, 0, CPUSET_BYTES, 0xFF));
    Require(scePthreadAttrDestroy(&attr) == 0);
}

static void CheckThreadCreation() {
    PthreadAttr attr = nullptr;
    Require(scePthreadAttrInit(&attr) == 0);
    Cpuset set{};
    set[0] = 0x02;
    Require(pthread_attr_setaffinity_np_nid_postfix(&attr, CPUSET_BYTES, set.data()) == 0);
    KernelCpumask reported = 0;
    Pthread thread = nullptr;
    Require(scePthreadCreate(&thread, &attr, ReportAffinity, &reported, "attr_affinity") == 0);
    Require(scePthreadJoin(thread, nullptr) == 0);
    Require(reported == 0x2);
    Require(scePthreadAttrDestroy(&attr) == 0);
}

int main() {
    CheckArguments();
    CheckRoundTrip();
    CheckThreadCreation();
    return 0;
}
