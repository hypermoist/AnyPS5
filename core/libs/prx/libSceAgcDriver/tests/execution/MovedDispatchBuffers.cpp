#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <new>
#include <string>
#include <vector>

// A dispatch whose read-only input V# moved since the resource template was built (a per-frame ring
// allocation) hits the template and rebinds the input: in place from the host import at an offset
// the device can bind, from a snapshot at one it cannot. Each dispatch must read its own input.

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 64;
constexpr std::size_t BlockBytes = 65536;
// The output in the registered block too: a buffer written outside host-imported memory is copied
// back after the dispatch, which keeps the template from being reused.
std::uint32_t* Output = nullptr;

// v_lshlrev_b32 v1, 2, v0; buffer_load_dword v2, v1, s[0:3], 0 offen; s_waitcnt vmcnt(0);
// buffer_store_dword v2, v1, s[4:7], 0 offen; s_endpgm
alignas(256) constexpr std::array<std::uint32_t, 8> CopyCode{0x34020082u, 0xe0301000u, 0x80000201u, 0xbf8c3f70u, 0xe0701000u, 0x80010201u, 0xbf810000u, 0u};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), count * 4u, 0x31016facu};
}

void Copy(AgcDriver::VulkanDevice& device, const std::uint32_t* input) {
    std::fill(Output, Output + Threads, 0xdeadbeefu);
    std::vector<std::uint32_t> userData(8, 0u);
    const auto in = BufferDescriptor(input, Threads);
    const auto out = BufferDescriptor(Output, Threads);
    std::copy(in.begin(), in.end(), userData.begin());
    std::copy(out.begin(), out.end(), userData.begin() + 4);
    const std::span<const std::uint32_t> code(CopyCode.data(), 7);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}

void Check(const std::uint32_t* input, const std::string& what) {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) Require(Output[tid] == input[tid], what + ": thread " + std::to_string(tid) + " stored " + std::to_string(Output[tid]) + ", expected " + std::to_string(input[tid]));
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        auto* block = static_cast<std::uint32_t*>(::operator new(BlockBytes, std::align_val_t{BlockBytes}));
        for (std::size_t index = 0; index < BlockBytes / 4; ++index) block[index] = static_cast<std::uint32_t>(index * 2654435761u);
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, block, BlockBytes, true, true);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
        Output = block + 12288;
        int failures = 0;
        try {
            const auto* first = block;
            const auto* aligned = block + 4096;
            const auto* offset = block + 8193;
            Copy(*device, first);
            Check(first, "the template's input");
            Copy(*device, aligned);
            Check(aligned, "an input moved to an aligned offset");
            Copy(*device, offset);
            Check(offset, "an input moved to an unaligned offset");
            Copy(*device, first);
            Check(first, "the first input again");
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            failures = 1;
        }
        mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, block);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
        ::operator delete(block, std::align_val_t{BlockBytes});
        if (failures != 0) return 1;
        std::puts("moved dispatch buffer tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
