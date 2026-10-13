#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstdio>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Spilled = 41;
alignas(256) std::array<std::uint32_t, Threads> Output{};

// v_mov_b32 v1, v0; s_cmp_eq_u32 s12, 0; s_cbranch_scc1 +3; s_add_u32 s10, s8, 1;
// v_writelane_b32 v3, s10, 0 (a value made inside the branch, spilled to lane 0); join:
// v_readlane_b32 s9, v3, 0; v_mov_b32 v2, s9; buffer_store_dword v2, v1, s[4:7] idxen; s_endpgm.
// The read's only lane write is on one path, so that write's value does not dominate the read.
alignas(256) constexpr std::array<std::uint32_t, 12> Code{
    0x7e020300, 0xbf06800c, 0xbf850003, 0x800a8108, 0xd7610003, 0x0001000a, 0xd7600009,
    0x00010103, 0x7e040209, 0xe0702000, 0x80010201, 0xbf810000,
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (4u << 16u), count, 0x11016facu};
}

void Run(AgcDriver::VulkanDevice& device) {
    const std::span<const std::uint32_t> code(Code);
    Output.fill(0xdeadbeefu);
    std::vector<std::uint32_t> userData(16, 0u);
    const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size()));
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    userData[8] = Spilled;
    userData[12] = 1;
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}

void Check() {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        Require(Output[tid] == Spilled + 1u, "branch lane spill: lane " + std::to_string(tid) + " stored " + std::to_string(Output[tid]) + " instead of the value spilled inside the branch");
    }
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (device->Target().subgroupSize < Threads) {
            std::printf("skipped, subgroup size %u cannot hold a wave32\n", device->Target().subgroupSize);
            return VulkanTestSkipped;
        }
        Run(*device);
        Check();
        std::puts("branch lane spill tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
