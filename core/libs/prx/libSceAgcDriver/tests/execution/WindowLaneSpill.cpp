#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
alignas(256) std::array<std::uint32_t, Threads> Output{};

// v_mov_b32 v1, v0; s_mov_b32 m0, s12; v_movreld_b32 v2, v0 (the relative-addressing window starts at
// v2); v_writelane_b32 v3, s4..s7, 0..3 (the output V# spilled into a window register);
// s_mov_b32 s4..s7, 0; v_readlane_b32 s8..s11, v3, 0..3; buffer_store_dword v0, v1, s[8:11] idxen;
// s_endpgm.
alignas(256) constexpr std::array<std::uint32_t, 26> Code{
    0x7e020300, 0xbefc030c, 0x7e048500, 0xd7610003, 0x00010004, 0xd7610003, 0x00010205, 0xd7610003, 0x00010406,
    0xd7610003, 0x00010607, 0xbe840380, 0xbe850380, 0xbe860380, 0xbe870380, 0xd7600008, 0x00010103, 0xd7600009,
    0x00010303, 0xd760000a, 0x00010503, 0xd760000b, 0x00010703, 0xe0702000, 0x80020001, 0xbf810000,
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (4u << 16u), count, 0x11016facu};
}

void Run(AgcDriver::VulkanDevice& device) {
    Output.fill(0xdeadbeefu);
    std::vector<std::uint32_t> userData(16, 0u);
    const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size()));
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    const std::span<const std::uint32_t> code(Code);
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
        Require(Output[tid] == tid, "window lane spill: lane " + std::to_string(tid) + " stored " + std::to_string(Output[tid]) + " through the V# read back from a window register");
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
        std::puts("window lane spill tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
