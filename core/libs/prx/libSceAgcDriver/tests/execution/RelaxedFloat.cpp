#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <new>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 64;
constexpr std::size_t BlockBytes = 65536;

alignas(256) constexpr std::array<std::uint32_t, 9> SquareCode{0x34020082u, 0xe0301000u, 0x80000201u, 0xbf8c3f70u, 0x10040502u, 0xe0701000u, 0x80010201u, 0xbf810000u, 0u};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), count * 4u, 0x31016facu};
}

ShaderRecompiler::RecompileResult Compile(AgcDriver::VulkanDevice& device, const std::uint32_t* input, const std::uint32_t* output, bool relaxed) {
    std::vector<std::uint32_t> userData(8, 0u);
    const auto in = BufferDescriptor(input, Threads);
    const auto out = BufferDescriptor(output, Threads);
    std::copy(in.begin(), in.end(), userData.begin());
    std::copy(out.begin(), out.end(), userData.begin() + 4);
    const std::span<const std::uint32_t> code(SquareCode.data(), 8);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    request.target.relaxedFloat = relaxed;
    return ShaderRecompiler::Recompile(request);
}

bool IsNan(std::uint32_t bits) {
    return (bits & 0x7fffffffu) > 0x7f800000u;
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        auto* block = static_cast<std::uint32_t*>(::operator new(BlockBytes, std::align_val_t{BlockBytes}));
        std::memset(block, 0, BlockBytes);
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, block, BlockBytes, true, true);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
        int failures = 0;
        try {
            auto* input = block;
            auto* output = block + 4096;
            const std::array<float, 6> normals{1.5f, -2.0f, 0.375f, 1024.0f, -0.0f, 3.0e19f};
            for (std::uint32_t tid = 0; tid < Threads; ++tid) input[tid] = std::bit_cast<std::uint32_t>(normals[tid % normals.size()]);
            input[7] = 0x7f812345u;
            input[8] = 0xffc00001u;
            input[9] = 0x7f800000u;
            const auto exact = Compile(*device, input, output, false);
            const auto relaxed = Compile(*device, input, output, true);
            Require(relaxed.spirv.size() < exact.spirv.size(), "relaxed float: the module kept the NaN result emulation (" + std::to_string(relaxed.spirv.size()) + " words, exact " + std::to_string(exact.spirv.size()) + ")");
            std::fill(output, output + Threads, 0xdeadbeefu);
            device->Dispatch(relaxed, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(SquareCode.data()));
            device->WaitIdle();
            for (std::uint32_t tid = 0; tid < Threads; ++tid) {
                const float value = std::bit_cast<float>(input[tid]);
                if (tid == 7 || tid == 8) {
                    Require(IsNan(output[tid]), "relaxed float: thread " + std::to_string(tid) + " squared a NaN into a non-NaN");
                    continue;
                }
                const auto wanted = std::bit_cast<std::uint32_t>(value * value);
                Require(output[tid] == wanted, "relaxed float: thread " + std::to_string(tid) + " stored " + std::to_string(output[tid]) + ", expected " + std::to_string(wanted));
            }
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            failures = 1;
        }
        mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, block);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
        ::operator delete(block, std::align_val_t{BlockBytes});
        if (failures != 0) return 1;
        std::puts("relaxed float tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
