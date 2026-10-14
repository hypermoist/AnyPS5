#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t Stored = 0x12345678u;
constexpr std::uint32_t Untouched = 0xdeadbeefu;

alignas(256) std::array<std::uint32_t, Lanes> First{};
alignas(256) std::array<std::uint32_t, Lanes> Second{};
alignas(256) std::array<std::uint32_t, 4> Table{};
alignas(256) std::array<std::uint32_t, 4> MovedTable{};

alignas(256) constexpr std::array<std::uint32_t, 6> InlineCode{
    0x34020082u, 0x7e0402ffu, 0x12345678u, 0xe0701000u, 0x80000201u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 9> PointerCode{
    0xf4080100u, 0xfa000000u, 0x34020082u, 0x7e0402ffu, 0x12345678u, 0xbf8cc07fu, 0xe0701000u, 0x80010201u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 11> OffsetPointerCode{
    0x80009000u, 0x82018001u, 0xf4080100u, 0xfa000000u, 0x34020082u, 0x7e0402ffu, 0x12345678u, 0xbf8cc07fu, 0xe0701000u, 0x80010201u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 11> MaskedPointerCode{
    0xf4080100u, 0xfa000000u, 0x34020082u, 0x7e0402ffu, 0x12345678u, 0xbf8cc07fu, 0x8705ff05u, 0x3fffffffu, 0xe0701000u, 0x80010201u, 0xbf810000u,
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x30027facu};
}

template <std::size_t CodeWords>
ShaderRecompiler::RecompileRequest Request(AgcDriver::VulkanDevice& device, const std::array<std::uint32_t, CodeWords>& code, std::span<const std::uint32_t> userData) {
    const std::span<const std::uint32_t> span(code);
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0u, {false, false, false}, false, 1};
    return ShaderRecompiler::RecompileRequest{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(span.data()), span, 0, {}},
        {64, 0, userData, compute, std::nullopt, std::nullopt, {}},
        device.Target(),
        {0, 0, 0, 128}
    };
}

std::shared_ptr<const ShaderRecompiler::ResourceCapture> Capture(ShaderRecompiler::RecompileRequest& request, AgcDriver::ShaderMemory& memory, std::vector<ShaderRecompiler::MemoryRegion>& regions) {
    auto capture = memory.Capture(request);
    regions = memory.Regions();
    request.context.memory = regions;
    return capture;
}

void RequireSameResult(const ShaderRecompiler::RecompileResult& fresh, const ShaderRecompiler::RecompileResult& patched) {
    Require(fresh.PipelineVariantId() == patched.PipelineVariantId(), "user word hits: the patched capture chose another pipeline variant");
    Require(fresh.pushConstants == patched.pushConstants, "user word hits: the patched capture's push constants differ from a fresh capture's");
    Require(fresh.bindings.size() == patched.bindings.size(), "user word hits: the patched capture's binding count differs from a fresh capture's");
    for (std::size_t b = 0; b < fresh.bindings.size(); ++b) {
        Require(fresh.bindings[b].role == patched.bindings[b].role && fresh.bindings[b].guestDescriptor == patched.bindings[b].guestDescriptor, "user word hits: binding " + std::to_string(b) + " of the patched capture differs from a fresh capture's");
    }
}

void InlineDescriptor(AgcDriver::VulkanDevice& device) {
    const auto first = BufferDescriptor(First.data(), sizeof(First));
    const auto second = BufferDescriptor(Second.data(), sizeof(Second));
    std::vector<std::uint32_t> firstUser(first.begin(), first.end());
    std::vector<std::uint32_t> secondUser(second.begin(), second.end());

    auto firstRequest = Request(device, InlineCode, firstUser);
    AgcDriver::ShaderMemory firstMemory({});
    std::vector<ShaderRecompiler::MemoryRegion> firstRegions;
    const auto capture = Capture(firstRequest, firstMemory, firstRegions);
    Require(capture->plan != nullptr, "user word hits: the inline-descriptor capture has no plan");
    const auto words = ShaderRecompiler::ComputeDescriptorUserWords(*capture->plan);
    Require(words.patchable.size() >= 2u && words.patchable[0] != 0u && words.patchable[1] != 0u, "user word hits: the inline V#'s address words are not patchable");
    Require(!words.words.empty(), "user word hits: the inline V# lists no patchable descriptor dword");

    auto secondRequest = Request(device, InlineCode, secondUser);
    auto patched = std::make_shared<ShaderRecompiler::ResourceCapture>(*capture);
    std::copy_n(secondUser.begin(), std::min(patched->snapshot.userData.size(), secondUser.size()), patched->snapshot.userData.begin());
    for (const auto& word : words.words) {
        Require(patched->snapshot.buffers.at(word.buffer).dwords.at(word.dword) == firstUser.at(word.user), "user word hits: a patchable dword does not hold its user word in the capture");
        patched->snapshot.buffers.at(word.buffer).dwords.at(word.dword) = secondUser.at(word.user);
    }
    secondRequest.context.memory = firstRegions;
    const auto reused = ShaderRecompiler::Recompile(secondRequest, *patched);

    auto freshRequest = Request(device, InlineCode, secondUser);
    AgcDriver::ShaderMemory freshMemory({});
    std::vector<ShaderRecompiler::MemoryRegion> freshRegions;
    const auto freshCapture = Capture(freshRequest, freshMemory, freshRegions);
    const auto fresh = ShaderRecompiler::Recompile(freshRequest, *freshCapture);
    RequireSameResult(*fresh, *reused);

    First.fill(Untouched);
    Second.fill(Untouched);
    device.Dispatch(*reused, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(InlineCode.data()));
    device.WaitIdle();
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        Require(Second[lane] == Stored, "user word hits: lane " + std::to_string(lane) + " did not store through the moved descriptor");
        Require(First[lane] == Untouched, "user word hits: lane " + std::to_string(lane) + " stored through the stale descriptor");
    }
}

std::vector<std::uint32_t> Pointer(const void* table) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(table));
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u)};
}

void PointerToDescriptor(AgcDriver::VulkanDevice& device) {
    Table = BufferDescriptor(First.data(), sizeof(First));
    MovedTable = Table;
    const auto user = Pointer(Table.data());
    auto request = Request(device, PointerCode, user);
    AgcDriver::ShaderMemory memory({});
    std::vector<ShaderRecompiler::MemoryRegion> regions;
    const auto capture = Capture(request, memory, regions);
    Require(capture->plan != nullptr, "user word hits: the pointer capture has no plan");
    const auto words = ShaderRecompiler::ComputeDescriptorUserWords(*capture->plan);
    Require(words.words.empty(), "user word hits: a pointer the walk reads through was taken for descriptor words");
    Require(words.pointers.size() == 1u && words.pointers[0].low == 0u && words.pointers[0].high == 1u && words.pointers[0].first == 0 && words.pointers[0].end == 16, "user word hits: the table pointer is not one relocatable pointer over its 16 bytes");
    Require(words.patchable.size() >= 2u && words.patchable[0] != 0u && words.patchable[1] != 0u, "user word hits: the table pointer's words are not left out of the key");

    const auto movedUser = Pointer(MovedTable.data());
    auto movedRequest = Request(device, PointerCode, movedUser);
    auto relocated = std::make_shared<ShaderRecompiler::ResourceCapture>(*capture);
    std::copy_n(movedUser.begin(), std::min(relocated->snapshot.userData.size(), movedUser.size()), relocated->snapshot.userData.begin());
    movedRequest.context.memory = regions;
    const auto reused = ShaderRecompiler::Recompile(movedRequest, *relocated);
    auto freshRequest = Request(device, PointerCode, movedUser);
    AgcDriver::ShaderMemory freshMemory({});
    std::vector<ShaderRecompiler::MemoryRegion> freshRegions;
    const auto freshCapture = Capture(freshRequest, freshMemory, freshRegions);
    const auto fresh = ShaderRecompiler::Recompile(freshRequest, *freshCapture);
    RequireSameResult(*fresh, *reused);

    const auto bufferWords = std::count_if(words.tableWords.begin(), words.tableWords.end(), [](const ShaderRecompiler::DescriptorTableWord& word) { return word.kind == 0u && word.index == 0u && word.offset == static_cast<std::int64_t>(word.dword) * 4; });
    const auto flatWords = std::count_if(words.tableWords.begin(), words.tableWords.end(), [](const ShaderRecompiler::DescriptorTableWord& word) { return word.kind == 3u; });
    Require(bufferWords == 4 && flatWords == 4 && words.tableWords.size() == 8u, "user word hits: the table V# is not four buffer words and four flattened slots");
    MovedTable = BufferDescriptor(Second.data(), sizeof(Second));
    auto retargeted = std::make_shared<ShaderRecompiler::ResourceCapture>(*capture);
    std::copy_n(movedUser.begin(), std::min(retargeted->snapshot.userData.size(), movedUser.size()), retargeted->snapshot.userData.begin());
    for (const auto& word : words.tableWords) {
        const auto value = MovedTable.at(static_cast<std::size_t>(word.offset / 4));
        if (word.kind == 3u) retargeted->snapshot.flattenedSrt.at(word.index) = value;
        else retargeted->snapshot.buffers.at(word.index).dwords.at(word.dword) = value;
    }
    auto retargetedRequest = Request(device, PointerCode, movedUser);
    retargetedRequest.context.memory = regions;
    const auto patched = ShaderRecompiler::Recompile(retargetedRequest, *retargeted);
    auto retargetedFreshRequest = Request(device, PointerCode, movedUser);
    AgcDriver::ShaderMemory retargetedMemory({});
    std::vector<ShaderRecompiler::MemoryRegion> retargetedRegions;
    const auto retargetedCapture = Capture(retargetedFreshRequest, retargetedMemory, retargetedRegions);
    RequireSameResult(*ShaderRecompiler::Recompile(retargetedFreshRequest, *retargetedCapture), *patched);
    First.fill(Untouched);
    Second.fill(Untouched);
    device.Dispatch(*patched, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(PointerCode.data()));
    device.WaitIdle();
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        Require(Second[lane] == Stored && First[lane] == Untouched, "user word hits: lane " + std::to_string(lane) + " did not store through the table's new descriptor");
    }
}

void MaskedTableDescriptor(AgcDriver::VulkanDevice& device) {
    Table = BufferDescriptor(First.data(), sizeof(First));
    const auto user = Pointer(Table.data());
    auto request = Request(device, MaskedPointerCode, user);
    AgcDriver::ShaderMemory memory({});
    std::vector<ShaderRecompiler::MemoryRegion> regions;
    const auto capture = Capture(request, memory, regions);
    Require(capture->plan != nullptr, "user word hits: the masked table capture has no plan");
    const auto words = ShaderRecompiler::ComputeDescriptorUserWords(*capture->plan);
    Require(words.pointers.size() == 1u, "user word hits: the masked table pointer is not relocatable");
    const auto masked = std::find_if(words.derivedWords.begin(), words.derivedWords.end(), [](const ShaderRecompiler::DerivedTableWord& word) { return word.offset == 4; });
    Require(masked != words.derivedWords.end() && !masked->sources.empty(), "user word hits: the masked V# word is not a derived table word with its descriptor source");
    Require(std::none_of(words.tableWords.begin(), words.tableWords.end(), [](const ShaderRecompiler::DescriptorTableWord& word) { return word.offset == 4 && word.kind != 3u; }), "user word hits: the masked V# word was taken for a direct descriptor dword");

    MovedTable = BufferDescriptor(Second.data(), sizeof(Second));
    MovedTable[1] |= 4u << 16u;
    const auto movedUser = Pointer(MovedTable.data());
    auto retargeted = std::make_shared<ShaderRecompiler::ResourceCapture>(*capture);
    std::copy_n(movedUser.begin(), std::min(retargeted->snapshot.userData.size(), movedUser.size()), retargeted->snapshot.userData.begin());
    std::vector<std::uint32_t> sources;
    for (const auto& word : words.derivedWords) {
        retargeted->snapshot.flattenedSrt.at(word.flat) = MovedTable.at(static_cast<std::size_t>(word.offset / 4));
        sources.insert(sources.end(), word.sources.begin(), word.sources.end());
    }
    std::sort(sources.begin(), sources.end());
    sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
    AgcDriver::DriverDetail::ReevaluateDescriptorSources(*retargeted, sources, movedUser);
    auto retargetedRequest = Request(device, MaskedPointerCode, movedUser);
    retargetedRequest.context.memory = regions;
    const auto reused = ShaderRecompiler::Recompile(retargetedRequest, *retargeted);
    auto freshRequest = Request(device, MaskedPointerCode, movedUser);
    AgcDriver::ShaderMemory freshMemory({});
    std::vector<ShaderRecompiler::MemoryRegion> freshRegions;
    const auto freshCapture = Capture(freshRequest, freshMemory, freshRegions);
    RequireSameResult(*ShaderRecompiler::Recompile(freshRequest, *freshCapture), *reused);
    First.fill(Untouched);
    Second.fill(Untouched);
    device.Dispatch(*reused, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(MaskedPointerCode.data()));
    device.WaitIdle();
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        Require(Second[lane] == Stored && First[lane] == Untouched, "user word hits: lane " + std::to_string(lane) + " did not store through the re-evaluated descriptor");
    }
}

void OffsetPointer(AgcDriver::VulkanDevice& device) {
    const auto user = Pointer(Table.data());
    auto request = Request(device, OffsetPointerCode, user);
    AgcDriver::ShaderMemory memory({});
    std::vector<ShaderRecompiler::MemoryRegion> regions;
    const auto capture = Capture(request, memory, regions);
    Require(capture->plan != nullptr, "user word hits: the offset pointer capture has no plan");
    const auto words = ShaderRecompiler::ComputeDescriptorUserWords(*capture->plan);
    Require(words.pointers.empty() && words.words.empty() && std::all_of(words.patchable.begin(), words.patchable.end(), [](std::uint8_t patchable) { return patchable == 0u; }), "user word hits: a pointer the shader adds to before reading was left out of the key");
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        InlineDescriptor(*device);
        PointerToDescriptor(*device);
        MaskedTableDescriptor(*device);
        OffsetPointer(*device);
        std::puts("user word descriptor hit tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
