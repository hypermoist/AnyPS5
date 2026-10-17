#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace AgcDriver::Graphics {

// APS5_DEBUG_LABELS=1: a copy of `spirv` whose entry function is named `name` (its OpName) and whose
// source file is `name`.spv (an OpString + OpSource before the names), so external GPU profilers
// that read SPIR-V debug info list the guest program; empty when off or when the module has no
// OpName for its entry function (the caller then keeps the original words).
inline std::vector<std::uint32_t> DebugNamedSpirv(std::span<const std::uint32_t> spirv, const char* name) {
    static const bool enabled = std::getenv("APS5_DEBUG_LABELS") != nullptr;
    if (!enabled || spirv.size() < 5) return {};
    const auto literal = [](const std::string& text) {
        std::vector<std::uint32_t> words((text.size() + 4) / 4, 0u);
        std::memcpy(words.data(), text.data(), text.size());
        return words;
    };
    constexpr std::uint32_t OpSource = 3, OpString = 7, OpName = 5, OpEntryPoint = 15;
    std::uint32_t entry = 0;
    std::size_t firstName = 0;
    for (std::size_t at = 5; at < spirv.size();) {
        const auto count = spirv[at] >> 16;
        const auto opcode = spirv[at] & 0xffffu;
        if (count == 0 || at + count > spirv.size()) return {};
        if (opcode == OpEntryPoint && entry == 0 && count >= 3) entry = spirv[at + 2];
        if (opcode == OpName && firstName == 0) firstName = at;
        at += count;
    }
    if (entry == 0 || firstName == 0) return {};
    const auto nameWords = literal(name);
    const auto fileWords = literal(std::string(name) + ".spv");
    const auto fileId = spirv[3];
    std::vector<std::uint32_t> out(spirv.begin(), spirv.begin() + static_cast<std::ptrdiff_t>(firstName));
    out[3] = fileId + 1u;
    out.push_back(static_cast<std::uint32_t>((2u + fileWords.size()) << 16) | OpString);
    out.push_back(fileId);
    out.insert(out.end(), fileWords.begin(), fileWords.end());
    out.push_back((4u << 16) | OpSource);
    out.push_back(0u);
    out.push_back(0u);
    out.push_back(fileId);
    for (std::size_t at = firstName; at < spirv.size();) {
        const auto count = spirv[at] >> 16;
        if ((spirv[at] & 0xffffu) == OpName && spirv[at + 1] == entry) {
            out.push_back(static_cast<std::uint32_t>((2u + nameWords.size()) << 16) | OpName);
            out.push_back(entry);
            out.insert(out.end(), nameWords.begin(), nameWords.end());
        } else {
            out.insert(out.end(), spirv.begin() + static_cast<std::ptrdiff_t>(at), spirv.begin() + static_cast<std::ptrdiff_t>(at + count));
        }
        at += count;
    }
    return out;
}

}
