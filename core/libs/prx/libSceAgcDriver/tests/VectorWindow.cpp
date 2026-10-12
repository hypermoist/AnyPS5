#include "Recompiler.hpp"
#include <spirv/unified1/spirv.hpp>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ShaderRecompiler;

constexpr std::uint32_t MoveM0Two = 0xbefc0382u;
constexpr std::uint32_t MoveV4One = 0x7e0802f2u;
constexpr std::uint32_t MoveV5Two = 0x7e0a02f4u;
constexpr std::uint32_t MoveV6Four = 0x7e0c02f6u;
constexpr std::uint32_t MovrelsV1V4 = 0x7e028704u;
constexpr std::uint32_t MovreldV4V1 = 0x7e088501u;
constexpr std::uint32_t MoveV1V5 = 0x7e020305u;
constexpr std::uint32_t EndProgram = 0xbf810000u;

struct Shape {
    std::size_t functionArrays = 0;
    std::uint32_t arrayLength = 0;
    std::size_t accessChains = 0;
    std::size_t selects = 0;
};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::uint32_t> RecompileProgram(std::span<const std::uint32_t> code) {
    const std::array<std::uint32_t, 7> capabilities{spv::CapabilityShader, spv::CapabilityGroupNonUniform, spv::CapabilityGroupNonUniformBallot, spv::CapabilityGroupNonUniformVote, spv::CapabilityGroupNonUniformShuffle, spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses};
    const std::array<std::uint32_t, 4> userData{};
    const std::array<std::string_view, 2> extensions{"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"};
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, 0x20000u, code, 0, {}};
    request.context.waveSize = 32;
    request.context.userDataBaseRegister = 0;
    request.context.userData = userData;
    request.context.compute = ShaderComputeStageInfo{{64u, 1u, 1u}, 4u, {false, false, false}, false, 1u};
    request.target.vulkanVersion = 0x00403000u;
    request.target.spirvVersion = 0x00010600u;
    request.target.subgroupSize = 32;
    request.target.bdaAbiVersion = 1;
    request.target.supportedCapabilities = capabilities;
    request.target.supportedExtensions = extensions;
    request.target.maxWorkgroupSize = {1024u, 1024u, 64u};
    request.target.maxWorkgroupInvocations = 1024;
    request.target.maxWorkgroupSharedMemoryBytes = 49152;
    request.layout = {0, 0, 0, 128};
    request.useCache = false;
    auto result = Recompile(request);
    Require(!result.spirv.empty(), "the relative-addressing program did not recompile");
    return std::vector<std::uint32_t>(result.spirv.Words().begin(), result.spirv.Words().end());
}

Shape Scan(std::span<const std::uint32_t> words) {
    Shape shape;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> constants;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> arrays;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> functionPointers;
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> spv::WordCountShift;
        const auto op = static_cast<spv::Op>(words[cursor] & spv::OpCodeMask);
        Require(count != 0 && cursor + count <= words.size(), "truncated SPIR-V instruction");
        if (op == spv::OpConstant && count == 4) constants.emplace_back(words[cursor + 2], words[cursor + 3]);
        if (op == spv::OpTypeArray) arrays.emplace_back(words[cursor + 1], words[cursor + 3]);
        if (op == spv::OpTypePointer && words[cursor + 2] == spv::StorageClassFunction) functionPointers.emplace_back(words[cursor + 1], words[cursor + 3]);
        if (op == spv::OpVariable && words[cursor + 3] == spv::StorageClassFunction) {
            for (const auto& [pointer, pointee] : functionPointers) {
                if (pointer != words[cursor + 1]) continue;
                for (const auto& [array, length] : arrays) {
                    if (array != pointee) continue;
                    ++shape.functionArrays;
                    for (const auto& [constant, value] : constants) {
                        if (constant == length) shape.arrayLength = value;
                    }
                }
            }
        }
        if (op == spv::OpAccessChain) ++shape.accessChains;
        if (op == spv::OpSelect) ++shape.selects;
        cursor += count;
    }
    return shape;
}

bool run(const char* name, bool (*test)()) {
    try {
        if (test()) return true;
        std::fprintf(stderr, "%s\n", name);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s: %s\n", name, error.what());
    }
    return false;
}

}

int main() {
    bool passed = true;

    passed &= run("a relative read and write index a function-scope window instead of selecting over the registers", [] {
        const std::array<std::uint32_t, 8> code{MoveM0Two, MoveV4One, MoveV5Two, MoveV6Four, MovrelsV1V4, MovreldV4V1, MoveV1V5, EndProgram};
        const auto shape = Scan(RecompileProgram(code));
        if (shape.functionArrays != 1 || shape.arrayLength != 3 || shape.accessChains < 2 || shape.selects > 4) std::fprintf(stderr, "  window shape: arrays %zu length %u access chains %zu selects %zu\n", shape.functionArrays, shape.arrayLength, shape.accessChains, shape.selects);
        return shape.functionArrays == 1 && shape.arrayLength == 3 && shape.accessChains >= 2 && shape.selects <= 4;
    });

    passed &= run("a program without relative addressing keeps its registers in SSA values", [] {
        const std::array<std::uint32_t, 5> code{MoveV4One, MoveV5Two, MoveV6Four, MoveV1V5, EndProgram};
        const auto shape = Scan(RecompileProgram(code));
        return shape.functionArrays == 0 && shape.accessChains == 0;
    });

    return passed ? 0 : 1;
}
