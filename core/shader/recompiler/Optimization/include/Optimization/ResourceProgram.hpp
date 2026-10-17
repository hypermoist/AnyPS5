#ifndef CORE_SHADER_RECOMPILER_OPTIMIZATION_RESOURCEPROGRAM_HPP
#define CORE_SHADER_RECOMPILER_OPTIMIZATION_RESOURCEPROGRAM_HPP

#include "Recompiler.hpp"
#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/SrtWalker.hpp"
#include <cstdint>
#include <memory>

namespace ShaderRecompiler {

struct PreparedControlFlow;
struct SourceEntry;
class ShaderPreparationContext {
public:
    [[nodiscard]] std::shared_ptr<const PreparedControlFlow> AcquireFrontend(const RecompileRequest& request);
    [[nodiscard]] std::uint32_t ResourcePrograms() const;

private:
    friend IrProgram PrepareResourceProgram(const RecompileRequest& request, ShaderPreparationContext* preparation);
    friend struct PreparedInterface;
    std::shared_ptr<const PreparedControlFlow> frontend;
    std::shared_ptr<SourceEntry> interfaceSource;
    std::unique_ptr<IrProgram> interfaceProgram;
    std::uint32_t resourcePrograms = 0;
};

[[nodiscard]] IrProgram PrepareResourceProgram(const RecompileRequest& request);
[[nodiscard]] IrProgram PrepareResourceProgram(const RecompileRequest& request, ShaderPreparationContext* preparation);
[[nodiscard]] std::uint64_t FrontEndBuilds();
[[nodiscard]] std::shared_ptr<const IrResourcePlan> GetResourcePlan(const RecompileRequest& request);

struct ResourceCapture {
    std::shared_ptr<const IrResourcePlan> plan;
    ResourceSnapshot snapshot;
    // The cache entry the plan belongs to; null when the request bypasses the cache.
    std::shared_ptr<SourceEntry> source;
    // The walk's read addresses when the plan has pure flat slots (the leaf of each pure slot,
    // and every other read sorted and deduplicated); empty otherwise.
    SrtReadTrace readTrace;
    SrtUserDataUse userDataUse;
    // APS5_PROFILE_DRAW: what CaptureResources spent resolving the source (the stage inputs, the
    // key over the code, the plan) before the walk.
    std::uint64_t sourceNanoseconds = 0;
};
[[nodiscard]] std::shared_ptr<const ResourceCapture> CaptureResources(const RecompileRequest& request, const SrtRuntime& runtime);

// The resolved source of a request (its cache entry with the plan built), for a driver that
// memoizes it per registered shader: ResolveSource is what CaptureResources does before the walk
// (the stage input validation, the key over the code, the lookup, the first-sight plan build), and
// the overload below captures over a handle without repeating it. A handle stays valid for every
// request with the same code and the same cache key fields (RecompileCacheKey::ContextHash plus
// the target); the vertex stages' input validation is repeated per capture because it reads V#
// fields the key does not cover. Null for a request that bypasses the cache (useCache false).
struct CompiledVariant;
struct SourceHandle {
    std::shared_ptr<SourceEntry> source;
    std::shared_ptr<const CompiledVariant> artifact;
    std::vector<std::uint64_t> staticKey;
};
class PreparedShaderInvocation {
public:
    static std::optional<PreparedShaderInvocation> TryCreate(const RecompileRequest& request, const std::shared_ptr<const SourceHandle>& handle);
    static std::optional<PreparedShaderInvocation> TryCreate(const RecompileRequest& request, const std::shared_ptr<const SourceHandle>& handle, std::span<const std::uint64_t> key);
    const RecompileRequest& Request() const { return request; }
    std::shared_ptr<const ResourceCapture> Capture(const SrtRuntime& runtime) const;
    std::shared_ptr<const RecompileResult> Materialize(const ResourceCapture& capture) const;

private:
    PreparedShaderInvocation(const RecompileRequest& request, const std::shared_ptr<const SourceHandle>& handle);
    RecompileRequest request;
    std::shared_ptr<const SourceHandle> handle;
};
[[nodiscard]] std::span<const std::uint32_t> GetPreparedCode(const SourceHandle& handle);
[[nodiscard]] std::shared_ptr<const SourceHandle> PrepareShader(const RecompileRequest& request);
[[nodiscard]] std::shared_ptr<const SourceHandle> PrepareShader(const RecompileRequest& request, ShaderPreparationContext* preparation);
[[nodiscard]] bool MatchesPreparedShader(const RecompileRequest& request, const SourceHandle& handle);
void BuildPreparedShaderKey(const RecompileRequest& request, std::vector<std::uint64_t>& key);
[[nodiscard]] bool MatchesPreparedShader(const RecompileRequest& request, const SourceHandle& handle, std::span<const std::uint64_t> key);
[[nodiscard]] const CompiledShaderArtifact& GetPreparedArtifact(const SourceHandle& handle);
[[nodiscard]] std::shared_ptr<const RecompileResult> RegistrationResult(const RecompileRequest& request, const SourceHandle& handle);
[[nodiscard]] std::shared_ptr<const RecompileResult> MaterializeShader(const RecompileRequest& request, const ResourceCapture& capture, const SourceHandle& handle);
[[nodiscard]] std::shared_ptr<const SourceHandle> ResolveSource(const RecompileRequest& request);
[[nodiscard]] std::shared_ptr<const ResourceCapture> CaptureResources(const RecompileRequest& request, const SrtRuntime& runtime, const SourceHandle& handle);

// The user-data words a plan's CPU walk consumes only as whole dwords of buffer descriptors: no
// other descriptor, condition, uniform-fill value, flattened SRT slot or read address reaches them,
// so two captures that differ only there differ only in those snapshot.buffers dwords and in
// snapshot.userData. `words` lists each such dword (snapshot.buffers[buffer].dwords[dword] is
// userData[user]); `pointers` lists the relocatable pointer pairs; `patchable[user]` is 1 for each
// user word of either kind. A plan with an indirect image or
// an incomplete SRT plan has none.
struct DescriptorUserWord {
    std::uint32_t buffer = 0;
    std::uint32_t dword = 0;
    std::uint32_t user = 0;
};
// A user-data pointer pair (`low`, `high`) the walk uses only as the base of dword reads at
// constant offsets: every read lies in [base + first, base + end). Two captures whose pointers
// differ read the same table words at the moved addresses, so a capture validated there is valid.
struct UserPointer {
    std::uint32_t low = 0;
    std::uint32_t high = 0;
    std::int64_t first = 0;
    std::int64_t end = 0;
};
// A descriptor dword the walk reads straight from a relocatable pointer's table (pointers[pointer],
// at base + offset) and consumes nowhere else: snapshot.<kind>[index].dwords[dword] is that word,
// kind 0 a buffer, 1 an image, 2 a sampler; kind 3 is the flattened SRT slot snapshot.flattenedSrt[index]
// that holds the word.
struct DescriptorTableWord {
    std::uint32_t pointer = 0;
    std::int64_t offset = 0;
    std::uint32_t kind = 0;
    std::uint32_t index = 0;
    std::uint32_t dword = 0;
};
// A word of a relocatable pointer's table (pointers[pointer], at base + offset) that the walk reads
// into the flattened SRT slot `flat` and consumes only inside the dwords of the descriptor sources
// `sources` and the conditions `conditions`: when it changes and every such condition evaluates as
// before, flattenedSrt[flat] takes the new word and re-evaluating the sources that were active
// (snapshot.activeSources) gives the new descriptors.
struct DerivedTableWord {
    std::uint32_t pointer = 0;
    std::int64_t offset = 0;
    std::uint32_t flat = 0;
    std::vector<std::uint32_t> sources;
    // The control-flow blocks (plan.controlFlow indices) whose condition the word reaches: a capture
    // reused over a changed word is valid only where each evaluates as it did over the old word.
    std::vector<std::uint32_t> conditions;
};
struct DescriptorUserWords {
    std::vector<DescriptorUserWord> words;
    std::vector<UserPointer> pointers;
    std::vector<DescriptorTableWord> tableWords;
    std::vector<DerivedTableWord> derivedWords;
    std::vector<std::uint8_t> patchable;
};
[[nodiscard]] DescriptorUserWords ComputeDescriptorUserWords(const IrResourcePlan& plan);

}

#endif
