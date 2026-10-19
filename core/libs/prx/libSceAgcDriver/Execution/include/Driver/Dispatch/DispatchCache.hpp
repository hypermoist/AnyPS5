#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DISPATCHCACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DISPATCHCACHE_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/Recipe.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <utility>
#include <vector>
#include "prx/libc/include/general/AtomicSharedPtr.hpp"

namespace ShaderRecompiler {
struct DescriptorUserWords;
struct IrResourcePlan;
struct ResourceCapture;
}

namespace AgcDriver::DriverDetail {

inline constexpr std::size_t MaxDispatchVariants = 8;

// A program's user-data words that only feed buffer descriptors (ComputeDescriptorUserWords over the
// plan it was captured with): the dispatch key leaves them out, and a hit whose user data differs
// only there re-materializes the stored capture with the live words instead of walking again.
struct UserWordMask {
    std::shared_ptr<const ShaderRecompiler::IrResourcePlan> plan;
    std::shared_ptr<const ShaderRecompiler::DescriptorUserWords> words;
};

struct DispatchVariant {

    std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
    std::vector<std::uint32_t> words;
    std::vector<SharedWords> shared;
    std::uint64_t forgetSerial = 0;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
    std::array<std::uint32_t, 4> meshIndexDescriptor{};

    std::shared_ptr<const ShaderSnapshot> shader;

    AtomicSharedPtr<const Recipe> recipe;

    static constexpr std::uint32_t NoFlatBinding = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> dataPositions;
    std::vector<std::uint32_t> dataSlots;
    std::uint32_t flatBinding = NoFlatBinding;
    struct AddressSlot {
        std::uint32_t position;
        std::uint32_t binding;
        std::uint32_t slot;
    };
    std::vector<AddressSlot> addressSlots;
    std::vector<std::uint32_t> maskPositions;

    std::uint32_t pushOffset = 0;
    std::vector<std::int8_t> runBases;
    std::uint64_t relocatableWords = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> shaderDataPatches;
    std::shared_ptr<const ShaderRecompiler::ShaderVertexStageInfo> vertexInfo;

    std::shared_ptr<const ShaderRecompiler::ResourceCapture> capture;
    std::vector<std::uint32_t> userData;
    std::shared_ptr<const UserWordMask> userMask;
    bool userPatchable = false;
    struct Relocation {
        std::uint32_t low = 0;
        std::uint32_t high = 0;
        std::uint64_t base = 0;
        std::vector<std::uint32_t> runs;
    };
    std::vector<Relocation> relocations;
    bool relocatable = false;
    struct TableSlot {
        std::uint32_t position = 0;
        std::uint32_t kind = 0;
        std::uint32_t index = 0;
        std::uint32_t dword = 0;
    };
    std::vector<TableSlot> tableSlots;
    struct DerivedSlot {
        std::uint32_t position = 0;
        std::uint32_t flat = 0;
        const std::vector<std::uint32_t>* sources = nullptr;
        const std::vector<std::uint32_t>* conditions = nullptr;
    };
    std::vector<DerivedSlot> derivedSlots;
    std::uint64_t derivedRejectedFlat = 0;
    std::vector<std::uint32_t> tablePositions;

    std::shared_ptr<ShaderMemory> memory;
    std::vector<ShaderRecompiler::MemoryRegion> captured;
    std::vector<std::pair<std::uint64_t, std::size_t>> spans;
    std::atomic<std::uint64_t> generation{0};
};

struct DispatchEntry {
    std::vector<std::shared_ptr<DispatchVariant>> variants;

    std::uint64_t touched = 0;

    std::list<std::uint64_t>::iterator order;
};

struct EntryCounters {
    std::uint64_t lookups = 0, absent = 0, equal = 0, differing = 0, inaccessible = 0, queuedLabel = 0, flushingImage = 0, publishMoved = 0, pendingMoved = 0, forgetMoved = 0, imagesFlushed = 0, runsSynced = 0, forgetSinceInsert = 0, replaced = 0, inserts = 0, unstable = 0, touches = 0;

    std::uint64_t runsValidated = 0, runsInserted = 0, retriesEqual = 0, retriesMoved = 0;
    double validateUs = 0;

    std::uint64_t differingClassified = 0, differingAddress = 0, differingData = 0, differingWalk = 0, differingMixed = 0, differingRunsChanged = 0, differingMatchedPrior = 0, differingWords = 0;
    std::array<std::uint64_t, 4> differingWordBuckets{};

    std::array<std::uint64_t, MaxDispatchVariants> variantHitsByRank{};
    std::uint64_t variantsCompared = 0, variantsInserted = 0, variantsEvicted = 0;

    std::uint64_t userRefusedData = 0, userRefusedPointer = 0, userRefusedDescriptor = 0, userRefusedOther = 0, derivedInserts = 0, derivedRejectedFlat = 0, derivedHits = 0, derivedSourcesEvaluated = 0, derivedConditionsEvaluated = 0, derivedConditionsChanged = 0, tableHits = 0, tableWordsRefreshed = 0, tableInserts = 0, relocatedHits = 0, relocatedCompared = 0, relocatedInserts = 0, userHits = 0, userRefused = 0, userInserts = 0, userMaterializeFailures = 0, userVerified = 0, userMismatches = 0;
    std::uint64_t addressHits = 0, addressRefused = 0, addressInserts = 0, dataHits = 0, dataWordsRefreshed = 0, dataVerified = 0, dataInserts = 0, dataPositionsInserted = 0, dataLeavesUnmapped = 0, dataLeavesMismatched = 0, dataLeavesAliased = 0;
    std::array<std::uint64_t, MaxDispatchVariants> dataHitsByRank{};
    std::set<std::size_t> differingPositions;
    std::size_t differingFirstPosition = std::numeric_limits<std::size_t>::max(), differingLastPosition = 0;
    std::map<std::uint64_t, std::uint64_t> differingByProgram;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

struct DataMask {
    std::span<const std::uint32_t> positions;
    std::vector<std::pair<std::uint32_t, std::uint32_t>>* live;
};

enum class EntryOutcome { Equal, EqualData, Differing, Inaccessible, QueuedLabel, FlushingImage, PublishMoved, PendingMoved, ForgetMoved };

bool CacheableResult(const ShaderRecompiler::RecompileResult& compiled);
// Re-evaluates the descriptor sources `sources` of the capture's plan over live guest memory and
// `userData`, and stores the values in the capture's snapshot (every buffer, image and sampler made
// from one of them). Throws when a source cannot be evaluated.
void ReevaluateDescriptorSources(ShaderRecompiler::ResourceCapture& capture, std::span<const std::uint32_t> sources, std::span<const std::uint32_t> userData);
// Whether the control-flow conditions `blocks` of `plan` evaluate the same over live guest memory
// as over `words` (the words of `runs`, in order) in place of the memory there. Throws when a
// condition cannot be evaluated.
bool ConditionsUnchanged(const ShaderRecompiler::IrResourcePlan& plan, std::span<const std::uint32_t> blocks, std::span<const std::uint32_t> userData, std::span<const std::pair<std::uint64_t, std::uint64_t>> runs, std::span<const std::uint32_t> words);

}

#endif
