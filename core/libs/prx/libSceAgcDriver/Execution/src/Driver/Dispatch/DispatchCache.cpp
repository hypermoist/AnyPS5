#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <stdexcept>
#include <cstring>
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchCache.hpp"
#include <cstdlib>

namespace AgcDriver::DriverDetail {

bool Driver::stampValidate() {
    static const bool stamp = std::getenv("APS5_DISPATCH_STAMP_VALIDATE") != nullptr;
    return stamp;
}

bool Driver::dataHits() {
    static const bool hits = std::getenv("APS5_NO_DATA_HITS") == nullptr && VulkanDevice::TemplateDataRefresh();
    return hits;
}

bool Driver::userWordHits() {
    static const bool enabled = std::getenv("APS5_NO_USER_WORD_HITS") == nullptr && !stampValidate();
    return enabled;
}

std::uint64_t Driver::userPointerBase(std::span<const std::uint32_t> userData, std::uint32_t low, std::uint32_t high) {
    if (low >= userData.size() || high >= userData.size()) return 0;
    return ((static_cast<std::uint64_t>(userData[high]) << 32u) | userData[low]) & 0x0000ffffffffffffull;
}

std::shared_ptr<DispatchVariant> Driver::relocatedVariant(const DispatchVariant& variant, std::span<const std::uint32_t> userData) {
    bool moved = false;
    for (const auto& relocation : variant.relocations) moved = moved || userPointerBase(userData, relocation.low, relocation.high) != relocation.base;
    if (!moved || !variant.relocatable) return nullptr;
    auto relocated = std::make_shared<DispatchVariant>();
    relocated->runs = variant.runs;
    for (const auto& relocation : variant.relocations) {
        const auto base = userPointerBase(userData, relocation.low, relocation.high);
        if (base == 0) return nullptr;
        const auto delta = base - relocation.base;
        for (const auto run : relocation.runs) {
            relocated->runs[run].first += delta;
            relocated->runs[run].second += delta;
        }
    }
    relocated->words = variant.words;
    relocated->compiled = variant.compiled;
    relocated->shader = variant.shader;
    relocated->maskPositions = variant.maskPositions;
    return relocated;
}

void ReevaluateDescriptorSources(ShaderRecompiler::ResourceCapture& capture, std::span<const std::uint32_t> sources, std::span<const std::uint32_t> userData) {
    const auto& plan = *capture.plan;
    ShaderRecompiler::SrtRuntime runtime;
    runtime.userData = userData;
    runtime.readMemory = [](void*, std::uint64_t address, std::uint32_t* value) {
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), sizeof(std::uint32_t))) return false;
        std::memcpy(value, reinterpret_cast<const void*>(address), sizeof(std::uint32_t));
        return true;
    };
    std::vector<ShaderRecompiler::DescriptorValue> values;
    ShaderRecompiler::SrtWalker{}.EvaluateDescriptorSources(plan, sources, runtime, values);
    if (values.size() != sources.size()) throw std::runtime_error("descriptor sources were not re-evaluated");
    auto& snapshot = capture.snapshot;
    const auto active = [&snapshot](std::size_t index) { return snapshot.activeSources.empty() || (index < snapshot.activeSources.size() && snapshot.activeSources[index] != 0u); };
    for (std::size_t k = 0; k < sources.size(); ++k) {
        if (!active(sources[k])) continue;
        for (std::size_t i = 0; i < plan.info.buffers.size() && i < snapshot.buffers.size(); ++i) {
            if (plan.info.buffers[i].source == sources[k]) snapshot.buffers[i] = values[k];
        }
        for (std::size_t i = 0; i < plan.info.images.size() && i < snapshot.images.size(); ++i) {
            if (plan.info.images[i].source == sources[k] && active(plan.descriptorSources.size() + i)) snapshot.images[i] = values[k];
        }
        for (std::size_t i = 0; i < plan.info.samplers.size() && i < snapshot.samplers.size(); ++i) {
            if (plan.info.samplers[i].source == sources[k]) snapshot.samplers[i] = values[k];
        }
    }
}

bool ConditionsUnchanged(const ShaderRecompiler::IrResourcePlan& plan, std::span<const std::uint32_t> blocks, std::span<const std::uint32_t> userData, std::span<const std::pair<std::uint64_t, std::uint64_t>> runs, std::span<const std::uint32_t> words) {
    struct Stored {
        std::span<const std::pair<std::uint64_t, std::uint64_t>> runs;
        std::span<const std::uint32_t> words;
    } stored{runs, words};
    const auto live = [](void*, std::uint64_t address, std::uint32_t* value) {
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), sizeof(std::uint32_t))) return false;
        std::memcpy(value, reinterpret_cast<const void*>(address), sizeof(std::uint32_t));
        return true;
    };
    const auto old = [](void* context, std::uint64_t address, std::uint32_t* value) {
        const auto& memory = *static_cast<const Stored*>(context);
        std::size_t offset = 0;
        for (const auto& [begin, end] : memory.runs) {
            if (begin <= address && address + sizeof(std::uint32_t) <= end) {
                *value = memory.words[offset + static_cast<std::size_t>((address - begin) / sizeof(std::uint32_t))];
                return true;
            }
            offset += static_cast<std::size_t>((end - begin) / sizeof(std::uint32_t));
        }
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), sizeof(std::uint32_t))) return false;
        std::memcpy(value, reinterpret_cast<const void*>(address), sizeof(std::uint32_t));
        return true;
    };
    std::vector<ShaderRecompiler::IrValue*> conditions;
    for (const auto block : blocks) conditions.push_back(plan.controlFlow.at(block).condition);
    std::vector<std::uint32_t> before(conditions.size()), after(conditions.size());
    ShaderRecompiler::SrtRuntime runtime;
    runtime.userData = userData;
    runtime.userContext = &stored;
    runtime.readMemory = old;
    runtime.readSpecializationMemory = old;
    ShaderRecompiler::SrtWalker{}.EvaluateUniformValues(plan, conditions, runtime, before);
    runtime.readMemory = live;
    runtime.readSpecializationMemory = live;
    ShaderRecompiler::SrtWalker{}.EvaluateUniformValues(plan, conditions, runtime, after);
    return before == after;
}

bool Driver::verifyUserWordHits() {
    static const bool verify = std::getenv("APS5_VERIFY_USER_WORD_HITS") != nullptr;
    return verify;
}

std::shared_ptr<const UserWordMask> Driver::userWordMask(std::uint64_t address) {
    std::lock_guard lock(userWordMaskMutex);
    const auto found = userWordMasks.find(address);
    return found != userWordMasks.end() ? found->second : nullptr;
}

std::shared_ptr<const UserWordMask> Driver::learnUserWordMask(std::uint64_t address, const std::shared_ptr<const ShaderRecompiler::IrResourcePlan>& plan) {
    {
        std::lock_guard lock(userWordMaskMutex);
        const auto found = userWordMasks.find(address);
        if (found != userWordMasks.end() && found->second->plan == plan) return found->second;
    }
    auto mask = std::make_shared<UserWordMask>();
    mask->plan = plan;
    mask->words = std::make_shared<const ShaderRecompiler::DescriptorUserWords>(ShaderRecompiler::ComputeDescriptorUserWords(*plan));
    std::shared_ptr<const UserWordMask> learned = std::move(mask);
    std::lock_guard lock(userWordMaskMutex);
    userWordMasks[address] = learned;
    return learned;
}

bool Driver::verifyDataHits() {
    static const bool verify = std::getenv("APS5_VERIFY_DATA_HITS") != nullptr;
    return verify;
}

std::size_t Driver::dispatchVariants() {
    static const std::size_t variants = [] {
        if (stampValidate()) return std::size_t{1};
        const char* text = std::getenv("APS5_DISPATCH_VARIANTS");
        const auto parsed = text != nullptr ? std::strtoull(text, nullptr, 10) : 4ull;
        return static_cast<std::size_t>(std::clamp<unsigned long long>(parsed, 1, MaxDispatchVariants));
    }();
    return variants;
}

bool Driver::insertCompare() {
    static const bool compare = std::getenv("APS5_INSERT_COMPARE") != nullptr;
    return compare;
}

std::size_t Driver::dispatchCacheEntries() {
    static const std::size_t entries = [] {
        const char* text = std::getenv("APS5_DISPATCH_CACHE_ENTRIES");
        const auto parsed = text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
        if (parsed != 0) return static_cast<std::size_t>(parsed);
        if (stampValidate()) return std::size_t{4096};
        return dispatchVariants() > 2 ? std::size_t{65536} : std::size_t{131072};
    }();
    return entries;
}

std::uint64_t Driver::DispatchCacheBytes() {
    static const std::uint64_t bytes = [] {
        const char* text = std::getenv("APS5_DISPATCH_CACHE_BYTES");
        const auto parsed = text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
        return parsed != 0 ? static_cast<std::uint64_t>(parsed) : std::uint64_t{256} << 20u;
    }();
    return bytes;
}

Driver::DispatchCacheFootprint Driver::DispatchCacheSize() {
    std::lock_guard lock(dispatchCacheMutex);
    return {dispatchCache.size(), dispatchCacheVariantBytes};
}

std::uint64_t Driver::variantBytes(const DispatchVariant& variant) {
    return sizeof(DispatchVariant) + variant.words.size() * sizeof(std::uint32_t) + variant.shared.size() * sizeof(SharedWords) + variant.runs.size() * sizeof(std::pair<std::uint64_t, std::uint64_t>) + variant.captured.size() * sizeof(ShaderRecompiler::MemoryRegion) + (variant.vertexInfo != nullptr ? sizeof(ShaderRecompiler::ShaderVertexStageInfo) : 0);
}

void Driver::accountVariant(const DispatchVariant& variant, bool added) {
    if (added) {
        ++dispatchCacheVariants;
        dispatchCacheVariantBytes += variantBytes(variant);
    } else {
        --dispatchCacheVariants;
        dispatchCacheVariantBytes -= variantBytes(variant);
    }
}

void Driver::linkDispatchVariant(std::uint64_t key, const std::shared_ptr<DispatchVariant>& variant) {
    std::lock_guard cacheLock(dispatchCacheMutex);
    const auto found = dispatchCache.find(key);
    if (found == dispatchCache.end()) {
        auto created = std::make_shared<DispatchEntry>();
        created->touched = dispatchCacheHits;
        accountVariant(*variant, true);
        created->variants.push_back(variant);
        dispatchOrder.push_front(key);
        created->order = dispatchOrder.begin();
        dispatchCache.emplace(key, std::move(created));
    } else {
        const auto& old = found->second;
        if (!old->variants.empty() && old->variants.front() == variant) return;
        auto replacement = std::make_shared<DispatchEntry>();
        replacement->variants.reserve(dispatchVariants());
        accountVariant(*variant, true);
        replacement->variants.push_back(variant);
        for (const auto& kept : old->variants) {
            if (kept != variant && replacement->variants.size() < dispatchVariants()) replacement->variants.push_back(kept);
            else accountVariant(*kept, false);
        }
        replacement->touched = dispatchCacheHits;
        replacement->order = old->order;
        dispatchOrder.splice(dispatchOrder.begin(), dispatchOrder, replacement->order);
        found->second = std::move(replacement);
    }
    if (dispatchCache.size() > dispatchCacheEntries()) {
        eraseDispatchEntry(dispatchCache.find(dispatchOrder.back()));
        ++dispatchCacheEvictions;
    }
}

void Driver::eraseDispatchEntry(std::unordered_map<std::uint64_t, std::shared_ptr<DispatchEntry>>::iterator it) {
    for (const auto& variant : it->second->variants) accountVariant(*variant, false);
    dispatchOrder.erase(it->second->order);
    dispatchCache.erase(it);
}

}
