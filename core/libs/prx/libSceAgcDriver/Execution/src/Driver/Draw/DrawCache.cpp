#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include <algorithm>
#include <bit>
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/WriteEvidence.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <cstdlib>
#include <utility>

namespace AgcDriver::DriverDetail {

bool DrawRecipeRecord::Matches(const std::vector<std::shared_ptr<DispatchVariant>>& variants) const {
    if (stages.size() != variants.size()) return false;
    for (std::size_t i = 0; i < stages.size(); ++i) {
        if (stages[i].owner_before(variants[i]) || variants[i].owner_before(stages[i])) return false;
    }
    return true;
}

bool DrawRecipeRecord::Expired() const {
    return std::any_of(stages.begin(), stages.end(), [](const std::weak_ptr<const DispatchVariant>& stage) { return stage.expired(); });
}

bool Driver::drawEntries() {
    static const bool entries = std::getenv("APS5_NO_DRAW_SRT_ENTRIES") == nullptr && !stampValidate();
    return entries;
}

bool Driver::drawDataHits() {
    static const bool hits = std::getenv("APS5_NO_DRAW_DATA_HITS") == nullptr;
    return hits;
}

bool Driver::drawStageDataHits() {
    static const bool hits = std::getenv("APS5_NO_DRAW_STAGE_DATA_HITS") == nullptr && !stampValidate();
    return hits;
}

bool Driver::verifyDrawDataHits() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_DATA_HITS") != nullptr;
    return verify;
}

bool Driver::verifyDrawEntries() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_ENTRIES") != nullptr;
    return verify;
}

bool Driver::registerKeyEnabled() {
    static const bool registerKey = std::getenv("APS5_NO_DRAW_KEY") == nullptr;
    return registerKey;
}

bool Driver::verifyDrawRecipe() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_RECIPE") != nullptr;
    return verify;
}

std::size_t Driver::drawCacheEntries() {
    static const std::size_t entries = [] {
        const char* text = std::getenv("APS5_DRAW_CACHE_ENTRIES");
        const auto parsed = text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
        return parsed != 0 ? static_cast<std::size_t>(parsed) : std::size_t{4096};
    }();
    return entries;
}

void Driver::accountDrawVariant(const DispatchVariant& variant, bool added) {
    if (added) {
        ++drawCacheVariants;
        drawCacheVariantBytes += variantBytes(variant);
    } else {
        --drawCacheVariants;
        drawCacheVariantBytes -= variantBytes(variant);
    }
}

DrawKeySightings::DrawKeySightings(std::size_t keyCount) {
    Graphics::Require(keyCount >= Ways, "draw key sightings need at least one bucket");
    const auto buckets = std::bit_ceil(keyCount / Ways);
    keys.assign(buckets * Ways, 0);
    filled.assign(buckets, 0);
    next.assign(buckets, 0);
    bucketMask = buckets - 1;
}

std::size_t DrawKeySightings::bucketOf(std::uint64_t key) const {
    auto mixed = key;
    mixed = (mixed ^ (mixed >> 30u)) * 0xbf58476d1ce4e5b9ull;
    mixed = (mixed ^ (mixed >> 27u)) * 0x94d049bb133111ebull;
    mixed ^= mixed >> 31u;
    return static_cast<std::size_t>(mixed) & bucketMask;
}

bool DrawKeySightings::Seen(std::uint64_t key) const {
    const auto bucket = bucketOf(key);
    const auto* ways = keys.data() + bucket * Ways;
    return std::find(ways, ways + filled[bucket], key) != ways + filled[bucket];
}

bool DrawKeySightings::SeenBefore(std::uint64_t key) {
    const auto bucket = bucketOf(key);
    auto* ways = keys.data() + bucket * Ways;
    if (std::find(ways, ways + filled[bucket], key) != ways + filled[bucket]) return true;
    ways[next[bucket]] = key;
    next[bucket] = static_cast<std::uint8_t>((next[bucket] + 1u) % Ways);
    if (filled[bucket] < Ways) ++filled[bucket];
    return false;
}

bool Driver::admitDrawKey(std::uint64_t key, std::uint64_t shape) {
    std::lock_guard cacheLock(drawCacheMutex);
    if (drawCache.contains(key)) return true;
    return sightDrawKey(key, shape);
}

bool Driver::sightDrawKey(std::uint64_t key, std::uint64_t shape) {
    if (!drawKeySightings) drawKeySightings.emplace(drawCacheEntries() * DrawKeySightings::Ways);
    const bool seen = drawKeySightings->SeenBefore(key);
    if (seen || shape == 0 || shapeHasOtherEntry(key, shape)) return seen;
    if (!drawShapeSightings) drawShapeSightings.emplace(drawCacheEntries() * DrawKeySightings::Ways);
    return drawShapeSightings->SeenBefore(shape);
}

bool Driver::drawKeyAdmitted(std::uint64_t key, std::uint64_t shape) const {
    if (drawKeySightings && drawKeySightings->Seen(key)) return true;
    if (shape == 0 || shapeHasOtherEntry(key, shape)) return false;
    return drawShapeSightings && drawShapeSightings->Seen(shape);
}

bool Driver::shapeHasOtherEntry(std::uint64_t key, std::uint64_t shape) const {
    const auto owner = drawShapes.find(shape);
    return owner != drawShapes.end() && owner->second != key;
}

void Driver::insertDrawEntry(std::uint64_t key, std::vector<std::shared_ptr<DispatchVariant>>& fresh, std::shared_ptr<const DrawDecode> decode, std::uint64_t shape, bool ahead) {
    std::vector<std::shared_ptr<DrawEntry>> released;
    std::lock_guard cacheLock(drawCacheMutex);
    auto& counters = drawEntryCounters;
    ++counters.inserts;
    const auto found = drawCache.find(key);
    auto replacement = std::make_shared<DrawEntry>();
    replacement->stages.resize(fresh.size());
    replacement->decode = std::move(decode);
    replacement->shape = shape;
    if (found != drawCache.end()) {
        if (found->second->stages.size() == fresh.size()) replacement->stages = found->second->stages;
        if (found->second->decode != nullptr) replacement->decode = found->second->decode;
        replacement->recipes.store(found->second->recipes.load());
    }
    for (std::size_t i = 0; i < fresh.size(); ++i) {
        if (fresh[i] == nullptr) continue;
        auto& variants = replacement->stages[i];
        const auto present = std::find_if(variants.begin(), variants.end(), [&](const std::shared_ptr<DispatchVariant>& kept) { return kept->pushOffset == fresh[i]->pushOffset && kept->runs == fresh[i]->runs && kept->words == fresh[i]->words; });
        if (present != variants.end()) {
            ++counters.present;
            fresh[i] = *present;
            continue;
        }
        accountDrawVariant(*fresh[i], true);
        variants.insert(variants.begin(), fresh[i]);
        ++counters.variantsInserted;
        while (variants.size() > dispatchVariants()) {
            accountDrawVariant(*variants.back(), false);
            variants.pop_back();
            ++counters.variantsEvicted;
        }
    }
    replacement->touched = drawCacheHits;
    const bool probation = ahead && found == drawCache.end() && !drawKeyAdmitted(key, shape);
    replacement->probation = found == drawCache.end() ? probation : found->second->probation;
    if (!replacement->probation && replacement->decode != nullptr && shape != 0) drawShapes[shape] = key;
    if (found == drawCache.end()) {
        auto& order = replacement->probation ? drawProbation : drawOrder;
        order.push_front(key);
        replacement->order = order.begin();
        drawCache.emplace(key, std::move(replacement));
    } else {
        replacement->order = found->second->order;
        auto& order = replacement->probation ? drawProbation : drawOrder;
        order.splice(order.begin(), order, replacement->order);
        releasedDrawEntries.push_back(std::exchange(found->second, std::move(replacement)));
    }
    evictDrawEntries();
    released.swap(releasedDrawEntries);
}

void Driver::sightDrawEntry(std::uint64_t key, DrawEntry& entry, std::vector<std::shared_ptr<DrawEntry>>& released) {
    if (!entry.probation || !sightDrawKey(key, entry.shape)) return;
    drawOrder.splice(drawOrder.begin(), drawProbation, entry.order);
    entry.probation = false;
    entry.touched = drawCacheHits;
    if (entry.decode != nullptr && entry.shape != 0) drawShapes[entry.shape] = key;
    evictDrawEntries();
    released.swap(releasedDrawEntries);
}

void Driver::evictDrawEntries() {
    const auto evict = [&](std::list<std::uint64_t>& order) {
        const auto last = drawCache.find(order.back());
        for (const auto& variants : last->second->stages) {
            for (const auto& variant : variants) accountDrawVariant(*variant, false);
        }
        if (traceDrawCache()) traceInsert(traceKeysEvicted, last->first);
        if (const auto shapeOf = drawShapes.find(last->second->shape); shapeOf != drawShapes.end() && shapeOf->second == last->first) drawShapes.erase(shapeOf);
        order.erase(last->second->order);
        releasedDrawEntries.push_back(std::move(last->second));
        drawCache.erase(last);
        ++drawCacheEvictions;
        ++drawEntryCounters.evictions;
    };
    while (drawOrder.size() > drawCacheEntries()) evict(drawOrder);
    while (drawProbation.size() > drawCacheEntries()) evict(drawProbation);
}

void Driver::eraseDrawEntry(std::uint64_t key, const std::shared_ptr<DrawEntry>& expected) {
    const auto found = drawCache.find(key);
    if (found == drawCache.end() || found->second != expected) return;
    for (const auto& variants : found->second->stages) {
        for (const auto& variant : variants) accountDrawVariant(*variant, false);
    }
    if (const auto shapeOf = drawShapes.find(found->second->shape); shapeOf != drawShapes.end() && shapeOf->second == key) drawShapes.erase(shapeOf);
    (found->second->probation ? drawProbation : drawOrder).erase(found->second->order);
    drawCache.erase(found);
}

bool DecodeProgramsCurrent(const DrawDecode& decode, const ShaderRegistry&) {
    return std::all_of(decode.programs.begin(), decode.programs.end(), [&](const DrawProgram& program) {
        if (!program.snapshot->header.empty()) return true;
        return ReadRawShader(program.snapshot->codeAddress + program.codeOffset * sizeof(std::uint32_t)) == program.snapshot &&
            (program.tailContinuation == nullptr || ReadRawShader(program.tailContinuation->codeAddress) == program.tailContinuation);
    });
}

std::shared_ptr<const DrawRecipe> Driver::findDrawRecipe(std::uint64_t key, const std::vector<std::shared_ptr<DispatchVariant>>& stages) {
    std::shared_ptr<const std::vector<DrawRecipeRecord>> records;
    {
        std::lock_guard cacheLock(drawCacheMutex);
        const auto found = drawCache.find(key);
        if (found == drawCache.end()) return nullptr;
        records = found->second->recipes.load();
    }
    if (records == nullptr) return nullptr;
    for (const auto& record : *records) {
        if (record.Matches(stages)) return record.recipe;
    }
    return nullptr;
}

void Driver::attachDrawRecipe(std::uint64_t key, const std::vector<std::shared_ptr<DispatchVariant>>& stages, std::shared_ptr<const DrawRecipe> recipe) {
    std::shared_ptr<DrawEntry> entry;
    {
        std::lock_guard cacheLock(drawCacheMutex);
        const auto found = drawCache.find(key);
        if (found == drawCache.end()) return;
        entry = found->second;
    }
    const auto old = entry->recipes.load();
    auto records = std::make_shared<std::vector<DrawRecipeRecord>>();
    records->push_back({std::vector<std::weak_ptr<const DispatchVariant>>(stages.begin(), stages.end()), std::move(recipe)});
    if (old != nullptr) {
        for (const auto& record : *old) {
            if (record.Matches(stages) || record.Expired() || records->size() >= dispatchVariants()) continue;
            records->push_back(record);
        }
    }
    entry->recipes.store(std::move(records));
    VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, VulkanDevice::RecipeKind::Draw);
}

}
