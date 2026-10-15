#include "Optimization/SrtWalker/SrtFlatSlotClasses.hpp"
#include "Optimization/SrtWalker/SrtInstructionPredicates.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "IntermediateRepresentation/GuestRegister.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace ShaderRecompiler::Detail {

std::vector<std::uint8_t> ComputePureFlatSlots(const IrResourcePlan& plan) {
    std::vector<std::uint8_t> pure(plan.srtReads.size(), 1u);
    if (pure.empty()) return pure;
    const bool disqualified = !plan.srtPlanComplete || plan.requiresSpecializationMemory
        || std::any_of(plan.descriptorSources.begin(), plan.descriptorSources.end(), [](const DescriptorSource& source) { return source.tableColumn.has_value(); })
        || (plan.uniformFill.fill.kind != UniformFillKind::None && plan.uniformFill.fill.words > plan.uniformFill.values.size());
    if (disqualified) {
        std::fill(pure.begin(), pure.end(), 0u);
        return pure;
    }
    // The raw read node of each slot: a descriptor source dword may point at it directly (the
    // tracker resolved the handle argument before the plan builder patched the operands).
    std::unordered_map<const IrValue*, std::uint32_t> readSlots;
    for (std::uint32_t slot = 0; slot < plan.srtReads.size(); ++slot) {
        if (plan.srtReads[slot].value != nullptr) readSlots.emplace(plan.srtReads[slot].value->Resolve(), slot);
    }
    std::unordered_set<const IrValue*> visited;
    std::vector<IrValue*> pending;
    const auto mark = [&](IrValue* root) {
        if (root == nullptr) return;
        pending.push_back(root);
        while (!pending.empty()) {
            IrValue* value = pending.back()->Resolve();
            pending.pop_back();
            if (!visited.insert(value).second) continue;
            if (value->Opcode() == IrOpcode::ReadConst && value->ArgumentCount() >= 2 && value->Argument(1) != nullptr) {
                const IrValue* slot = value->Argument(1)->Resolve();
                if (slot->HasImmediate() && slot->Type() == IrType::U32 && slot->ImmediateU32() < pure.size()) {
                    const auto index = slot->ImmediateU32();
                    pure[index] = 0u;
                    pending.push_back(plan.srtReads[index].value);
                }
            }
            if (const auto found = readSlots.find(value); found != readSlots.end()) pure[found->second] = 0u;
            for (std::size_t i = 0; i < value->ArgumentCount(); ++i) {
                if (value->Argument(i) != nullptr) pending.push_back(value->Argument(i));
            }
        }
    };
    for (const auto& source : plan.descriptorSources) {
        for (std::uint32_t i = 0; i < source.dwordCount && i < source.dwords.size(); ++i) mark(source.dwords[i]);
    }
    for (const auto& block : plan.controlFlow) mark(block.condition);
    if (plan.uniformFill.fill.kind != UniformFillKind::None) {
        for (std::uint32_t i = 0; i < plan.uniformFill.fill.words; ++i) mark(plan.uniformFill.values[i]);
    }
    // The address cones: the arguments of every slot's raw read, never the read itself.
    for (const auto& read : plan.srtReads) {
        if (read.value == nullptr) continue;
        IrValue* raw = read.value->Resolve();
        for (std::size_t i = 0; i < raw->ArgumentCount(); ++i) mark(raw->Argument(i));
    }
    return pure;
}

std::vector<std::uint32_t> ComputeGuardedFlatSlots(const IrResourcePlan& plan) {
    std::unordered_map<const IrValue*, std::vector<const IrValue*>> users;
    std::unordered_set<const IrValue*> seen;
    std::vector<const IrValue*> pending;
    std::vector<const IrValue*> poisonable;
    for (const auto& read : plan.srtReads) {
        if (read.value != nullptr) pending.push_back(read.value->Resolve());
    }
    while (!pending.empty()) {
        const IrValue* value = pending.back();
        pending.pop_back();
        if (!seen.insert(value).second) continue;
        const IrValue* handle = value->ArgumentCount() != 0u && value->Argument(0) != nullptr ? value->Argument(0)->Resolve() : nullptr;
        if (IsRawRead(plan, *value) && handle != nullptr && LoadedFromMemory(*handle)) poisonable.push_back(value);
        const auto follow = [&](const IrValue* argument) {
            if (argument == nullptr) return;
            argument = argument->Resolve();
            users[argument].push_back(value);
            pending.push_back(argument);
        };
        for (std::size_t i = 0; i < value->ArgumentCount(); ++i) follow(value->Argument(i));
        if (value->Opcode() == IrOpcode::ReadConst && value->ArgumentCount() >= 2 && value->Argument(1) != nullptr) {
            const IrValue* slot = value->Argument(1)->Resolve();
            if (slot->HasImmediate() && slot->Type() == IrType::U32 && slot->ImmediateU32() < plan.srtReads.size()) follow(plan.srtReads[slot->ImmediateU32()].value);
        }
    }
    std::unordered_set<const IrValue*> reaching(poisonable.begin(), poisonable.end());
    while (!poisonable.empty()) {
        const IrValue* value = poisonable.back();
        poisonable.pop_back();
        const auto found = users.find(value);
        if (found == users.end()) continue;
        for (const IrValue* user : found->second) {
            if (reaching.insert(user).second) poisonable.push_back(user);
        }
    }
    std::vector<std::uint32_t> guarded;
    for (std::uint32_t slot = 0; slot < plan.srtReads.size(); ++slot) {
        if (plan.srtReads[slot].value != nullptr && reaching.contains(plan.srtReads[slot].value->Resolve())) guarded.push_back(slot);
    }
    return guarded;
}

}

namespace ShaderRecompiler {

DescriptorUserWords ComputeDescriptorUserWords(const IrResourcePlan& plan) {
    DescriptorUserWords result;
    result.patchable.assign(plan.userDataCount, 0u);
    const bool disqualified = !plan.srtPlanComplete || plan.materializationSources.size() < plan.info.buffers.size()
        || std::any_of(plan.descriptorSources.begin(), plan.descriptorSources.end(), [](const DescriptorSource& source) { return source.tableColumn.has_value(); })
        || (plan.uniformFill.fill.kind != UniformFillKind::None && plan.uniformFill.fill.words > plan.uniformFill.values.size());
    if (disqualified) return result;
    const auto userIndex = [&plan](const IrValue* value) -> std::optional<std::uint32_t> {
        value = value->Resolve();
        if (value->Opcode() != IrOpcode::GetUserData || value->ArgumentCount() == 0u || value->Argument(0) == nullptr) return std::nullopt;
        const auto reg = RegIndex(static_cast<ScalarReg>(value->Argument(0)->Register().index));
        if (reg < plan.userDataBase || reg - plan.userDataBase >= plan.userDataCount) return std::nullopt;
        return reg - plan.userDataBase;
    };
    std::vector<std::uint8_t> consumed(plan.userDataCount, 0u);
    std::unordered_set<const IrValue*> visited;
    std::unordered_set<const IrValue*> roots;
    std::unordered_map<const IrValue*, std::vector<std::pair<const IrValue*, std::size_t>>> uses;
    std::vector<const IrValue*> userNodes;
    std::vector<std::pair<IrValue*, const IrValue*>> pending;
    std::unordered_set<const IrValue*> otherRoots;
    std::unordered_map<const IrValue*, std::vector<std::pair<std::uint32_t, std::uint32_t>>> descriptorDwords;
    const auto mark = [&](IrValue* root, int rootKind = 0) {
        if (root == nullptr) return;
        roots.insert(root->Resolve());
        if (rootKind == 0) otherRoots.insert(root->Resolve());
        pending.emplace_back(root, nullptr);
        while (!pending.empty()) {
            auto [next, parent] = pending.back();
            pending.pop_back();
            IrValue* value = next->Resolve();
            if (parent != nullptr) {
                for (std::size_t i = 0; i < parent->ArgumentCount(); ++i) {
                    if (parent->Argument(i) != nullptr && parent->Argument(i)->Resolve() == value) uses[value].emplace_back(parent, i);
                }
            }
            if (!visited.insert(value).second) continue;
            if (const auto user = userIndex(value)) {
                consumed[*user] = 1u;
                userNodes.push_back(value);
            }
            if (value->Opcode() == IrOpcode::ReadConst && value->ArgumentCount() >= 2 && value->Argument(1) != nullptr) {
                const IrValue* slot = value->Argument(1)->Resolve();
                if (slot->HasImmediate() && slot->Type() == IrType::U32 && slot->ImmediateU32() < plan.srtReads.size()) pending.emplace_back(plan.srtReads[slot->ImmediateU32()].value, nullptr);
            }
            for (std::size_t i = 0; i < value->ArgumentCount(); ++i) {
                if (value->Argument(i) != nullptr) pending.emplace_back(value->Argument(i), value);
            }
        }
    };
    std::vector<std::uint8_t> bufferSource(plan.descriptorSources.size(), 0u);
    for (std::size_t i = 0; i < plan.info.buffers.size(); ++i) {
        if (plan.materializationSources[i] < bufferSource.size()) bufferSource[plan.materializationSources[i]] = 1u;
    }
    for (std::size_t i = plan.info.buffers.size(); i < plan.materializationSources.size(); ++i) {
        if (plan.materializationSources[i] < bufferSource.size()) bufferSource[plan.materializationSources[i]] = 0u;
    }
    std::vector<DescriptorUserWord> candidates;
    std::vector<std::tuple<const IrValue*, std::uint32_t, std::uint32_t>> tableCandidates;
    std::unordered_map<const IrValue*, std::vector<std::uint32_t>> dwordSources;
    for (std::uint32_t index = 0; index < plan.descriptorSources.size(); ++index) {
        const auto& source = plan.descriptorSources[index];
        for (std::uint32_t dword = 0; dword < source.dwordCount && dword < source.dwords.size(); ++dword) {
            if (source.dwords[dword] == nullptr) continue;
            dwordSources[source.dwords[dword]->Resolve()].push_back(index);
            if (bufferSource[index] != 0u) {
                if (const auto user = userIndex(source.dwords[dword])) {
                    for (std::uint32_t buffer = 0; buffer < plan.info.buffers.size(); ++buffer) {
                        if (plan.materializationSources[buffer] == index) candidates.push_back({buffer, dword, *user});
                    }
                    continue;
                }
            }
            const IrValue* resolved = source.dwords[dword]->Resolve();
            const bool tableRead = resolved->Opcode() == IrOpcode::LoadAddressU32 || resolved->Opcode() == IrOpcode::ReadConst;
            if (tableRead) {
                tableCandidates.emplace_back(resolved, index, dword);
                descriptorDwords[resolved].emplace_back(index, dword);
            }
            mark(source.dwords[dword], 1);
        }
    }
    for (const auto& block : plan.controlFlow) mark(block.condition);
    if (plan.uniformFill.fill.kind != UniformFillKind::None) {
        for (std::uint32_t i = 0; i < plan.uniformFill.fill.words; ++i) mark(plan.uniformFill.values[i]);
    }
    for (const auto& read : plan.srtReads) mark(read.value, 2);
    for (const auto& candidate : candidates) {
        if (consumed[candidate.user] != 0u) continue;
        result.words.push_back(candidate);
        result.patchable[candidate.user] = 1u;
    }
    std::vector<std::uint8_t> direct(plan.userDataCount, 0u);
    for (const auto& candidate : candidates) direct[candidate.user] = 1u;
    struct Base {
        std::optional<std::uint32_t> partner;
        bool low = false;
        bool valid = true;
        std::int64_t first = std::numeric_limits<std::int64_t>::max();
        std::int64_t end = std::numeric_limits<std::int64_t>::min();
    };
    std::vector<Base> bases(plan.userDataCount);
    std::vector<std::uint8_t> seen(plan.userDataCount, 0u);
    for (const IrValue* node : userNodes) {
        const auto user = *userIndex(node);
        auto& base = bases[user];
        if (seen[user] != 0u && !base.valid) continue;
        seen[user] = 1u;
        const auto found = uses.find(node);
        if (roots.contains(node) || direct[user] != 0u || found == uses.end() || found->second.empty()) {
            base.valid = false;
            continue;
        }
        for (const auto& [handle, position] : found->second) {
            const auto other = handle->ArgumentCount() == 2u && handle->Argument(1u - position) != nullptr ? userIndex(handle->Argument(1u - position)) : std::nullopt;
            const auto handleUses = uses.find(handle);
            if (!other || roots.contains(handle) || handleUses == uses.end() || (base.partner && *base.partner != *other) || (seen[user] > 1u && base.low != (position == 0u))) {
                base.valid = false;
                break;
            }
            base.partner = *other;
            base.low = position == 0u;
            seen[user] = 2u;
            for (const auto& [read, argument] : handleUses->second) {
                const IrValue* offset = read->ArgumentCount() >= 2u && read->Argument(1) != nullptr ? read->Argument(1)->Resolve() : nullptr;
                if (argument != 0u || read->Opcode() != IrOpcode::LoadAddressU32 || !Detail::IsRawRead(plan, *read) || offset == nullptr || !offset->HasImmediate()) {
                    base.valid = false;
                    break;
                }
                const auto immediate = static_cast<std::int64_t>(static_cast<std::int32_t>(plan.memoryInfo[read->Flags<MemoryFlags>().index].offset)) & ~std::int64_t{3};
                const auto at = immediate + static_cast<std::int64_t>(offset->ImmediateU32() & ~3u);
                base.first = std::min(base.first, at);
                base.end = std::max(base.end, at + 4);
            }
            if (!base.valid) break;
        }
    }
    for (std::uint32_t low = 0; low < plan.userDataCount; ++low) {
        const auto& base = bases[low];
        if (seen[low] == 0u || !base.valid || !base.low || !base.partner || *base.partner >= plan.userDataCount) continue;
        const auto& high = bases[*base.partner];
        if (seen[*base.partner] == 0u || !high.valid || high.low || !high.partner || *high.partner != low || base.first >= base.end) continue;
        result.pointers.push_back({low, *base.partner, base.first, base.end});
        result.patchable[low] = 1u;
        result.patchable[*base.partner] = 1u;
    }
    const auto tableOffset = [&](const IrValue* read) -> std::optional<std::pair<std::uint32_t, std::int64_t>> {
        if (read->Opcode() != IrOpcode::LoadAddressU32 || read->ArgumentCount() < 2u || read->Argument(0) == nullptr || read->Argument(1) == nullptr || !Detail::IsRawRead(plan, *read)) return std::nullopt;
        const IrValue* handle = read->Argument(0)->Resolve();
        const IrValue* offset = read->Argument(1)->Resolve();
        if (handle->ArgumentCount() != 2u || handle->Argument(0) == nullptr || handle->Argument(1) == nullptr || !offset->HasImmediate()) return std::nullopt;
        const auto low = userIndex(handle->Argument(0));
        const auto high = userIndex(handle->Argument(1));
        if (!low || !high) return std::nullopt;
        const auto pointer = std::find_if(result.pointers.begin(), result.pointers.end(), [&](const UserPointer& candidate) { return candidate.low == *low && candidate.high == *high; });
        if (pointer == result.pointers.end()) return std::nullopt;
        const auto immediate = static_cast<std::int64_t>(static_cast<std::int32_t>(plan.memoryInfo[read->Flags<MemoryFlags>().index].offset)) & ~std::int64_t{3};
        return std::make_pair(static_cast<std::uint32_t>(pointer - result.pointers.begin()), immediate + static_cast<std::int64_t>(offset->ImmediateU32() & ~3u));
    };
    const auto addTargets = [&](std::uint32_t pointer, std::int64_t at, std::uint32_t source, std::uint32_t dword) {
        for (std::uint32_t i = 0; i < plan.info.buffers.size(); ++i) {
            if (plan.info.buffers[i].source == source) result.tableWords.push_back({pointer, at, 0u, i, dword});
        }
        for (std::uint32_t i = 0; i < plan.info.images.size(); ++i) {
            if (plan.info.images[i].source == source) result.tableWords.push_back({pointer, at, 1u, i, dword});
        }
        for (std::uint32_t i = 0; i < plan.info.samplers.size(); ++i) {
            if (plan.info.samplers[i].source == source) result.tableWords.push_back({pointer, at, 2u, i, dword});
        }
    };
    for (const auto& [read, source, dword] : tableCandidates) {
        if (read->Opcode() != IrOpcode::LoadAddressU32 || uses.contains(read) || otherRoots.contains(read)) continue;
        if (const auto found = tableOffset(read)) addTargets(found->first, found->second, source, dword);
    }
    std::unordered_map<std::uint32_t, std::vector<const IrValue*>> slotReaders;
    for (const IrValue* value : visited) {
        if (value->Opcode() != IrOpcode::ReadConst || value->ArgumentCount() < 2u || value->Argument(1) == nullptr) continue;
        const IrValue* slot = value->Argument(1)->Resolve();
        if (slot->HasImmediate() && slot->Type() == IrType::U32 && slot->ImmediateU32() < plan.srtReads.size()) slotReaders[slot->ImmediateU32()].push_back(value);
    }
    for (std::uint32_t slot = 0; slot < plan.srtReads.size(); ++slot) {
        if (plan.srtReads[slot].value == nullptr) continue;
        const IrValue* raw = plan.srtReads[slot].value->Resolve();
        if (uses.contains(raw) || otherRoots.contains(raw)) continue;
        const auto found = tableOffset(raw);
        if (!found) continue;
        const auto& readers = slotReaders[slot];
        const bool exclusive = std::all_of(readers.begin(), readers.end(), [&](const IrValue* reader) { return !uses.contains(reader) && !otherRoots.contains(reader); });
        if (!exclusive) continue;
        result.tableWords.push_back({found->first, found->second, 3u, plan.srtReads[slot].flatOffset, 0u});
        for (const IrValue* reader : readers) {
            const auto dwords = descriptorDwords.find(reader);
            if (dwords == descriptorDwords.end()) continue;
            for (const auto& [source, dword] : dwords->second) addTargets(found->first, found->second, source, dword);
        }
    }
    std::vector<std::uint8_t> conditional(plan.descriptorSources.size(), 0u);
    for (const auto& block : plan.controlFlow) {
        for (const auto source : block.sources) {
            if (source < conditional.size()) conditional[source] = 1u;
        }
        for (const auto image : block.images) {
            if (image < plan.info.images.size() && plan.info.images[image].source < conditional.size()) conditional[plan.info.images[image].source] = 1u;
        }
    }
    std::unordered_map<const IrValue*, std::uint32_t> slotValues;
    for (std::uint32_t slot = 0; slot < plan.srtReads.size(); ++slot) {
        if (plan.srtReads[slot].value != nullptr) slotValues.emplace(plan.srtReads[slot].value->Resolve(), slot);
    }
    for (std::uint32_t slot = 0; slot < plan.srtReads.size(); ++slot) {
        if (plan.srtReads[slot].value == nullptr) continue;
        const IrValue* raw = plan.srtReads[slot].value->Resolve();
        const auto found = tableOffset(raw);
        if (!found) continue;
        constexpr std::int64_t Scalar = -1;
        const auto immediateLane = [](const IrValue* value) -> std::optional<std::int64_t> {
            if (value == nullptr) return std::nullopt;
            value = value->Resolve();
            if (!value->HasImmediate() || value->Type() != IrType::U32) return std::nullopt;
            return static_cast<std::int64_t>(value->ImmediateU32() % 32u);
        };
        std::vector<std::pair<const IrValue*, std::int64_t>> frontier;
        for (const IrValue* reader : slotReaders[slot]) frontier.emplace_back(reader, Scalar);
        frontier.emplace_back(raw, Scalar);
        std::set<std::pair<const IrValue*, std::int64_t>> reached;
        std::vector<std::uint32_t> sources;
        std::vector<std::uint32_t> conditions;
        bool derived = true;
        while (derived && !frontier.empty()) {
            const auto [value, lane] = frontier.back();
            frontier.pop_back();
            if (!reached.insert({value, lane}).second) continue;
            if (reached.size() > 65536u) {
                derived = false;
                break;
            }
            if (lane == Scalar) {
                std::size_t conditionsHere = 0;
                for (std::uint32_t block = 0; block < plan.controlFlow.size(); ++block) {
                    if (plan.controlFlow[block].condition != nullptr && plan.controlFlow[block].condition->Resolve() == value) {
                        conditions.push_back(block);
                        ++conditionsHere;
                    }
                }
                if (conditionsHere == 0u && otherRoots.contains(value)) derived = false;
                if (const auto other = slotValues.find(value); other != slotValues.end() && other->second != slot) derived = false;
                if (const auto feeds = dwordSources.find(value); feeds != dwordSources.end()) sources.insert(sources.end(), feeds->second.begin(), feeds->second.end());
            }
            const auto parents = uses.find(value);
            if (parents == uses.end()) continue;
            for (const auto& [parent, position] : parents->second) {
                if (Detail::IsRawRead(plan, *parent) || parent->Opcode() == IrOpcode::ReadConst) {
                    derived = false;
                    break;
                }
                if (parent->Opcode() == IrOpcode::WriteLane) {
                    const auto selector = immediateLane(parent->ArgumentCount() > 1u ? parent->Argument(1) : nullptr);
                    if (!selector || position == 1u || (position == 0u && lane != Scalar) || (position == 2u && lane == Scalar)) {
                        derived = false;
                        break;
                    }
                    frontier.emplace_back(parent, position == 0u ? *selector : lane);
                    continue;
                }
                if (parent->Opcode() == IrOpcode::ReadLane) {
                    const auto selector = immediateLane(parent->ArgumentCount() > 1u ? parent->Argument(1) : nullptr);
                    if (!selector || position != 0u || lane == Scalar) {
                        derived = false;
                        break;
                    }
                    if (*selector == lane) frontier.emplace_back(parent, Scalar);
                    continue;
                }
                if (lane != Scalar && !parent->IsPhi() && !(parent->Opcode() == IrOpcode::SelectU32 && position != 0u)) {
                    derived = false;
                    break;
                }
                frontier.emplace_back(parent, lane);
            }
        }
        std::sort(sources.begin(), sources.end());
        sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
        if (!derived || std::any_of(sources.begin(), sources.end(), [&](std::uint32_t source) { return source >= conditional.size(); })) continue;
        std::sort(conditions.begin(), conditions.end());
        conditions.erase(std::unique(conditions.begin(), conditions.end()), conditions.end());
        result.derivedWords.push_back({found->first, found->second, plan.srtReads[slot].flatOffset, std::move(sources), std::move(conditions)});
    }
    return result;
}

}
