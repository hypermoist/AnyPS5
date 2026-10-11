#include "prx/libSceAgcDriver/Eq/include/Event.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libkernel/Equeue/Equeue.hpp"

namespace {

constexpr int16_t EvfiltGraphicsCore = -14;

struct Registration {
    KernelEqueue eq;
    int id;
};

std::mutex g_mutex;
std::vector<Registration> g_registrations;

}

void AgcDriverDeliverEopInterrupt(std::uint32_t queue) {
    std::vector<Registration> registrations;
    {
        std::lock_guard lock(g_mutex);
        registrations = g_registrations;
    }
    static const bool trace = std::getenv("APS5_TRACE_EOP") != nullptr;
    if (trace) {
        std::size_t matched = 0;
        for (const auto& registration : registrations) matched += registration.id == static_cast<int>(queue);
        std::fprintf(stderr, "[eop] interrupt queue 0x%x: %zu of %zu registrations matched\n", queue, matched, registrations.size());
    }
    const auto compute = [](int id) { return id >= 0x20 && id < 0x58; };
    const bool exact = std::any_of(registrations.begin(), registrations.end(), [&](const Registration& registration) { return registration.id == static_cast<int>(queue); });
    for (const auto& registration : registrations) {
        if (registration.id == static_cast<int>(queue) || (!exact && compute(static_cast<int>(queue)) && compute(registration.id))) {
            EqueueTriggerEvent_nid_postfix(registration.eq, static_cast<uintptr_t>(registration.id), EvfiltGraphicsCore, nullptr);
        }
    }
}

extern "C" {

int APS5_VABI sceAgcDriverAddEqEvent(KernelEqueue eq, int id, void* udata) {
    static const bool trace = std::getenv("APS5_TRACE_EOP") != nullptr;
    if (trace) std::fprintf(stderr, "[eop] sceAgcDriverAddEqEvent eq %p id %d\n", reinterpret_cast<void*>(eq), id);
    if (eq == 0 || !EqueuePin_nid_postfix(eq)) {
        throw std::runtime_error(std::string(__func__) + ": invalid event queue");
    }
    KernelEqueueEvent event{};
    event.event.ident = static_cast<uintptr_t>(id);
    event.event.filter = EvfiltGraphicsCore;
    event.event.flags = EV_ADD | EV_CLEAR;
    event.event.udata = udata;
    event.filter.triggerFunc = [](KernelEqueueEvent* e, void*) {
        e->event.data = e->triggered ? e->event.data + 1 : 1;
        e->triggered = true;
    };
    event.filter.resetFunc = [](KernelEqueueEvent* e) {
        e->triggered = false;
        e->event.data = 0;
    };
    const int result = EqueueAddEvent_nid_postfix(eq, event);
    if (result != EQUEUE_OK) {
        throw std::runtime_error(std::string(__func__) + ": event queue rejected the event");
    }
    std::lock_guard lock(g_mutex);
    if (std::none_of(g_registrations.begin(), g_registrations.end(), [&](const Registration& r) { return r.eq == eq && r.id == id; })) {
        g_registrations.push_back({eq, id});
    }
    return 0;
}

int APS5_VABI sceAgcDriverDeleteEqEvent(KernelEqueue eq, int id) {
    {
        std::lock_guard lock(g_mutex);
        std::erase_if(g_registrations, [&](const Registration& r) { return r.eq == eq && r.id == id; });
    }
    if (EqueueDeleteEvent_nid_postfix(eq, static_cast<uintptr_t>(id), EvfiltGraphicsCore) != EQUEUE_OK) {
        throw std::runtime_error(std::string(__func__) + ": event is not registered");
    }
    return 0;
}

}
