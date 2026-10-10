#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <functional>
#include <string_view>

namespace AgcDriver::DriverDetail {

// Debug aid: APS5_SKIP_FAILED_WORK=1 reports shader preparation, draws and dispatches that fail and
// goes on without them. Their results are then missing, which can break the title further on.
bool Driver::skipFailedWork() {
    static const bool skip = std::getenv("APS5_SKIP_FAILED_WORK") != nullptr;
    return skip;
}

void Driver::reportSkip(const char* kind, const std::string& what) {
    static std::mutex reportedMutex;
    static std::set<std::size_t> reported;
    static std::map<std::string, int> requestDumps;

    static const bool traceSkips = std::getenv("APS5_TRACE_SKIPS") != nullptr;
    static const int dumpLimit = [] { const char* text = std::getenv("APS5_SKIP_REQUEST_DUMPS"); return text ? std::atoi(text) : 8; }();
    static constexpr std::string_view marker = "\nRecompileRequest:\n";
    std::lock_guard lock(reportedMutex);
    std::string line;
    static bool announced = false;
    if (!announced) {
        announced = true;
        char text[160];
        std::snprintf(text, sizeof(text), "[gpu] skip reports: one write each, RecompileRequest dumps capped at %d per reason (APS5_SKIP_REQUEST_DUMPS)\n", dumpLimit);
        line = text;
    }
    const std::string prefix = "[gpu] skipped " + std::string(kind);
    if (reported.insert(std::hash<std::string>{}(what)).second) {
        const auto request = what.find(marker);
        if (request == std::string::npos) {
            line += prefix + ": " + what + "\n";
        } else {
            const auto reason = what.substr(0, request);
            const auto suffix = what.find(' ', request + marker.size());
            auto& dumps = requestDumps[reason];
            if (dumpLimit < 0 || dumps < dumpLimit) {
                ++dumps;
                line += prefix + ": " + what + "\n";
            } else {
                if (dumps == dumpLimit) {
                    ++dumps;
                    line += prefix + ": further RecompileRequest dumps for \"" + reason + "\" are left out\n";
                }
                line += prefix + ": " + reason + " (request left out)" + (suffix == std::string::npos ? std::string() : what.substr(suffix)) + "\n";
            }
        }
    } else if (traceSkips) {
        line += prefix + " again: " + what.substr(0, 100) + "\n";
    }
    if (!line.empty()) std::fwrite(line.data(), 1, line.size(), stderr);
}

std::string Driver::dumpRequest(std::uint64_t address, const ShaderRecompiler::RecompileRequest& request) {
    static std::mutex dumpMutex;
    static std::set<std::uint64_t> dumped;
    // APS5_DUMP_SHADERS=all writes every request (shader_<address>_<n>.req), to compare the variants
    // of one program; otherwise only its first.
    static const bool every = [] {
        const char* value = std::getenv("APS5_DUMP_SHADERS");
        return value != nullptr && std::strcmp(value, "all") == 0;
    }();
    static std::map<std::uint64_t, std::uint32_t> counts;
    char name[64];
    std::lock_guard lock(dumpMutex);
    if (every) {
        std::snprintf(name, sizeof(name), "shader_%llx_%u.req", static_cast<unsigned long long>(address), counts[address]++);
    } else {
        std::snprintf(name, sizeof(name), "shader_%llx.req", static_cast<unsigned long long>(address));
        if (!dumped.insert(address).second) return name;
    }
    try {
        const auto text = ShaderRecompiler::RequestSerializer{}.Serialize(request);
        if (std::FILE* file = std::fopen(name, "wb")) {
            std::fwrite(text.data(), 1, text.size(), file);
            std::fclose(file);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[gpu] could not serialize request for 0x%llx: %s\n", static_cast<unsigned long long>(address), error.what());
    }
    return name;
}

}
