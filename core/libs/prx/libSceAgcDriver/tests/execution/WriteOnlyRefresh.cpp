#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>
#include <sys/mman.h>

// StorageTexture::Refresh for a write-only use: a storage image the next dispatch only stores to
// takes nothing from another image's results pending in its memory (Unreal's transient targets
// alias differently shaped images that way), and a CPU store into its memory since its last use
// still reaches it.

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t IdentitySwizzle = 0xfacu;
constexpr std::uint32_t TwoChannelSwizzle = 0x22cu;
constexpr std::uint32_t TileR64KBX = 27;
constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t SurfaceBytes = 65536;

alignas(256) std::array<std::uint32_t, Threads * 4> Input{};

// buffer_load_dwordx4 v[0:3] from the input at tid * 16; image_store of them at (tid, 0) with dmask
// 0xf and at (tid, 1) with dmask 0x5: the image is only stored to.
alignas(256) constexpr std::array<std::uint32_t, 12> StoreCode{
    0x34020084, 0xe0381000, 0x80001401, 0xbf8c3f70, 0x7e3c0300, 0x7e3e0280, 0xf0201f08, 0x0001141e,
    0x7e3e0281, 0xf0201508, 0x0001141e, 0xbf810000,
};

struct Shape {
    std::uint32_t format;
    std::uint32_t swizzle;
    std::uint32_t width;
    std::uint32_t height;
};

// B: 32_32_SINT 16x16 and A: 32_SINT 32x8, both one 64 KiB tile block (the unit write tracking and
// the write-only refresh work in) at the same address: another shape, so B cannot borrow A's units
// on the device.
constexpr Shape B{63u, TwoChannelSwizzle, 16u, 16u};
constexpr Shape A{21u, IdentitySwizzle, 32u, 8u};

// Write-watched, 64 KiB-aligned guest memory: the refresh judges CPU stores by the tracker's stamps.
class GuestBlock {
public:
    GuestBlock() {
        void* raw = mmap(nullptr, BlockBytes * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        Require(raw != MAP_FAILED, "write-only refresh: cannot map the guest block");
        mapping = static_cast<std::uint8_t*>(raw);
        block = reinterpret_cast<std::uint8_t*>((reinterpret_cast<std::uintptr_t>(raw) + BlockBytes - 1) & ~(BlockBytes - 1));
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, BlockBytes);
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true);
    }
    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, BlockBytes);
        munmap(mapping, BlockBytes * 2);
    }
    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;
    std::uint8_t* Data() { return block; }

private:
    std::uint8_t* mapping = nullptr;
    std::uint8_t* block = nullptr;
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

std::array<std::uint32_t, 8> TextureDescriptor(const void* data, const Shape& shape) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (shape.format << 20u) | (((shape.width - 1u) & 3u) << 30u),
        ((shape.width - 1u) >> 2u) | ((shape.height - 1u) << 14u),
        shape.swizzle | (TileR64KBX << 20u) | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

// Stores `seed`-derived values through the image without writing anything back.
void Store(AgcDriver::VulkanDevice& device, std::uint8_t* texels, const Shape& shape, std::uint32_t seed) {
    for (std::uint32_t index = 0; index < Input.size(); ++index) Input[index] = (index + 1u) * 0x01010101u * seed;
    std::vector<std::uint32_t> userData(16, 0u);
    const auto buffer = BufferDescriptor(Input.data(), static_cast<std::uint32_t>(Input.size() * 4u));
    const auto texture = TextureDescriptor(texels, shape);
    std::copy(buffer.begin(), buffer.end(), userData.begin());
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    const std::span<const std::uint32_t> code(StoreCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}

void Flush(AgcDriver::VulkanDevice& device, std::uint8_t* texels) {
    AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(texels), SurfaceBytes, nullptr, "test");
    device.WaitIdle();
}

void Initialize(std::uint8_t* texels, std::uint32_t seed) {
    for (std::size_t index = 0; index < SurfaceBytes; ++index) texels[index] = static_cast<std::uint8_t>(seed + 37u * index + 11u * (index >> 4u));
}

std::vector<std::uint8_t> Bytes(const std::uint8_t* texels) {
    return std::vector<std::uint8_t>(texels, texels + SurfaceBytes);
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (!AgcDriver::GuestMemory::WriteWatched()) {
            std::puts("write-only refresh: no write watching, skipped");
            return VulkanTestSkipped;
        }
        GuestBlock reference;
        GuestBlock aliased;

        // The reference: B stored, written back, stored again with other values.
        Initialize(reference.Data(), 0x80u);
        Store(*device, reference.Data(), B, 1u);
        Flush(*device, reference.Data());
        Store(*device, reference.Data(), B, 3u);
        Flush(*device, reference.Data());

        // The same, with A's results left pending in B's memory between B's two uses: the write-only
        // use drops them instead of storing them and uploading them into B.
        Initialize(aliased.Data(), 0x80u);
        Store(*device, aliased.Data(), B, 1u);
        Flush(*device, aliased.Data());
        Store(*device, aliased.Data(), A, 2u);
        Store(*device, aliased.Data(), B, 3u);
        Flush(*device, aliased.Data());
        Require(Bytes(aliased.Data()) == Bytes(reference.Data()), "write-only refresh: B's write-only use took A's pending results in");

        // A CPU store since B's last use reaches B's image: the refresh takes the regular path, so
        // the texels B leaves unwritten hold the CPU's new bytes, as in a block where B is new.
        GuestBlock fresh;
        Initialize(aliased.Data(), 0x31u);
        Initialize(fresh.Data(), 0x31u);
        Store(*device, aliased.Data(), B, 5u);
        Store(*device, fresh.Data(), B, 5u);
        Flush(*device, aliased.Data());
        Flush(*device, fresh.Data());
        Require(Bytes(aliased.Data()) == Bytes(fresh.Data()), "write-only refresh: a CPU store before a write-only use was lost");

        std::puts("write-only refresh tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
