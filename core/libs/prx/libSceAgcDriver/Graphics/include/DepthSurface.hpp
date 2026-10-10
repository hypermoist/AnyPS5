#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEPTHSURFACE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEPTHSURFACE_HPP

#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace AgcDriver::Graphics {

class Texture;

VkImageView DepthSurfaceView(const Context& context, const DepthTarget& target);
std::uint64_t DepthSliceBytes(VkExtent2D extent, std::uint32_t bytesPerTexel);
void ClearDepthSurfaces(VkDevice device);
bool DepthSurfaceAt(std::uint64_t address);
// Whether a depth surface at `address` has this extent: only then is a storage image there one of its planes.
bool DepthSurfaceAt(std::uint64_t address, VkExtent2D extent);
// Records a copy of the depth plane of the depth surface at `address` into mip 0 of `destination`, a
// single-channel color image of the same extent and texel size in GENERAL layout, ordered after the
// depth tests that wrote the plane and before the shader reads that follow. Storage images cannot
// have a depth format, so a shader that uses the surface as a storage image works on this copy.
void RecordDepthSurfaceCopy(const Context& context, VkCommandBuffer commands, std::uint64_t address, VkImage destination, VkFormat destinationFormat, VkExtent2D destinationExtent);
// The way back after a shader wrote that image: its mip 0 replaces the depth plane, ordered after the
// shader's writes and before everything recorded later.
void RecordDepthSurfaceStore(const Context& context, VkCommandBuffer commands, std::uint64_t address, VkImage source, VkFormat sourceFormat, VkExtent2D sourceExtent);
std::shared_ptr<Texture> DepthSurfaceTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components);

}

#endif
