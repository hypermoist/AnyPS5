#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_CONSTANTFOLDER_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_CONSTANTFOLDER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "IntermediateRepresentation/IrValue.hpp"

#include <span>

namespace ShaderRecompiler {

class ConstantFolder {
public:
    void Fold(IrProgram& program) const;

    void Fold(IrProgram& program, std::span<IrBlock* const> blocks) const;

    // The packed pixel ancillary lowering Fold ends with, alone: again after a pass that removes
    // the exec-mask selects a reused VGPR merged the word into (MaskedSelectEliminator).
    void LowerPackedAncillary(IrProgram& program) const;

private:
    void lowerAncillary(IrProgram& program, std::span<IrBlock* const> blocks) const;
    [[nodiscard]] bool tryFoldValue(IrProgram& program, IrValue& value) const;
};

}

#endif
