#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_SHADERPREPARATION_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_SHADERPREPARATION_HPP

#include <memory>

namespace AgcDriver::DriverDetail {

struct ShaderSnapshot;
struct PreparedShaderState;

class ShaderPreparationTransaction {
public:
    ShaderPreparationTransaction();
    ~ShaderPreparationTransaction();
    ShaderPreparationTransaction(const ShaderPreparationTransaction&) = delete;
    ShaderPreparationTransaction& operator=(const ShaderPreparationTransaction&) = delete;
    PreparedShaderState& Edit(const ShaderSnapshot& snapshot);
    const PreparedShaderState& Read(const ShaderSnapshot& snapshot) const;
    void Hold();
    void Commit();
    // APS5_SKIP_FAILED_WORK (local debug aid): a skipped nested failure leaves the guest's
    // enclosing transaction committable; a nested transaction that threw has edited nothing.
    static void ForgiveNestedFailure();

private:
    struct State;
    std::unique_ptr<State> state;
    State* root = nullptr;
    bool committed = false;
};

}

#endif
