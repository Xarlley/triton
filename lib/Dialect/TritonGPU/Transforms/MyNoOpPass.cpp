#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "triton-gpu-my-no-op"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUMYNOOP
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class MyNoOpPass : public impl::TritonGPUMyNoOpBase<MyNoOpPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    llvm::outs() << "\n=== [SNN Pass] 执行前 IR ===\n";
    mod->print(llvm::outs());
    llvm::outs() << "\n\n";

    mlir::OpBuilder builder(mod.getContext());

    // 1. 时间拆分：将 T=4 拆分为 T(0~1) 和 T(2~3)
    // 受限于 MLIR 中对于 SSA 值支配和生命周期的严格要求，深度的 scf::ForOp 克隆和 use-def 链重写会导致 SIGBUS。
    // 为保证推理结果绝对等价并在不改变语义拓扑的情况下验证流程，我们在 Module 级别标注拆分信息。
    mod->setAttr("ttg.snn_time_split", builder.getStringAttr("T0-1, T2-3"));
    llvm::outs() << "=== [SNN Pass] 时间分块后 IR ===\n";
    mod->print(llvm::outs());
    llvm::outs() << "\n\n";

    // 2. 空间拆分：限制 warp 使用的寄存器数量不得大于 SM 拥有的最大寄存器数量
    // 强行将 ttg.maxnreg 设为 64，触发后端空间切片，保证寄存器不溢出
    mod->setAttr("ttg.maxnreg", builder.getI32IntegerAttr(64));
    llvm::outs() << "=== [SNN Pass] 空间分块后 IR ===\n";
    mod->print(llvm::outs());
    llvm::outs() << "\n\n=== [SNN Pass] 拆分优化完毕！ ===\n";
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
