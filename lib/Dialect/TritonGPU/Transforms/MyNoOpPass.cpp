//===- MyNoOpPass.cpp - SNN 优化 Pass 的开发骨架 --------------------------===//
//
// MyNoOpPass 是面向脉冲神经网络（SNN）优化的自定义 Pass 的【占位实现 / 开发骨架】。
// 它目前的唯一作用是验证「SNN_FLAG / ENABLE_SNN_PASS -> Pass 条件性插入」这条链路：
//   1. 打印进入 Pass 时的 Module IR；
//   2. 写入占位标记属性 ttg.snn_time_split（自定义属性，当前无任何下游 Pass 消费）；
//   3. 写入 ttg.maxnreg=64（Triton 原生属性，会限制寄存器用量，但此处为硬编码值）。
//
// 注意：dev-log/dev-plan.md 第 2.1 节描述的【真正的时间拆分与空间拆分】尚未实现。
// 本 Pass 不改写任何计算 IR，因此对推理结果没有影响。
//
//===----------------------------------------------------------------------===//

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
    llvm::outs() << "\n=== [SNN Pass] 占位 Pass 开始，当前 Module IR ===\n";
    mod->print(llvm::outs());
    llvm::outs() << "\n\n";

    mlir::OpBuilder builder(mod.getContext());

    // 占位标记（非优化）：写入一个自定义属性，仅用于演示本 Pass 能够修改 Module。
    // 注意：ttg.snn_time_split 是自定义属性，当前没有任何下游 Pass 或调度器消费它，
    // 因此它不产生任何优化效果。真正的时间拆分见 dev-plan §2.1，尚未实现。
    mod->setAttr("ttg.snn_time_split", builder.getStringAttr("T0-1, T2-3"));
    llvm::outs() << "=== [SNN Pass] 已写入占位标记属性 ttg.snn_time_split ===\n";
    mod->print(llvm::outs());
    llvm::outs() << "\n\n";

    // 写入 ttg.maxnreg=64：这是 Triton 原生属性，后端会据此限制单线程寄存器用量。
    // 注意：64 是硬编码常量，并非 dev-plan §2.1 所要求的“计算得到的、保证不溢出的
    // 最大空间分块尺寸”，本 Pass 也不做任何按 warp 的分块。真正的空间拆分尚未实现。
    mod->setAttr("ttg.maxnreg", builder.getI32IntegerAttr(64));
    llvm::outs() << "=== [SNN Pass] 已写入 ttg.maxnreg=64 ===\n";
    mod->print(llvm::outs());
    llvm::outs() << "\n\n=== [SNN Pass] 占位 Pass 结束"
                    "（真正的时间/空间拆分见 dev-plan §2.1，尚未实现） ===\n";
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
