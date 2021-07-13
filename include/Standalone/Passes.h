#include "mlir/Pass/Pass.h"
#include <memory> 
namespace mlir{
std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> createTestMatchersPass();

#include "Standalone/Passes.h.inc"
}
