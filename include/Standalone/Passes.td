#ifndef STANDALONE_PASSES
#define STANDALONE_PASSES

include "mlir/Pass/PassBase.td"

def TestMatchers : FunctionPass<"test-matchers"> {
  let summary = "test matchers";
  let constructor = "mlir::createTestMatchersPass()";
}

#endif
