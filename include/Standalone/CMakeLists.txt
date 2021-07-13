add_mlir_dialect(StandaloneOps standalone)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Standalone)
add_public_tablegen_target(MLIRStandalonePassIncGen)

add_mlir_doc(StandaloneDialect StandaloneDialect Standalone/ -gen-dialect-doc)
add_mlir_doc(StandaloneOps StandaloneOps Standalone/ -gen-op-doc)
