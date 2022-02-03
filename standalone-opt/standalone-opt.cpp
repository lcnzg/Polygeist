//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneDialect.h"
#include "Standalone/TestMatchersPass.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
/*
static llvm::cl::OptionCategory toolOptions("standalone mlir - tool options");

static llvm::cl::opt<bool>
    showDialects("show-dialects",
                 llvm::cl::desc("Print the list of registered dialects"),
                 llvm::cl::init(false), llvm::cl::cat(toolOptions));
*/
int main(int argc, char **argv) {
  mlir::registerAllPasses();
  // TODO: Register standalone passes here.

  mlir::DialectRegistry registry;
  registry.insert<mlir::standalone::StandaloneDialect>();
  registry.insert<mlir::StandardOpsDialect>();
  mlir::MLIRContext context(registry);
  // TODO: why not with context?
  mlir::registerAllDialects(registry);
  mlir::registerTestMatchersPass();
  /*
    //if (showDialects) {
      llvm::outs() << "Registered Dialects:\n";
      for (mlir::Dialect *dialect : context.getLoadedDialects()) {
        llvm::outs() << dialect->getNamespace() << "\n";
      }
      //return 0;
    //}
  */
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Standalone optimizer driver\n", registry));
}
