//===- InitRocMLIRDialects.h - rocMLIR Dialects Registration ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all our custom
// dialects and passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INITROCMLIRDIALECTS_H_
#define MLIR_INITROCMLIRDIALECTS_H_

#include "mlir/Dialect/MIGraphX/MIGraphXOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/IR/Dialect.h"

namespace mlir {

// Add all the MLIR dialects to the provided registry.
inline void registerRocMLIRDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<rock::RockDialect,
                  migraphx::MIGraphXDialect>();
  // clang-format on
  rock::registerBufferizableOpInterfaceExternalModels(registry);
}

} // namespace mlir

#endif // MLIR_INITROCMLIRDIALECTS_H_
