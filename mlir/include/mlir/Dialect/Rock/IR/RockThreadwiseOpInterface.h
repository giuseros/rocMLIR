//===- RockAcceptingViewOpInterface.td - ops that accept rock views--------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockThreadwiseOpInterface, which abstracts rock
// operations that accept a view as operands.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_IR_ROCK_ROCKTHREADWISEOP_INTERFACE_H
#define MLIR_DIALECT_ROCK_IR_ROCK_ROCKTHREADWISEOP_INTERFACE_H

#include "mlir/IR/OpDefinition.h"

#include "mlir/Dialect/Rock/IR/RockTypes.h"

#include "mlir/Dialect/Rock/IR/RockThreadwiseOpInterface.h.inc"

#endif // MLIR_DIALECT_ROCK_IR_ROCK_ROCKTHREADWISEOP_INTERFACE_H
