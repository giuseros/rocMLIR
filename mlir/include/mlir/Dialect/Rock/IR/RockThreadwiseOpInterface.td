//===- RockThreadwiseOpInterface.td - ops that accept rock views---------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockThreadwiseOpInterface, which abstracts rock operations
// that accept a view as operands.
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_THREADWISEOP_INTERFACE
#define ROCK_THREADWISEOP_INTERFACE

include "mlir/IR/OpBase.td"

def RockThreadwiseOpInterface : OpInterface<"ThreadwiseOpInterface"> {
  let description = [{
    This encompasses all the threadwise_* operations
  }];
  let cppNamespace = "::mlir::rock";

  let methods = [
    // TODO: more methods here as needed
  ];
}

#endif // ROCK_THREADWISEOP_INTERFACE
