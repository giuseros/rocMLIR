//===- CleanMath.cpp - Clean up math after lowering/unrolling loops  ---===//
//
// Copyright 2022 AMD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKEXPANDTHREADWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-expand-threadwise"

using namespace mlir;
using namespace rock;

static int64_t getLoopStep(Value sourceView, Value dest, int64_t dim) {
  auto sourceViewType = cast<MemRefType>(sourceView.getType());
  MemRefType dstBufferType = dest.getType().cast<MemRefType>();
  auto elementType = sourceViewType.getElementType();
  int64_t srcStride;

  // This is the basic case
  int64_t numValues = dstBufferType.getNumElements();

  bool isSrcVectorBuffer = sourceViewType.getElementType().isa<VectorType>();
  bool isDstVectorBuffer = dstBufferType.getElementType().isa<VectorType>();
  int64_t vectorSrcLen, vectorDstLen;
  if (isSrcVectorBuffer) {
    srcStride = 1;
  } else {
    VectorizationResult vectorSrcRes =
        getMaxVectorization(sourceView, dim, /*inputDimLen=*/numValues);
    vectorSrcLen = vectorSrcRes.max;
    srcStride = vectorSrcLen;
  }

  return srcStride;
}

static ArrayAttr addFreezeRunningDimTransform(PatternRewriter &rw, Location loc,
                                              ArrayRef<int64_t> inputShape,
                                              ArrayAttr extraViewsAttr, int64_t numValues) {

  ArrayRef<int64_t> startShape;
  if (!extraViewsAttr.empty()){
    startShape = extraViewsAttr[0].cast<TransformMapAttr>().getUpperBounds();
  } else {
    startShape = inputShape;
  }

  // Create a transformation
  BottomUpTMBuilder freezeRunningDim(rw, startShape, loc);
  {
    SmallVector<uint32_t> startEndIndices;
    for (uint32_t i = 0; i < startShape.size(); i++) {
      startEndIndices.push_back(i);
    }
    freezeRunningDim.addDim("dummy_k", startShape.size(), numValues);
    freezeRunningDim.passThrough(startEndIndices, startEndIndices);
  }

  SmallVector<Attribute> extraViews =
      llvm::to_vector(extraViewsAttr.getValue());
  SmallVector<Attribute> newExtraViews{freezeRunningDim.get()};
  newExtraViews.insert(newExtraViews.end(), extraViews.begin(),
                       extraViews.end());
  ArrayAttr newExtraViewsAttr = rw.getArrayAttr(newExtraViews);
  return newExtraViewsAttr;
}

Value staticSubview(PatternRewriter &rw, Location loc, Value buffer,
                    Value offset, int64_t len) {
  auto memrefDestType = buffer.getType().dyn_cast<MemRefType>();
  IntegerAttr zero = rw.getIndexAttr(0);
  IntegerAttr one = rw.getIndexAttr(1);
  IntegerAttr numValuesV = rw.getIndexAttr(len);
  SmallVector<OpFoldResult> offsets(1, offset);
  SmallVector<OpFoldResult> sizes(1, numValuesV);
  SmallVector<OpFoldResult> strides(1, one);
  auto subviewType =
      cast<MemRefType>(memref::SubViewOp::inferResultType(
         memrefDestType , offsets, sizes, strides));
  Value subview = rw.create<memref::SubViewOp>(loc,subviewType, buffer,
                                               offsets, sizes, strides);
  return subview;
}

struct ExpandThreadwiseReadIntoOp
    : public OpRewritePattern<rock::ThreadwiseReadIntoOp> {
  using OpRewritePattern<rock::ThreadwiseReadIntoOp>::OpRewritePattern;

  mutable int counter{0};

  LogicalResult matchAndRewrite(rock::ThreadwiseReadIntoOp op,
                                PatternRewriter &rw) const override {
    // If we already exapaned, stop
    auto threadwiseLoop = op->getParentOfType<affine::AffineForOp>();
    if (op->hasAttr("expanded"))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Expanding " << op << "\n");
    auto loc = op->getLoc();
    Value zero = rw.create<arith::ConstantIndexOp>(loc, 0);

    // Information about data types and vectorization
    auto memrefSourceType = op.getSource().getType().dyn_cast<MemRefType>();
    auto memrefDestType = op.getDest().getType().dyn_cast<MemRefType>();
    auto step = getLoopStep(op.getSource(), op.getDest(), op.getExtraIndices().size()-1);

    // Add an extra view that freezes the running dimension on the view
    ArrayAttr newExtraViewsAttr = addFreezeRunningDimTransform(
        rw, loc, memrefSourceType.getShape(), op.getExtraViews(), step);
    auto frozenExtraIndices = llvm::to_vector(op.getExtraIndices());

    // Generate the for loop with the right vectorization
    auto forOp = rw.create<affine::AffineForOp>(
        loc, 0, memrefDestType.getNumElements(), step);
    {

      PatternRewriter::InsertionGuard guard(rw);
      rw.setInsertionPointToStart(forOp.getBody());

      // Index into the view with the loop iteration variable
      Value kv = forOp.getInductionVar();
      frozenExtraIndices.push_back(kv);

      // Get a static subview of the destination
      auto subview = staticSubview(rw, loc, op.getDest(), kv, step);

      // Emit the new threadwise readinto
      auto newop = rw.create<ThreadwiseReadIntoOp>(
          loc, op.getSource(), subview, newExtraViewsAttr, frozenExtraIndices,
          op.getForceUnroll(), op.getUseIndexDiffs());
      newop->setAttr("expanded", rw.getBoolAttr(true));
    }
    forOp->setAttr("forceUnroll", rw.getBoolAttr(true));
    LLVM_DEBUG(llvm::dbgs() << "Result:  " << forOp << "\n");
    rw.eraseOp(op);
    return failure();
  }
};

namespace {
struct RockExpandThreadwisePass
    : public rock::impl::RockExpandThreadwisePassBase<
          RockExpandThreadwisePass> {
  void runOnOperation() override;
};
} // end namespace

/// This function hasn't come from anywhere and is relying on the overall
/// tests of the integer range inference implementation for its correctness.

void RockExpandThreadwisePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<ExpandThreadwiseReadIntoOp>(&getContext());
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}
