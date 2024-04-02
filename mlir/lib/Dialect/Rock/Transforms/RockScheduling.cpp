//===- RockScheduling.cpp   ---===//
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

#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/IR/RockThreadwiseOpInterface.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Transforms/RockMultibuffer.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"

#include <algorithm>
#include <map>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKSCHEDULINGPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-scheduling"

using namespace mlir;
using mlir::gpu::AddressSpace;

namespace {
struct RockSchedulingPass : public rock::impl::RockSchedulingPassBase<RockSchedulingPass> {
  using rock::impl::RockSchedulingPassBase<RockSchedulingPass>::RockSchedulingPassBase;
  void runOnOperation() override;
};
}
enum class MemoryAccessType : uint32_t { READ = 1, WRITE = 2, UNKNOWN = 3 };

MemoryAccessType getOperandAccessType(Operation *op, Value operand) {
  if (hasEffect<MemoryEffects::Write>(op, operand)) {
    return MemoryAccessType::WRITE;
  } else if (hasEffect<MemoryEffects::Read>(op, operand)) {
    return MemoryAccessType::READ;
  } else {
    return MemoryAccessType::UNKNOWN;
  }
}

template <typename MemrefTypedValue>
AddressSpace getAddressSpace(MemrefTypedValue val) {
  auto memrefType = dyn_cast<MemRefType>(val.getType());

  if (memrefType){
    return memrefType
        .getMemorySpace()
        .template cast<gpu::AddressSpaceAttr>()
        .getValue();
  }
  assert("Invalid non-memref valued type");
  return gpu::AddressSpace::Global;
}

// This is the scheduling unit:
// - bunch of valu math
// - last op always a write or read
// - accType describes if the access type of the last op
// - addrSpace describes the address space of the last read
struct SchedulingUnit{
  SmallVector<Operation *> operations;
  rock::GpuAllocOp mem;
  rock::StageOp stage;
  MemoryAccessType accType;
  AddressSpace addrSpace;

  void moveBefore(Operation * anchorOp){
    // llvm::errs()<<"----SCHEDULING----\n";
    for (Operation *schedOp: operations){
      if (schedOp->getParentOfType<rock::StageOp>()) {
        schedOp->moveBefore(anchorOp);
        // schedOp->dump();
      }
    }
    // llvm::errs()<<"------------------\n";
  }
};

// Simple rewrite pass to remove the stages and backward barriers in the
// prologue and in the Epilogue
struct RemoveStagesRewritePattern : public OpRewritePattern<rock::StageOp> {
  using OpRewritePattern<rock::StageOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::StageOp op,
                                PatternRewriter &rw) const override {
    Block *sourceBlock = &op.getRegion().front();
    if (!sourceBlock->empty() && sourceBlock->back().mightHaveTrait<OpTrait::IsTerminator>())
      rw.eraseOp(sourceBlock->getTerminator());
    bool isRemovableBarrier = (op.getName() == "__bwd_barrier__" &&
                               !dyn_cast<scf::ForOp>(op->getParentOp()));

    if (!sourceBlock->empty() && !isRemovableBarrier) {
     rw.inlineBlockBefore(sourceBlock, op);
    }
    rw.eraseOp(op);
    return failure();
  }
};

struct ScheduleStagesPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  using DependencySet = DenseMap<rock::GpuAllocOp, SmallVector<std::tuple<MemoryAccessType, rock::StageOp, Operation*>>>;

  std::pair<rock::GpuAllocOp, MemoryAccessType> findAllocFromOp(Operation * op) const{
    for (Value operand : op->getOperands()) {
      MemoryAccessType accType = getOperandAccessType(op, operand);
      auto maybeAlloc = rock::findGpuAlloc(operand);
      if (succeeded(maybeAlloc))
        return {*maybeAlloc, accType};
    }
    return {nullptr, MemoryAccessType::UNKNOWN};
  }

  bool hasGlobalMemAccess(Operation *op)const{
    for (Value operand : op->getOperands()) {
      MemoryAccessType accType = getOperandAccessType(op, operand);
      if (accType == MemoryAccessType::READ){
        if (operand.getType().isa<MemRefType>()){
          return (getAddressSpace(operand) == AddressSpace::Global);
        }
      }
    }
    return false;
  }

  // SmallVector<Value> getIndicesFromMemoryOp(Operation *op) const {
  //   if (auto vecLoad = dyn_cast<vector::LoadOp>(op)){
  //     return vecLoad.getIndices();
  //   } else if (auto vecStore= dyn_cast<vector::StoreOp>(op)){
  //     return vecStore.getIndices();
  //   } else if (auto memrefLoad = dyn_cast<memref::LoadOp>(op)){
  //     return memrefLoad.getIndices();
  //   } else if (auto memrefStore= dyn_cast<memref::StoreOp>(op)){
  //     return memrefStore.getIndices();
  //   } else if (auto trRead= dyn_cast<vector::TransferReadOp>(op)){
  //     return trRead.getIndices();
  //   } else if (auto trWrite = dyn_cast<vector::TransferWriteOp>(op)){
  //     return trWrite.getIndices();
  //   } else {
  //     assert("Unkown memory operation");
  //   }
  //   return {};
  // }

  void visit(Operation *op, SchedulingUnit& unit, DenseSet<Operation*>& visited) const{
    if (visited.contains(op))
       return;

    for (Value operand : op->getOperands()){
      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      if (!defOp->getParentOfType<rock::StageOp>())
        continue;
      visit(defOp, unit, visited);
    }

    unit.operations.push_back(op);
    visited.insert(op);
  }

  void fillSchedulingUnit(Operation *op, SchedulingUnit& unit) const{
    DenseSet<Operation *> visited;
    visit(op, unit, visited);
  }

  SmallVector<SmallVector<SchedulingUnit>> createParallelSchedUnits(ArrayRef <rock::StageOp> parallelStages) const{
    SmallVector<SmallVector<SchedulingUnit>> parallelUnits;
    for (auto p : parallelStages){
      SmallVector<SchedulingUnit> units;
      for (Operation &op : p.getRegion().front()){
        auto [alloc, accType] = findAllocFromOp(&op);
        AddressSpace addrSpace;
        if (hasGlobalMemAccess(&op)){
          accType = MemoryAccessType::READ;
          addrSpace =AddressSpace::Global;
          alloc = nullptr;
        } else if (accType != MemoryAccessType::UNKNOWN){
          addrSpace = getAddressSpace(alloc);
        } else {
          continue;
        }
        SchedulingUnit schedUnit{{}, alloc, p, accType, addrSpace};
        fillSchedulingUnit(&op, schedUnit);
        units.push_back(schedUnit);
      }
      parallelUnits.push_back(units);
    }
    return parallelUnits;
  }

  bool solveConflict(Operation *thisOp, rock::GpuAllocOp alloc,
                     MemoryAccessType thisAccType,
                     DependencySet &accessSet) const {

    rock::StageOp thisStage = thisOp->getParentOfType<rock::StageOp>();
    if (alloc) {
      if (getAddressSpace(alloc) != AddressSpace::Private)
        return false;
      if (!accessSet.contains(alloc)) {
        accessSet[alloc].push_back({thisAccType, thisStage, thisOp});
        return false;
      }

      for (auto [otherAccType, otherStage, otherOp] : accessSet.at(alloc)) {
        if (thisStage != otherStage && thisAccType == MemoryAccessType::READ &&
            otherAccType == MemoryAccessType::WRITE) {
          thisOp->moveBefore(otherOp);
          return true;
        }
      }
    }
    return false;
  }

  // We dont' have memory dependencies among parallel stages
  void schedule(PatternRewriter &rw, Location loc, ArrayRef<rock::StageOp> parallelStages) const {

    size_t numParallelStages = parallelStages.size();
    Operation *lastOp = parallelStages[0];
    int64_t stageIndex = 0;
    DependencySet regDependencies;

    // Simple strategy: interleave one op from each parallel stages
    while(true){
      if (llvm::all_of(parallelStages, [](rock::StageOp s){return s.getRegion().front().empty();})){
        break;
      }
      auto curStage = parallelStages[stageIndex % numParallelStages];
      while (curStage.getRegion().front().empty()){
        stageIndex++;
        curStage = parallelStages[stageIndex % numParallelStages];
      }
      Operation& op=curStage.getRegion().front().front();
      if (op.hasTrait<OpTrait::IsTerminator>()){
        rw.eraseOp(&op);
      } else {
        auto [alloc, accType] = findAllocFromOp(&op);
        if (!solveConflict(&op, alloc, accType, regDependencies)){
          op.moveAfter(lastOp);
          lastOp = &op;
        }
      }
      stageIndex++;
    }
  }

  void schedule(PatternRewriter &rw, Location loc, ArrayRef<SmallVector<SchedulingUnit>> parallelUnits) const {

    SmallVector<int64_t> stageSizes;
    for (const auto& parallelUnit: parallelUnits)
      stageSizes.push_back(parallelUnit.size());

    Operation *firstStage = parallelUnits[0].front().stage;

    size_t numParallelStages = parallelUnits.size();
    int64_t stageIndex = 0;
    SmallVector<int64_t> schedUnitIndices(numParallelStages, 0);

    // Simple strategy: interleave one schedule unit from each parallel stages
    while(llvm::any_of(stageSizes, [](int64_t size) {return size!=0;})){
      while (stageSizes[stageIndex] == 0)
        stageIndex = (stageIndex+1)%numParallelStages;
      int64_t schedIndex = schedUnitIndices[stageIndex]++;
      SchedulingUnit unit = parallelUnits[stageIndex][schedIndex];
      stageSizes[stageIndex]--;
      unit.moveBefore(firstStage);
      stageIndex = (stageIndex+1)%numParallelStages;
    }
  }

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rw) const override {
    auto pipelineAttr = op->removeAttr(rock::PipelineAttr::getMnemonic());
    if (!pipelineAttr)
      return failure();

    int64_t ii = pipelineAttr
                     .dyn_cast<rock::PipelineAttr>()
                     .getInitiationInterval();

    llvm::DenseMap<int64_t, llvm::SmallVector<rock::StageOp>> slots;
    int64_t slot = 0;
    int64_t curStage = 0;
    bool done = true;
    for(auto stage: op.getLoopBody().getOps<rock::StageOp>()){
      if (stage.getRegion().front().empty())
        continue;
      if (stage.getName() == "__empty_stage__")
        continue;
      if(stage.getName() == "__fwd_barrier__")
        continue;
      if(stage.getName() == "__bwd_barrier__")
        continue;
      done = false;

      slots[slot].push_back(stage);
      if (++curStage >= ii)
        slot++;
    }

    SmallVector<Operation *> unstagedOps;
    for(Operation &maybeUnstaged: op.getLoopBody().front()){
      if (!isa<rock::StageOp>(maybeUnstaged) &&
          !maybeUnstaged.getParentOfType<rock::StageOp>() &&
          !maybeUnstaged.hasTrait<OpTrait::IsTerminator>())
        unstagedOps.push_back(&maybeUnstaged);
    }


    if (done) return failure();

    for (auto [slot, parallelStages] : slots ){
      auto parallelSchedUnits = createParallelSchedUnits(parallelStages);
      schedule(rw, op.getLoc(), parallelSchedUnits);
    }

    // If there are operation that do not belong to stages, move at the beginning of the block
    for (Operation *unstagedOp: llvm::reverse(unstagedOps))
      unstagedOp->moveAfter(&op.getLoopBody().front(), op.getLoopBody().front().begin());

    {
      PatternRewriter::InsertionGuard guard(rw);
      rw.setInsertionPoint(op.getLoopBody().front().getTerminator());
      rw.create<ROCDL::IglpOpt>(op.getLoc(), rw.getI32IntegerAttr(0));
    }
    return success();
  }
};

// Simple rewrite pass to remove back-to-back barriers
struct RemoveBackToBackBarriersRewritePattern
    : public OpRewritePattern<rock::LDSBarrierOp> {
  using OpRewritePattern<rock::LDSBarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::LDSBarrierOp op,
                                PatternRewriter &rw) const override {
    if (dyn_cast_or_null<rock::LDSBarrierOp>(op->getNextNode())) {
      op->getNextNode()->erase();
      return success();
    }
    return failure();
  }
};

// Simple rewrite pass to hoist operations that do not
// access LDS before the barriers
struct PushBarrierDownRewritePattern
    : public OpRewritePattern<rock::LDSBarrierOp> {
  using OpRewritePattern<rock::LDSBarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::LDSBarrierOp op,
                                PatternRewriter &rw) const override {
    Operation *nextOp = op->getNextNode();

    // Make sure that there is a nextOp
    if (!nextOp)
      return failure();

    // Don't go over the terminator
    if (!nextOp->getNextNode())
      return failure();

    // We assume that operations that have a body may modify LDS
    if (nextOp->getNumRegions() > 0)
      return failure();

    bool moveDown = true;
    // Make sure that the "nextOp" doesn't modify LDS
    for (Value operand : nextOp->getOperands()) {
      auto maybeAlloc = rock::findGpuAlloc(operand);
      if (succeeded(maybeAlloc) &&
          getAddressSpace(*maybeAlloc) == AddressSpace::Workgroup)
        moveDown = false;
    }

    if (moveDown) {
      rw.setInsertionPointAfter(nextOp);
      rw.create<rock::LDSBarrierOp>(nextOp->getLoc());
      rw.eraseOp(op);
      return success();
    }
    return failure();
  }
};

void RockSchedulingPass::runOnOperation() {
  {
      RewritePatternSet patterns(&getContext());
      patterns.add<ScheduleStagesPattern>(
          &getContext());
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
  {
      RewritePatternSet patterns(&getContext());
      patterns.add<RemoveStagesRewritePattern, PushBarrierDownRewritePattern>(
          &getContext());
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
}
