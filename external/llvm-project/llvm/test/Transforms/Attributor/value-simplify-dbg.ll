; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --function-signature --check-attributes --check-globals
; RUN: opt -aa-pipeline=basic-aa -passes=attributor -attributor-manifest-internal  -attributor-max-iterations-verify -attributor-annotate-decl-cs -attributor-max-iterations=2 -S < %s | FileCheck %s --check-prefixes=CHECK,NOT_CGSCC_OPM,NOT_CGSCC_NPM,NOT_TUNIT_OPM,IS__TUNIT____,IS________NPM,IS__TUNIT_NPM
; RUN: opt -aa-pipeline=basic-aa -passes=attributor-cgscc -attributor-manifest-internal  -attributor-annotate-decl-cs -S < %s | FileCheck %s --check-prefixes=CHECK,NOT_TUNIT_NPM,NOT_TUNIT_OPM,NOT_CGSCC_OPM,IS__CGSCC____,IS________NPM,IS__CGSCC_NPM

@G = internal global i32 undef, align 4, !dbg !0

;.
; CHECK: @[[G:[a-zA-Z0-9_$"\\.-]+]] = internal global i32 undef, align 4, !dbg [[DBG0:![0-9]+]]
;.
define void @dest() !dbg !15 {
; CHECK-LABEL: define {{[^@]+}}@dest
; CHECK-SAME: () !dbg [[DBG15:![0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr @G, align 4, !dbg [[DBG19:![0-9]+]]
; CHECK-NEXT:    call void @use(i32 noundef [[TMP0]]), !dbg [[DBG20:![0-9]+]]
; CHECK-NEXT:    ret void, !dbg [[DBG21:![0-9]+]]
;
entry:
  %0 = load i32, ptr @G, align 4, !dbg !19
  call void @use(i32 noundef %0), !dbg !20
  ret void, !dbg !21
}

declare void @use(i32 noundef)

define void @src() norecurse !dbg !22 {
; CHECK: Function Attrs: norecurse nosync writeonly
; CHECK-LABEL: define {{[^@]+}}@src
; CHECK-SAME: () #[[ATTR0:[0-9]+]] !dbg [[DBG22:![0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @speculatable(), !dbg [[DBG23:![0-9]+]]
; CHECK-NEXT:    [[PLUS1:%.*]] = add i32 [[CALL]], 1
; CHECK-NEXT:    store i32 [[PLUS1]], ptr @G, align 4, !dbg [[DBG24:![0-9]+]]
; CHECK-NEXT:    ret void, !dbg [[DBG25:![0-9]+]]
;
entry:
  %call = call i32 @speculatable(), !dbg !23
  %plus1 = add i32 %call, 1
  store i32 %plus1, ptr @G, align 4, !dbg !24
  ret void, !dbg !25
}

declare i32 @speculatable() speculatable readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "G", scope: !2, file: !5, line: 1, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git ef94609d6ebe981767788e6877b0b3b731d425af)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "/app/example.c", directory: "/app", checksumkind: CSK_MD5, checksum: "b456b90cec5c3705a028b274d88ee970")
!4 = !{!0}
!5 = !DIFile(filename: "example.c", directory: "/app", checksumkind: CSK_MD5, checksum: "b456b90cec5c3705a028b274d88ee970")
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 8, !"PIC Level", i32 2}
!11 = !{i32 7, !"PIE Level", i32 2}
!12 = !{i32 7, !"uwtable", i32 2}
!13 = !{i32 7, !"frame-pointer", i32 2}
!14 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git ef94609d6ebe981767788e6877b0b3b731d425af)"}
!15 = distinct !DISubprogram(name: "dest", scope: !5, file: !5, line: 4, type: !16, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
!16 = !DISubroutineType(types: !17)
!17 = !{null}
!18 = !{}
!19 = !DILocation(line: 5, column: 9, scope: !15)
!20 = !DILocation(line: 5, column: 5, scope: !15)
!21 = !DILocation(line: 6, column: 1, scope: !15)
!22 = distinct !DISubprogram(name: "src", scope: !5, file: !5, line: 9, type: !16, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
!23 = !DILocation(line: 10, column: 9, scope: !22)
!24 = !DILocation(line: 10, column: 7, scope: !22)
!25 = !DILocation(line: 11, column: 1, scope: !22)
;.
; CHECK: attributes #[[ATTR0]] = { norecurse nosync writeonly }
; CHECK: attributes #[[ATTR1:[0-9]+]] = { readnone speculatable }
;.
; CHECK: [[DBG0]] = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
; CHECK: [[META1:![0-9]+]] = distinct !DIGlobalVariable(name: "G", scope: !2, file: !5, line: 1, type: !6, isLocal: true, isDefinition: true)
; CHECK: [[META2:![0-9]+]] = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git ef94609d6ebe981767788e6877b0b3b731d425af)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
; CHECK: [[META3:![0-9]+]] = !DIFile(filename: "/app/example.c", directory: "/app", checksumkind: CSK_MD5, checksum: "b456b90cec5c3705a028b274d88ee970")
; CHECK: [[META4:![0-9]+]] = !{!0}
; CHECK: [[META5:![0-9]+]] = !DIFile(filename: "example.c", directory: "/app", checksumkind: CSK_MD5, checksum: "b456b90cec5c3705a028b274d88ee970")
; CHECK: [[META6:![0-9]+]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
; CHECK: [[META7:![0-9]+]] = !{i32 7, !"Dwarf Version", i32 5}
; CHECK: [[META8:![0-9]+]] = !{i32 2, !"Debug Info Version", i32 3}
; CHECK: [[META9:![0-9]+]] = !{i32 1, !"wchar_size", i32 4}
; CHECK: [[META10:![0-9]+]] = !{i32 8, !"PIC Level", i32 2}
; CHECK: [[META11:![0-9]+]] = !{i32 7, !"PIE Level", i32 2}
; CHECK: [[META12:![0-9]+]] = !{i32 7, !"uwtable", i32 2}
; CHECK: [[META13:![0-9]+]] = !{i32 7, !"frame-pointer", i32 2}
; CHECK: [[META14:![0-9]+]] = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git ef94609d6ebe981767788e6877b0b3b731d425af)"}
; CHECK: [[DBG15]] = distinct !DISubprogram(name: "dest", scope: !5, file: !5, line: 4, type: !16, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
; CHECK: [[META16:![0-9]+]] = !DISubroutineType(types: !17)
; CHECK: [[META17:![0-9]+]] = !{null}
; CHECK: [[META18:![0-9]+]] = !{}
; CHECK: [[DBG19]] = !DILocation(line: 5, column: 9, scope: !15)
; CHECK: [[DBG20]] = !DILocation(line: 5, column: 5, scope: !15)
; CHECK: [[DBG21]] = !DILocation(line: 6, column: 1, scope: !15)
; CHECK: [[DBG22]] = distinct !DISubprogram(name: "src", scope: !5, file: !5, line: 9, type: !16, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
; CHECK: [[DBG23]] = !DILocation(line: 10, column: 9, scope: !22)
; CHECK: [[DBG24]] = !DILocation(line: 10, column: 7, scope: !22)
; CHECK: [[DBG25]] = !DILocation(line: 11, column: 1, scope: !22)
;.
