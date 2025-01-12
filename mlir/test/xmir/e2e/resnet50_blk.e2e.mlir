// RUN: rocmlir-driver -host-pipeline partition,highlevel -targets %arch,gfx908,gfx90a %s | rocmlir-gen -ph -print-results -rand_type float -rand 1 -fut resnet50 - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full -targets %arch,gfx908,gfx90a | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-driver -host-pipeline partition,highlevel -targets %arch,gfx908,gfx90a %s | rocmlir-gen -ph -print-results -rand_type float -rand 1 -fut resnet50 --verifier clone - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full -targets %arch,gfx908,gfx90a | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE

module {
// CHECK: Unranked Memref base@ = 0x{{.*}} rank = 4 offset = 0 sizes = [1, 32, 32, 64] strides = [65536, 2048, 64, 1] data =
// CHECK-NEXT: 0.680375,     -0.211234,     0.566198,     6.59688,     0.823295,     -0.604897,     5.67045,     6.53646,     5.55555,     6.10794,

// CLONE: Number of elements: 65536
// CLONE-NEXT: maxAbsDiff info:
// CLONE-NEXT: maxRelDiff info:
// CLONE-NEXT: RMS =
// CLONE-NEXT: Histogram of relDiff:
// CLONE: [1 1 0]
// CLONE-NEXT: Unranked Memref base@ = 0x{{.*}} rank = 4 offset = 0 sizes = [1, 32, 32, 64] strides = [65536, 2048, 64, 1] data =
// CLONE-NEXT: 0.680375,     -0.211234,     0.566198,     6.59688,     0.823295,     -0.604897,     5.67045,     6.53646,     5.55555,     6.10794,

  func.func @resnet50(%arg0: tensor<1x32x32x64xf32>, %arg1: tensor<64x3x3x64xf32>, %arg2: tensor<64x3x3x64xf32>) -> tensor<1x32x32x64xf32> {

    %cst = arith.constant dense<0.0> : tensor<64xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %cst) {
      dilation = [1, 1],
      pad = [1, 1, 1, 1],
      stride = [1, 1]
    }
     : (tensor<1x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>

    %1 = "tosa.clamp"(%0) {
      min_fp = 0.0 : f32,
      max_fp = 6.0 : f32,
      min_int = 0 : i64,
      max_int = 6 : i64
    }
     : (tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>

    //%cst1 = arith.constant dense<0.0> : tensor<64xf32>
    %2 = "tosa.conv2d"(%1, %arg2, %cst) {
      dilation = [1, 1],
      pad = [1, 1, 1, 1],
      stride = [1, 1]
    }
     : (tensor<1x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>

    %3 = "tosa.clamp"(%2) {
      min_fp = 0.0 : f32,
      max_fp = 6.0 : f32,
      min_int = 0 : i64,
      max_int = 6 : i64
    }
     : (tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>

    %4 = "tosa.add"(%arg0, %3)
     : (tensor<1x32x32x64xf32>, tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>

    return %4 : tensor<1x32x32x64xf32>
  }
}
