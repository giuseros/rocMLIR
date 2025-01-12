                                                            // Restore once kPack + padding work
                                                            // --perf_config 256,128,8,64,128,8,0,1
// RUN: rocmlir-gen --arch %arch --operation conv2d_bwd_weight --perf_config 256,128,8,64,128,1,0,1 -t f16 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 256 --in_channels 64 --in_h 56 --in_w 56 --out_channels 64 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 %pv %random_data -mfma=on | rocmlir-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_F16_WRW_PERF_CFG

// CHECK_F16_WRW_PERF_CFG: [1 1 1]
