grep after_soft_max_unpack_noscale run.log | sort &> after_soft_max_unpack_noscale.log
grep after_qk_gemm run.log | sort &> after_qk_gemm.log
grep after_mask run.log | sort &> after_mask.log
grep Gmem_tile_mask_load_after run.log | sort &> Gmem_tile_mask_load_after.log
grep Gmem_tile_mask_load_before run.log | sort &> Gmem_tile_mask_load_before.log
grep AttnMask run.log | sort &> AttnMask.log