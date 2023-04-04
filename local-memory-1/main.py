
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile

from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_ngimel/53/c53h3c5iak33k2o2ay7yesmclfvv53twgryrknrzhetc45m6nlrx.py
# Original ATen:

triton_fused_0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34, out_ptr35, out_ptr36, out_ptr37, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = (xindex // 40)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1600*x1)), None)
    tmp16 = tl.load(in_ptr0 + (40 + x0 + (1600*x1)), None)
    tmp29 = tl.load(in_ptr0 + (80 + x0 + (1600*x1)), None)
    tmp43 = tl.load(in_ptr0 + (160 + x0 + (1600*x1)), None)
    tmp57 = tl.load(in_ptr0 + (200 + x0 + (1600*x1)), None)
    tmp71 = tl.load(in_ptr0 + (240 + x0 + (1600*x1)), None)
    tmp85 = tl.load(in_ptr0 + (280 + x0 + (1600*x1)), None)
    tmp99 = tl.load(in_ptr0 + (320 + x0 + (1600*x1)), None)
    tmp113 = tl.load(in_ptr0 + (360 + x0 + (1600*x1)), None)
    tmp127 = tl.load(in_ptr0 + (400 + x0 + (1600*x1)), None)
    tmp141 = tl.load(in_ptr0 + (440 + x0 + (1600*x1)), None)
    tmp155 = tl.load(in_ptr0 + (480 + x0 + (1600*x1)), None)
    tmp169 = tl.load(in_ptr0 + (520 + x0 + (1600*x1)), None)
    tmp183 = tl.load(in_ptr0 + (560 + x0 + (1600*x1)), None)
    tmp197 = tl.load(in_ptr0 + (600 + x0 + (1600*x1)), None)
    tmp211 = tl.load(in_ptr0 + (640 + x0 + (1600*x1)), None)
    tmp225 = tl.load(in_ptr0 + (680 + x0 + (1600*x1)), None)
    tmp239 = tl.load(in_ptr0 + (720 + x0 + (1600*x1)), None)
    tmp253 = tl.load(in_ptr0 + (760 + x0 + (1600*x1)), None)
    tmp267 = tl.load(in_ptr0 + (800 + x0 + (1600*x1)), None)
    tmp281 = tl.load(in_ptr0 + (840 + x0 + (1600*x1)), None)
    tmp295 = tl.load(in_ptr0 + (880 + x0 + (1600*x1)), None)
    tmp309 = tl.load(in_ptr0 + (920 + x0 + (1600*x1)), None)
    tmp323 = tl.load(in_ptr0 + (960 + x0 + (1600*x1)), None)
    tmp337 = tl.load(in_ptr0 + (1000 + x0 + (1600*x1)), None)
    tmp351 = tl.load(in_ptr0 + (1040 + x0 + (1600*x1)), None)
    tmp365 = tl.load(in_ptr0 + (1080 + x0 + (1600*x1)), None)
    tmp379 = tl.load(in_ptr0 + (1120 + x0 + (1600*x1)), None)
    tmp393 = tl.load(in_ptr0 + (1160 + x0 + (1600*x1)), None)
    tmp407 = tl.load(in_ptr0 + (1200 + x0 + (1600*x1)), None)
    tmp421 = tl.load(in_ptr0 + (1240 + x0 + (1600*x1)), None)
    tmp435 = tl.load(in_ptr0 + (1280 + x0 + (1600*x1)), None)
    tmp449 = tl.load(in_ptr0 + (1320 + x0 + (1600*x1)), None)
    tmp463 = tl.load(in_ptr0 + (1360 + x0 + (1600*x1)), None)
    tmp477 = tl.load(in_ptr0 + (1400 + x0 + (1600*x1)), None)
    tmp491 = tl.load(in_ptr0 + (1440 + x0 + (1600*x1)), None)
    tmp505 = tl.load(in_ptr0 + (1480 + x0 + (1600*x1)), None)
    tmp519 = tl.load(in_ptr0 + (1520 + x0 + (1600*x1)), None)
    tmp1 = 0
    tmp2 = 1
    tmp3 = tmp1 < tmp2
    tmp4 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp3, other=0).to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp3, other=0).to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp3, other=0).to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp11 = tmp5 + tmp10
    tmp12 = tl.where(tmp3, tmp11, 0.0)
    tmp13 = 0.0
    tmp14 = tl.where(tmp3, tmp12, tmp13)
    tmp15 = tmp0 + tmp14
    tmp17 = tmp2 < tmp2
    tmp18 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp17, other=0).to(tl.float32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp17, other=0).to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp17, other=0).to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp19 + tmp24
    tmp26 = tl.where(tmp17, tmp25, 0.0)
    tmp27 = tl.where(tmp17, tmp26, tmp13)
    tmp28 = tmp16 + tmp27
    tmp30 = 2
    tmp31 = tmp30 < tmp2
    tmp32 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp31, other=0).to(tl.float32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp31, other=0).to(tl.float32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp31, other=0).to(tl.float32)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 + tmp37
    tmp39 = tmp33 + tmp38
    tmp40 = tl.where(tmp31, tmp39, 0.0)
    tmp41 = tl.where(tmp31, tmp40, tmp13)
    tmp42 = tmp29 + tmp41
    tmp44 = 4
    tmp45 = tmp44 < tmp2
    tmp46 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp45, other=0).to(tl.float32)
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp45, other=0).to(tl.float32)
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp45, other=0).to(tl.float32)
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp49 + tmp51
    tmp53 = tmp47 + tmp52
    tmp54 = tl.where(tmp45, tmp53, 0.0)
    tmp55 = tl.where(tmp45, tmp54, tmp13)
    tmp56 = tmp43 + tmp55
    tmp58 = 5
    tmp59 = tmp58 < tmp2
    tmp60 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp59, other=0).to(tl.float32)
    tmp61 = tmp60.to(tl.float32)
    tmp62 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp59, other=0).to(tl.float32)
    tmp63 = tmp62.to(tl.float32)
    tmp64 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp59, other=0).to(tl.float32)
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp63 + tmp65
    tmp67 = tmp61 + tmp66
    tmp68 = tl.where(tmp59, tmp67, 0.0)
    tmp69 = tl.where(tmp59, tmp68, tmp13)
    tmp70 = tmp57 + tmp69
    tmp72 = 6
    tmp73 = tmp72 < tmp2
    tmp74 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp73, other=0).to(tl.float32)
    tmp75 = tmp74.to(tl.float32)
    tmp76 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp73, other=0).to(tl.float32)
    tmp77 = tmp76.to(tl.float32)
    tmp78 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp73, other=0).to(tl.float32)
    tmp79 = tmp78.to(tl.float32)
    tmp80 = tmp77 + tmp79
    tmp81 = tmp75 + tmp80
    tmp82 = tl.where(tmp73, tmp81, 0.0)
    tmp83 = tl.where(tmp73, tmp82, tmp13)
    tmp84 = tmp71 + tmp83
    tmp86 = 7
    tmp87 = tmp86 < tmp2
    tmp88 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp87, other=0).to(tl.float32)
    tmp89 = tmp88.to(tl.float32)
    tmp90 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp87, other=0).to(tl.float32)
    tmp91 = tmp90.to(tl.float32)
    tmp92 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp87, other=0).to(tl.float32)
    tmp93 = tmp92.to(tl.float32)
    tmp94 = tmp91 + tmp93
    tmp95 = tmp89 + tmp94
    tmp96 = tl.where(tmp87, tmp95, 0.0)
    tmp97 = tl.where(tmp87, tmp96, tmp13)
    tmp98 = tmp85 + tmp97
    tmp100 = 8
    tmp101 = tmp100 < tmp2
    tmp102 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp101, other=0).to(tl.float32)
    tmp103 = tmp102.to(tl.float32)
    tmp104 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp101, other=0).to(tl.float32)
    tmp105 = tmp104.to(tl.float32)
    tmp106 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp101, other=0).to(tl.float32)
    tmp107 = tmp106.to(tl.float32)
    tmp108 = tmp105 + tmp107
    tmp109 = tmp103 + tmp108
    tmp110 = tl.where(tmp101, tmp109, 0.0)
    tmp111 = tl.where(tmp101, tmp110, tmp13)
    tmp112 = tmp99 + tmp111
    tmp114 = 9
    tmp115 = tmp114 < tmp2
    tmp116 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp115, other=0).to(tl.float32)
    tmp117 = tmp116.to(tl.float32)
    tmp118 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp115, other=0).to(tl.float32)
    tmp119 = tmp118.to(tl.float32)
    tmp120 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp115, other=0).to(tl.float32)
    tmp121 = tmp120.to(tl.float32)
    tmp122 = tmp119 + tmp121
    tmp123 = tmp117 + tmp122
    tmp124 = tl.where(tmp115, tmp123, 0.0)
    tmp125 = tl.where(tmp115, tmp124, tmp13)
    tmp126 = tmp113 + tmp125
    tmp128 = 10
    tmp129 = tmp128 < tmp2
    tmp130 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp129, other=0).to(tl.float32)
    tmp131 = tmp130.to(tl.float32)
    tmp132 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp129, other=0).to(tl.float32)
    tmp133 = tmp132.to(tl.float32)
    tmp134 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp129, other=0).to(tl.float32)
    tmp135 = tmp134.to(tl.float32)
    tmp136 = tmp133 + tmp135
    tmp137 = tmp131 + tmp136
    tmp138 = tl.where(tmp129, tmp137, 0.0)
    tmp139 = tl.where(tmp129, tmp138, tmp13)
    tmp140 = tmp127 + tmp139
    tmp142 = 11
    tmp143 = tmp142 < tmp2
    tmp144 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp143, other=0).to(tl.float32)
    tmp145 = tmp144.to(tl.float32)
    tmp146 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp143, other=0).to(tl.float32)
    tmp147 = tmp146.to(tl.float32)
    tmp148 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp143, other=0).to(tl.float32)
    tmp149 = tmp148.to(tl.float32)
    tmp150 = tmp147 + tmp149
    tmp151 = tmp145 + tmp150
    tmp152 = tl.where(tmp143, tmp151, 0.0)
    tmp153 = tl.where(tmp143, tmp152, tmp13)
    tmp154 = tmp141 + tmp153
    tmp156 = 12
    tmp157 = tmp156 < tmp2
    tmp158 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp157, other=0).to(tl.float32)
    tmp159 = tmp158.to(tl.float32)
    tmp160 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp157, other=0).to(tl.float32)
    tmp161 = tmp160.to(tl.float32)
    tmp162 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp157, other=0).to(tl.float32)
    tmp163 = tmp162.to(tl.float32)
    tmp164 = tmp161 + tmp163
    tmp165 = tmp159 + tmp164
    tmp166 = tl.where(tmp157, tmp165, 0.0)
    tmp167 = tl.where(tmp157, tmp166, tmp13)
    tmp168 = tmp155 + tmp167
    tmp170 = 13
    tmp171 = tmp170 < tmp2
    tmp172 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp171, other=0).to(tl.float32)
    tmp173 = tmp172.to(tl.float32)
    tmp174 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp171, other=0).to(tl.float32)
    tmp175 = tmp174.to(tl.float32)
    tmp176 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp171, other=0).to(tl.float32)
    tmp177 = tmp176.to(tl.float32)
    tmp178 = tmp175 + tmp177
    tmp179 = tmp173 + tmp178
    tmp180 = tl.where(tmp171, tmp179, 0.0)
    tmp181 = tl.where(tmp171, tmp180, tmp13)
    tmp182 = tmp169 + tmp181
    tmp184 = 14
    tmp185 = tmp184 < tmp2
    tmp186 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp185, other=0).to(tl.float32)
    tmp187 = tmp186.to(tl.float32)
    tmp188 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp185, other=0).to(tl.float32)
    tmp189 = tmp188.to(tl.float32)
    tmp190 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp185, other=0).to(tl.float32)
    tmp191 = tmp190.to(tl.float32)
    tmp192 = tmp189 + tmp191
    tmp193 = tmp187 + tmp192
    tmp194 = tl.where(tmp185, tmp193, 0.0)
    tmp195 = tl.where(tmp185, tmp194, tmp13)
    tmp196 = tmp183 + tmp195
    tmp198 = 15
    tmp199 = tmp198 < tmp2
    tmp200 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp199, other=0).to(tl.float32)
    tmp201 = tmp200.to(tl.float32)
    tmp202 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp199, other=0).to(tl.float32)
    tmp203 = tmp202.to(tl.float32)
    tmp204 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp199, other=0).to(tl.float32)
    tmp205 = tmp204.to(tl.float32)
    tmp206 = tmp203 + tmp205
    tmp207 = tmp201 + tmp206
    tmp208 = tl.where(tmp199, tmp207, 0.0)
    tmp209 = tl.where(tmp199, tmp208, tmp13)
    tmp210 = tmp197 + tmp209
    tmp212 = 16
    tmp213 = tmp212 < tmp2
    tmp214 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp213, other=0).to(tl.float32)
    tmp215 = tmp214.to(tl.float32)
    tmp216 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp213, other=0).to(tl.float32)
    tmp217 = tmp216.to(tl.float32)
    tmp218 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp213, other=0).to(tl.float32)
    tmp219 = tmp218.to(tl.float32)
    tmp220 = tmp217 + tmp219
    tmp221 = tmp215 + tmp220
    tmp222 = tl.where(tmp213, tmp221, 0.0)
    tmp223 = tl.where(tmp213, tmp222, tmp13)
    tmp224 = tmp211 + tmp223
    tmp226 = 17
    tmp227 = tmp226 < tmp2
    tmp228 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp227, other=0).to(tl.float32)
    tmp229 = tmp228.to(tl.float32)
    tmp230 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp227, other=0).to(tl.float32)
    tmp231 = tmp230.to(tl.float32)
    tmp232 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp227, other=0).to(tl.float32)
    tmp233 = tmp232.to(tl.float32)
    tmp234 = tmp231 + tmp233
    tmp235 = tmp229 + tmp234
    tmp236 = tl.where(tmp227, tmp235, 0.0)
    tmp237 = tl.where(tmp227, tmp236, tmp13)
    tmp238 = tmp225 + tmp237
    tmp240 = 18
    tmp241 = tmp240 < tmp2
    tmp242 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp241, other=0).to(tl.float32)
    tmp243 = tmp242.to(tl.float32)
    tmp244 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp241, other=0).to(tl.float32)
    tmp245 = tmp244.to(tl.float32)
    tmp246 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp241, other=0).to(tl.float32)
    tmp247 = tmp246.to(tl.float32)
    tmp248 = tmp245 + tmp247
    tmp249 = tmp243 + tmp248
    tmp250 = tl.where(tmp241, tmp249, 0.0)
    tmp251 = tl.where(tmp241, tmp250, tmp13)
    tmp252 = tmp239 + tmp251
    tmp254 = 19
    tmp255 = tmp254 < tmp2
    tmp256 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp255, other=0).to(tl.float32)
    tmp257 = tmp256.to(tl.float32)
    tmp258 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp255, other=0).to(tl.float32)
    tmp259 = tmp258.to(tl.float32)
    tmp260 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp255, other=0).to(tl.float32)
    tmp261 = tmp260.to(tl.float32)
    tmp262 = tmp259 + tmp261
    tmp263 = tmp257 + tmp262
    tmp264 = tl.where(tmp255, tmp263, 0.0)
    tmp265 = tl.where(tmp255, tmp264, tmp13)
    tmp266 = tmp253 + tmp265
    tmp268 = 20
    tmp269 = tmp268 < tmp2
    tmp270 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp269, other=0).to(tl.float32)
    tmp271 = tmp270.to(tl.float32)
    tmp272 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp269, other=0).to(tl.float32)
    tmp273 = tmp272.to(tl.float32)
    tmp274 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp269, other=0).to(tl.float32)
    tmp275 = tmp274.to(tl.float32)
    tmp276 = tmp273 + tmp275
    tmp277 = tmp271 + tmp276
    tmp278 = tl.where(tmp269, tmp277, 0.0)
    tmp279 = tl.where(tmp269, tmp278, tmp13)
    tmp280 = tmp267 + tmp279
    tmp282 = 21
    tmp283 = tmp282 < tmp2
    tmp284 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp283, other=0).to(tl.float32)
    tmp285 = tmp284.to(tl.float32)
    tmp286 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp283, other=0).to(tl.float32)
    tmp287 = tmp286.to(tl.float32)
    tmp288 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp283, other=0).to(tl.float32)
    tmp289 = tmp288.to(tl.float32)
    tmp290 = tmp287 + tmp289
    tmp291 = tmp285 + tmp290
    tmp292 = tl.where(tmp283, tmp291, 0.0)
    tmp293 = tl.where(tmp283, tmp292, tmp13)
    tmp294 = tmp281 + tmp293
    tmp296 = 22
    tmp297 = tmp296 < tmp2
    tmp298 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp297, other=0).to(tl.float32)
    tmp299 = tmp298.to(tl.float32)
    tmp300 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp297, other=0).to(tl.float32)
    tmp301 = tmp300.to(tl.float32)
    tmp302 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp297, other=0).to(tl.float32)
    tmp303 = tmp302.to(tl.float32)
    tmp304 = tmp301 + tmp303
    tmp305 = tmp299 + tmp304
    tmp306 = tl.where(tmp297, tmp305, 0.0)
    tmp307 = tl.where(tmp297, tmp306, tmp13)
    tmp308 = tmp295 + tmp307
    tmp310 = 23
    tmp311 = tmp310 < tmp2
    tmp312 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp311, other=0).to(tl.float32)
    tmp313 = tmp312.to(tl.float32)
    tmp314 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp311, other=0).to(tl.float32)
    tmp315 = tmp314.to(tl.float32)
    tmp316 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp311, other=0).to(tl.float32)
    tmp317 = tmp316.to(tl.float32)
    tmp318 = tmp315 + tmp317
    tmp319 = tmp313 + tmp318
    tmp320 = tl.where(tmp311, tmp319, 0.0)
    tmp321 = tl.where(tmp311, tmp320, tmp13)
    tmp322 = tmp309 + tmp321
    tmp324 = 24
    tmp325 = tmp324 < tmp2
    tmp326 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp325, other=0).to(tl.float32)
    tmp327 = tmp326.to(tl.float32)
    tmp328 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp325, other=0).to(tl.float32)
    tmp329 = tmp328.to(tl.float32)
    tmp330 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp325, other=0).to(tl.float32)
    tmp331 = tmp330.to(tl.float32)
    tmp332 = tmp329 + tmp331
    tmp333 = tmp327 + tmp332
    tmp334 = tl.where(tmp325, tmp333, 0.0)
    tmp335 = tl.where(tmp325, tmp334, tmp13)
    tmp336 = tmp323 + tmp335
    tmp338 = 25
    tmp339 = tmp338 < tmp2
    tmp340 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp339, other=0).to(tl.float32)
    tmp341 = tmp340.to(tl.float32)
    tmp342 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp339, other=0).to(tl.float32)
    tmp343 = tmp342.to(tl.float32)
    tmp344 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp339, other=0).to(tl.float32)
    tmp345 = tmp344.to(tl.float32)
    tmp346 = tmp343 + tmp345
    tmp347 = tmp341 + tmp346
    tmp348 = tl.where(tmp339, tmp347, 0.0)
    tmp349 = tl.where(tmp339, tmp348, tmp13)
    tmp350 = tmp337 + tmp349
    tmp352 = 26
    tmp353 = tmp352 < tmp2
    tmp354 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp353, other=0).to(tl.float32)
    tmp355 = tmp354.to(tl.float32)
    tmp356 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp353, other=0).to(tl.float32)
    tmp357 = tmp356.to(tl.float32)
    tmp358 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp353, other=0).to(tl.float32)
    tmp359 = tmp358.to(tl.float32)
    tmp360 = tmp357 + tmp359
    tmp361 = tmp355 + tmp360
    tmp362 = tl.where(tmp353, tmp361, 0.0)
    tmp363 = tl.where(tmp353, tmp362, tmp13)
    tmp364 = tmp351 + tmp363
    tmp366 = 27
    tmp367 = tmp366 < tmp2
    tmp368 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp367, other=0).to(tl.float32)
    tmp369 = tmp368.to(tl.float32)
    tmp370 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp367, other=0).to(tl.float32)
    tmp371 = tmp370.to(tl.float32)
    tmp372 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp367, other=0).to(tl.float32)
    tmp373 = tmp372.to(tl.float32)
    tmp374 = tmp371 + tmp373
    tmp375 = tmp369 + tmp374
    tmp376 = tl.where(tmp367, tmp375, 0.0)
    tmp377 = tl.where(tmp367, tmp376, tmp13)
    tmp378 = tmp365 + tmp377
    tmp380 = 28
    tmp381 = tmp380 < tmp2
    tmp382 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp381, other=0).to(tl.float32)
    tmp383 = tmp382.to(tl.float32)
    tmp384 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp381, other=0).to(tl.float32)
    tmp385 = tmp384.to(tl.float32)
    tmp386 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp381, other=0).to(tl.float32)
    tmp387 = tmp386.to(tl.float32)
    tmp388 = tmp385 + tmp387
    tmp389 = tmp383 + tmp388
    tmp390 = tl.where(tmp381, tmp389, 0.0)
    tmp391 = tl.where(tmp381, tmp390, tmp13)
    tmp392 = tmp379 + tmp391
    tmp394 = 29
    tmp395 = tmp394 < tmp2
    tmp396 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp395, other=0).to(tl.float32)
    tmp397 = tmp396.to(tl.float32)
    tmp398 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp395, other=0).to(tl.float32)
    tmp399 = tmp398.to(tl.float32)
    tmp400 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp395, other=0).to(tl.float32)
    tmp401 = tmp400.to(tl.float32)
    tmp402 = tmp399 + tmp401
    tmp403 = tmp397 + tmp402
    tmp404 = tl.where(tmp395, tmp403, 0.0)
    tmp405 = tl.where(tmp395, tmp404, tmp13)
    tmp406 = tmp393 + tmp405
    tmp408 = 30
    tmp409 = tmp408 < tmp2
    tmp410 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp409, other=0).to(tl.float32)
    tmp411 = tmp410.to(tl.float32)
    tmp412 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp409, other=0).to(tl.float32)
    tmp413 = tmp412.to(tl.float32)
    tmp414 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp409, other=0).to(tl.float32)
    tmp415 = tmp414.to(tl.float32)
    tmp416 = tmp413 + tmp415
    tmp417 = tmp411 + tmp416
    tmp418 = tl.where(tmp409, tmp417, 0.0)
    tmp419 = tl.where(tmp409, tmp418, tmp13)
    tmp420 = tmp407 + tmp419
    tmp422 = 31
    tmp423 = tmp422 < tmp2
    tmp424 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp423, other=0).to(tl.float32)
    tmp425 = tmp424.to(tl.float32)
    tmp426 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp423, other=0).to(tl.float32)
    tmp427 = tmp426.to(tl.float32)
    tmp428 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp423, other=0).to(tl.float32)
    tmp429 = tmp428.to(tl.float32)
    tmp430 = tmp427 + tmp429
    tmp431 = tmp425 + tmp430
    tmp432 = tl.where(tmp423, tmp431, 0.0)
    tmp433 = tl.where(tmp423, tmp432, tmp13)
    tmp434 = tmp421 + tmp433
    tmp436 = 32
    tmp437 = tmp436 < tmp2
    tmp438 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp437, other=0).to(tl.float32)
    tmp439 = tmp438.to(tl.float32)
    tmp440 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp437, other=0).to(tl.float32)
    tmp441 = tmp440.to(tl.float32)
    tmp442 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp437, other=0).to(tl.float32)
    tmp443 = tmp442.to(tl.float32)
    tmp444 = tmp441 + tmp443
    tmp445 = tmp439 + tmp444
    tmp446 = tl.where(tmp437, tmp445, 0.0)
    tmp447 = tl.where(tmp437, tmp446, tmp13)
    tmp448 = tmp435 + tmp447
    tmp450 = 33
    tmp451 = tmp450 < tmp2
    tmp452 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp451, other=0).to(tl.float32)
    tmp453 = tmp452.to(tl.float32)
    tmp454 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp451, other=0).to(tl.float32)
    tmp455 = tmp454.to(tl.float32)
    tmp456 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp451, other=0).to(tl.float32)
    tmp457 = tmp456.to(tl.float32)
    tmp458 = tmp455 + tmp457
    tmp459 = tmp453 + tmp458
    tmp460 = tl.where(tmp451, tmp459, 0.0)
    tmp461 = tl.where(tmp451, tmp460, tmp13)
    tmp462 = tmp449 + tmp461
    tmp464 = 34
    tmp465 = tmp464 < tmp2
    tmp466 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp465, other=0).to(tl.float32)
    tmp467 = tmp466.to(tl.float32)
    tmp468 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp465, other=0).to(tl.float32)
    tmp469 = tmp468.to(tl.float32)
    tmp470 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp465, other=0).to(tl.float32)
    tmp471 = tmp470.to(tl.float32)
    tmp472 = tmp469 + tmp471
    tmp473 = tmp467 + tmp472
    tmp474 = tl.where(tmp465, tmp473, 0.0)
    tmp475 = tl.where(tmp465, tmp474, tmp13)
    tmp476 = tmp463 + tmp475
    tmp478 = 35
    tmp479 = tmp478 < tmp2
    tmp480 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp479, other=0).to(tl.float32)
    tmp481 = tmp480.to(tl.float32)
    tmp482 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp479, other=0).to(tl.float32)
    tmp483 = tmp482.to(tl.float32)
    tmp484 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp479, other=0).to(tl.float32)
    tmp485 = tmp484.to(tl.float32)
    tmp486 = tmp483 + tmp485
    tmp487 = tmp481 + tmp486
    tmp488 = tl.where(tmp479, tmp487, 0.0)
    tmp489 = tl.where(tmp479, tmp488, tmp13)
    tmp490 = tmp477 + tmp489
    tmp492 = 36
    tmp493 = tmp492 < tmp2
    tmp494 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp493, other=0).to(tl.float32)
    tmp495 = tmp494.to(tl.float32)
    tmp496 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp493, other=0).to(tl.float32)
    tmp497 = tmp496.to(tl.float32)
    tmp498 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp493, other=0).to(tl.float32)
    tmp499 = tmp498.to(tl.float32)
    tmp500 = tmp497 + tmp499
    tmp501 = tmp495 + tmp500
    tmp502 = tl.where(tmp493, tmp501, 0.0)
    tmp503 = tl.where(tmp493, tmp502, tmp13)
    tmp504 = tmp491 + tmp503
    tmp506 = 37
    tmp507 = tmp506 < tmp2
    tmp508 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp507, other=0).to(tl.float32)
    tmp509 = tmp508.to(tl.float32)
    tmp510 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp507, other=0).to(tl.float32)
    tmp511 = tmp510.to(tl.float32)
    tmp512 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp507, other=0).to(tl.float32)
    tmp513 = tmp512.to(tl.float32)
    tmp514 = tmp511 + tmp513
    tmp515 = tmp509 + tmp514
    tmp516 = tl.where(tmp507, tmp515, 0.0)
    tmp517 = tl.where(tmp507, tmp516, tmp13)
    tmp518 = tmp505 + tmp517
    tmp520 = 38
    tmp521 = tmp520 < tmp2
    tmp522 = tl.load(in_ptr1 + (400 + x0 + (2640*x1) + tl.zeros([XBLOCK], tl.int32)), tmp521, other=0).to(tl.float32)
    tmp523 = tmp522.to(tl.float32)
    tmp524 = tl.load(in_ptr2 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp521, other=0).to(tl.float32)
    tmp525 = tmp524.to(tl.float32)
    tmp526 = tl.load(in_ptr3 + (10 + (66*x2) + tl.zeros([XBLOCK], tl.int32)), tmp521, other=0).to(tl.float32)
    tmp527 = tmp526.to(tl.float32)
    tmp528 = tmp525 + tmp527
    tmp529 = tmp523 + tmp528
    tmp530 = tl.where(tmp521, tmp529, 0.0)
    tmp531 = tl.where(tmp521, tmp530, tmp13)
    tmp532 = tmp519 + tmp531
    tl.store(out_ptr0 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp15, None)
    tl.store(out_ptr1 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp28, None)
    tl.store(out_ptr2 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp42, None)
    tl.store(out_ptr3 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp56, None)
    tl.store(out_ptr4 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp70, None)
    tl.store(out_ptr5 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp84, None)
    tl.store(out_ptr6 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp98, None)
    tl.store(out_ptr7 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp112, None)
    tl.store(out_ptr8 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp126, None)
    tl.store(out_ptr9 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp140, None)
    tl.store(out_ptr10 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp154, None)
    tl.store(out_ptr11 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp168, None)
    tl.store(out_ptr12 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp182, None)
    tl.store(out_ptr13 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp196, None)
    tl.store(out_ptr14 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp210, None)
    tl.store(out_ptr15 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp224, None)
    tl.store(out_ptr16 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp238, None)
    tl.store(out_ptr17 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp252, None)
    tl.store(out_ptr18 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp266, None)
    tl.store(out_ptr19 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp280, None)
    tl.store(out_ptr20 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp294, None)
    tl.store(out_ptr21 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp308, None)
    tl.store(out_ptr22 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp322, None)
    tl.store(out_ptr23 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp336, None)
    tl.store(out_ptr24 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp350, None)
    tl.store(out_ptr25 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp364, None)
    tl.store(out_ptr26 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp378, None)
    tl.store(out_ptr27 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp392, None)
    tl.store(out_ptr28 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp406, None)
    tl.store(out_ptr29 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp420, None)
    tl.store(out_ptr30 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp434, None)
    tl.store(out_ptr31 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp448, None)
    tl.store(out_ptr32 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp462, None)
    tl.store(out_ptr33 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp476, None)
    tl.store(out_ptr34 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp490, None)
    tl.store(out_ptr35 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp504, None)
    tl.store(out_ptr36 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp518, None)
    tl.store(out_ptr37 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp532, None)
''')


# kernel path: /tmp/torchinductor_ngimel/q6/cq6d4lii4iqclamjluckz5cndfr7rzu6o2n3kfkptnhnazgixbt5.py
# Original ATen:

triton_fused_1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = (xindex // 40)
    tmp0 = tl.load(in_ptr0 + (x0 + (1600*x1)), None)
    tl.store(out_ptr0 + (x0 + (1600*x1) + tl.zeros([XBLOCK], tl.int32)), tmp0, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf40 = empty_strided((1024, 1600), (1600, 1), device='cuda', dtype=torch.float32)
        buf0 = as_strided(buf40, (1024, 40), (1600, 1))  # alias
        buf1 = as_strided(buf40, (1024, 40), (1600, 1), 40)  # alias
        buf2 = as_strided(buf40, (1024, 40), (1600, 1), 80)  # alias
        buf4 = as_strided(buf40, (1024, 40), (1600, 1), 160)  # alias
        buf5 = as_strided(buf40, (1024, 40), (1600, 1), 200)  # alias
        buf6 = as_strided(buf40, (1024, 40), (1600, 1), 240)  # alias
        buf7 = as_strided(buf40, (1024, 40), (1600, 1), 280)  # alias
        buf8 = as_strided(buf40, (1024, 40), (1600, 1), 320)  # alias
        buf9 = as_strided(buf40, (1024, 40), (1600, 1), 360)  # alias
        buf10 = as_strided(buf40, (1024, 40), (1600, 1), 400)  # alias
        buf11 = as_strided(buf40, (1024, 40), (1600, 1), 440)  # alias
        buf12 = as_strided(buf40, (1024, 40), (1600, 1), 480)  # alias
        buf13 = as_strided(buf40, (1024, 40), (1600, 1), 520)  # alias
        buf14 = as_strided(buf40, (1024, 40), (1600, 1), 560)  # alias
        buf15 = as_strided(buf40, (1024, 40), (1600, 1), 600)  # alias
        buf16 = as_strided(buf40, (1024, 40), (1600, 1), 640)  # alias
        buf17 = as_strided(buf40, (1024, 40), (1600, 1), 680)  # alias
        buf18 = as_strided(buf40, (1024, 40), (1600, 1), 720)  # alias
        buf19 = as_strided(buf40, (1024, 40), (1600, 1), 760)  # alias
        buf20 = as_strided(buf40, (1024, 40), (1600, 1), 800)  # alias
        buf21 = as_strided(buf40, (1024, 40), (1600, 1), 840)  # alias
        buf22 = as_strided(buf40, (1024, 40), (1600, 1), 880)  # alias
        buf23 = as_strided(buf40, (1024, 40), (1600, 1), 920)  # alias
        buf24 = as_strided(buf40, (1024, 40), (1600, 1), 960)  # alias
        buf25 = as_strided(buf40, (1024, 40), (1600, 1), 1000)  # alias
        buf26 = as_strided(buf40, (1024, 40), (1600, 1), 1040)  # alias
        buf27 = as_strided(buf40, (1024, 40), (1600, 1), 1080)  # alias
        buf28 = as_strided(buf40, (1024, 40), (1600, 1), 1120)  # alias
        buf29 = as_strided(buf40, (1024, 40), (1600, 1), 1160)  # alias
        buf30 = as_strided(buf40, (1024, 40), (1600, 1), 1200)  # alias
        buf31 = as_strided(buf40, (1024, 40), (1600, 1), 1240)  # alias
        buf32 = as_strided(buf40, (1024, 40), (1600, 1), 1280)  # alias
        buf33 = as_strided(buf40, (1024, 40), (1600, 1), 1320)  # alias
        buf34 = as_strided(buf40, (1024, 40), (1600, 1), 1360)  # alias
        buf35 = as_strided(buf40, (1024, 40), (1600, 1), 1400)  # alias
        buf36 = as_strided(buf40, (1024, 40), (1600, 1), 1440)  # alias
        buf37 = as_strided(buf40, (1024, 40), (1600, 1), 1480)  # alias
        buf38 = as_strided(buf40, (1024, 40), (1600, 1), 1520)  # alias
        stream0 = get_cuda_stream(0)
        triton_fused_0.run(arg2_1, arg0_1, arg1_1, arg3_1, buf0, buf1, buf2, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, 40960, grid=grid(40960), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        buf3 = as_strided(buf40, (1024, 40), (1600, 1), 120)  # alias
        triton_fused_1.run(arg5_1, buf3, 40960, grid=grid(40960), stream=stream0)
        del arg5_1
        buf39 = as_strided(buf40, (1024, 40), (1600, 1), 1560)  # alias
        triton_fused_1.run(arg4_1, buf39, 40960, grid=grid(40960), stream=stream0)
        del arg4_1
        return (buf40, )


def benchmark_compiled_module():
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1024, 66, 40), (2640, 40, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((1024, 40, 66), (2640, 66, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1024, 40, 40), (1600, 40, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((40960, 66), (66, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((1024, 40), (1600, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((1024, 40), (1600, 1), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]))


if __name__ == "__main__":
    import argparse
    from torch._inductor.utils import benchmark_all_kernels

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-kernels", "-k", action="store_true", help="Whether to benchmark each individual kernels")
    parser.add_argument("--benchmark-all-configs", "-c", action="store_true", help="Whether to benchmark each individual config for a kernel")
    parser.add_argument("--profile", "-p", action="store_true", help="Whether to profile the compiled module")
    args = parser.parse_args()

    if args.benchmark_kernels:
        benchmark_all_kernels('None', args.benchmark_all_configs)
    else:
        benchmark_compiled_module()

