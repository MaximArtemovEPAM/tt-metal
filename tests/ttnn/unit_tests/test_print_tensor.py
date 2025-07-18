# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype


GOLDEN_TENSOR_STRINGS = {
    (
        ttnn.uint16,
        ttnn.ROW_MAJOR_LAYOUT,
    ): "ttnn.Tensor([[[[  684,   559,  ...,   777,   916],\n               [  115,   976,  ...,   459,   882],\n               ...,\n               [  649,   773,  ...,   778,   555],\n               [  955,   414,  ...,   389,   378]],\n\n              [[  856,   273,  ...,   632,     2],\n               [  785,   143,  ...,   358,   404],\n               ...,\n               [  738,   150,  ...,   423,   609],\n               [  105,   687,  ...,   580,   862]],\n\n              ...,\n\n              [[  409,   607,  ...,   290,   816],\n               [  375,   306,  ...,   954,   218],\n               ...,\n               [  204,   718,  ...,   130,   890],\n               [  653,   250,  ...,   282,   825]],\n\n              [[  707,   273,  ...,   437,   848],\n               [  591,    14,  ...,   882,   546],\n               ...,\n               [  670,   571,  ...,   178,    24],\n               [    0,  1017,  ...,   664,   815]]],\n\n             [[[   27,     5,  ...,   401,   490],\n               [  136,   533,  ...,   688,   427],\n               ...,\n               [  827,  1018,  ...,   595,   431],\n               [  649,   238,  ...,   872,   741]],\n\n              [[  143,   142,  ...,   440,   812],\n               [  872,    76,  ...,   305,   892],\n               ...,\n               [  193,    83,  ...,   940,   404],\n               [  987,    69,  ...,   368,   413]],\n\n              ...,\n\n              [[  839,   704,  ...,   218,   229],\n               [  363,   605,  ...,   857,   928],\n               ...,\n               [  708,   781,  ...,   231,   277],\n               [   72,   148,  ...,   781,  1009]],\n\n              [[  369,   372,  ...,   786,   868],\n               [  874,   957,  ...,   158,   258],\n               ...,\n               [  660,   839,  ...,   592,   448],\n               [  276,   587,  ...,   880,   695]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::UINT16, layout=Layout::ROW_MAJOR)",
    (
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
    ): "ttnn.Tensor([[[[  684,   559,  ...,   777,   916],\n               [  115,   976,  ...,   459,   882],\n               ...,\n               [  649,   773,  ...,   778,   555],\n               [  955,   414,  ...,   389,   378]],\n\n              [[  856,   273,  ...,   632,     2],\n               [  785,   143,  ...,   358,   404],\n               ...,\n               [  738,   150,  ...,   423,   609],\n               [  105,   687,  ...,   580,   862]],\n\n              ...,\n\n              [[  409,   607,  ...,   290,   816],\n               [  375,   306,  ...,   954,   218],\n               ...,\n               [  204,   718,  ...,   130,   890],\n               [  653,   250,  ...,   282,   825]],\n\n              [[  707,   273,  ...,   437,   848],\n               [  591,    14,  ...,   882,   546],\n               ...,\n               [  670,   571,  ...,   178,    24],\n               [    0,  1017,  ...,   664,   815]]],\n\n             [[[   27,     5,  ...,   401,   490],\n               [  136,   533,  ...,   688,   427],\n               ...,\n               [  827,  1018,  ...,   595,   431],\n               [  649,   238,  ...,   872,   741]],\n\n              [[  143,   142,  ...,   440,   812],\n               [  872,    76,  ...,   305,   892],\n               ...,\n               [  193,    83,  ...,   940,   404],\n               [  987,    69,  ...,   368,   413]],\n\n              ...,\n\n              [[  839,   704,  ...,   218,   229],\n               [  363,   605,  ...,   857,   928],\n               ...,\n               [  708,   781,  ...,   231,   277],\n               [   72,   148,  ...,   781,  1009]],\n\n              [[  369,   372,  ...,   786,   868],\n               [  874,   957,  ...,   158,   258],\n               ...,\n               [  660,   839,  ...,   592,   448],\n               [  276,   587,  ...,   880,   695]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::UINT32, layout=Layout::ROW_MAJOR)",
    (
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
    ): "ttnn.Tensor([[[[ 0.49626,  0.76822,  ...,  0.30510,  0.93200],\n               [ 0.17591,  0.26983,  ...,  0.20382,  0.65105],\n               ...,\n               [ 0.76926,  0.42571,  ...,  0.84923,  0.56027],\n               [ 0.44989,  0.81796,  ...,  0.82632,  0.29092]],\n\n              [[ 0.23870,  0.35561,  ...,  0.60709,  0.26819],\n               [ 0.30522,  0.16529,  ...,  0.58980,  0.36324],\n               ...,\n               [ 0.23448,  0.04438,  ...,  0.79019,  0.79197],\n               [ 0.40821,  0.77287,  ...,  0.61930,  0.06359]],\n\n              ...,\n\n              [[ 0.83083,  0.25181,  ...,  0.57106,  0.58434],\n               [ 0.36629,  0.82161,  ...,  0.59307,  0.03059],\n               ...,\n               [ 0.19764,  0.29350,  ...,  0.57648,  0.84179],\n               [ 0.63157,  0.61360,  ...,  0.61183,  0.73247]],\n\n              [[ 0.14732,  0.71010,  ...,  0.23446,  0.66704],\n               [ 0.80021,  0.18268,  ...,  0.80993,  0.10013],\n               ...,\n               [ 0.34751,  0.79996,  ...,  0.52534,  0.68817],\n               [ 0.58313,  0.48791,  ...,  0.25724,  0.24742]]],\n\n             [[[ 0.66742,  0.24011,  ...,  0.76113,  0.69809],\n               [ 0.64527,  0.37637,  ...,  0.88212,  0.59121],\n               ...,\n               [ 0.46611,  0.94733,  ...,  0.03122,  0.86672],\n               [ 0.19755,  0.84151,  ...,  0.17895,  0.65135]],\n\n              [[ 0.84791,  0.20442,  ...,  0.11282,  0.25896],\n               [ 0.79491,  0.29383,  ...,  0.44655,  0.89416],\n               ...,\n               [ 0.15174,  0.32483,  ...,  0.57135,  0.12307],\n               [ 0.12457,  0.01929,  ...,  0.79574,  0.12551]],\n\n              ...,\n\n              [[ 0.30748,  0.69975,  ...,  0.72877,  0.30830],\n               [ 0.16573,  0.45456,  ...,  0.94799,  0.36468],\n               ...,\n               [ 0.94468,  0.93938,  ...,  0.91499,  0.09071],\n               [ 0.57001,  0.48939,  ...,  0.71654,  0.78021]],\n\n              [[ 0.04604,  0.35653,  ...,  0.90001,  0.45373],\n               [ 0.09087,  0.64209,  ...,  0.97529,  0.16585],\n               ...,\n               [ 0.29423,  0.02880,  ...,  0.09598,  0.24148],\n               [ 0.29158,  0.08274,  ...,  0.43615,  0.71519]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::FLOAT32, layout=Layout::ROW_MAJOR)",
    (
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    ): "ttnn.Tensor([[[[ 0.67188,  0.18359,  ...,  0.03516,  0.57812],\n               [ 0.44922,  0.81250,  ...,  0.79297,  0.44531],\n               ...,\n               [ 0.53516,  0.01953,  ...,  0.03906,  0.16797],\n               [ 0.73047,  0.61719,  ...,  0.51953,  0.47656]],\n\n              [[ 0.34375,  0.06641,  ...,  0.46875,  0.00781],\n               [ 0.06641,  0.55859,  ...,  0.39844,  0.57812],\n               ...,\n               [ 0.88281,  0.58594,  ...,  0.65234,  0.37891],\n               [ 0.41016,  0.68359,  ...,  0.26562,  0.36719]],\n\n              ...,\n\n              [[ 0.59766,  0.37109,  ...,  0.13281,  0.18750],\n               [ 0.46484,  0.19531,  ...,  0.72656,  0.85156],\n               ...,\n               [ 0.79688,  0.80469,  ...,  0.50781,  0.47656],\n               [ 0.55078,  0.97656,  ...,  0.10156,  0.22266]],\n\n              [[ 0.76172,  0.06641,  ...,  0.70703,  0.31250],\n               [ 0.30859,  0.05469,  ...,  0.44531,  0.13281],\n               ...,\n               [ 0.61719,  0.23047,  ...,  0.69531,  0.09375],\n               [ 0.00000,  0.97266,  ...,  0.59375,  0.18359]]],\n\n             [[[ 0.10547,  0.01953,  ...,  0.56641,  0.91406],\n               [ 0.53125,  0.08203,  ...,  0.68750,  0.66797],\n               ...,\n               [ 0.23047,  0.97656,  ...,  0.32422,  0.68359],\n               [ 0.53516,  0.92969,  ...,  0.40625,  0.89453]],\n\n              [[ 0.55859,  0.55469,  ...,  0.71875,  0.17188],\n               [ 0.40625,  0.29688,  ...,  0.19141,  0.48438],\n               ...,\n               [ 0.75391,  0.32422,  ...,  0.67188,  0.57812],\n               [ 0.85547,  0.26953,  ...,  0.43750,  0.61328]],\n\n              ...,\n\n              [[ 0.27734,  0.75000,  ...,  0.85156,  0.89453],\n               [ 0.41797,  0.36328,  ...,  0.34766,  0.62500],\n               ...,\n               [ 0.76562,  0.05078,  ...,  0.90234,  0.08203],\n               [ 0.28125,  0.57812,  ...,  0.05078,  0.94141]],\n\n              [[ 0.44141,  0.45312,  ...,  0.07031,  0.39062],\n               [ 0.41406,  0.73828,  ...,  0.61719,  0.00781],\n               ...,\n               [ 0.57812,  0.27734,  ...,  0.31250,  0.75000],\n               [ 0.07812,  0.29297,  ...,  0.43750,  0.71484]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::BFLOAT16, layout=Layout::ROW_MAJOR)",
    (
        ttnn.uint16,
        ttnn.TILE_LAYOUT,
    ): "ttnn.Tensor([[[[  684,   559,  ...,   777,   916],\n               [  115,   976,  ...,   459,   882],\n               ...,\n               [  649,   773,  ...,   778,   555],\n               [  955,   414,  ...,   389,   378]],\n\n              [[  856,   273,  ...,   632,     2],\n               [  785,   143,  ...,   358,   404],\n               ...,\n               [  738,   150,  ...,   423,   609],\n               [  105,   687,  ...,   580,   862]],\n\n              ...,\n\n              [[  409,   607,  ...,   290,   816],\n               [  375,   306,  ...,   954,   218],\n               ...,\n               [  204,   718,  ...,   130,   890],\n               [  653,   250,  ...,   282,   825]],\n\n              [[  707,   273,  ...,   437,   848],\n               [  591,    14,  ...,   882,   546],\n               ...,\n               [  670,   571,  ...,   178,    24],\n               [    0,  1017,  ...,   664,   815]]],\n\n             [[[   27,     5,  ...,   401,   490],\n               [  136,   533,  ...,   688,   427],\n               ...,\n               [  827,  1018,  ...,   595,   431],\n               [  649,   238,  ...,   872,   741]],\n\n              [[  143,   142,  ...,   440,   812],\n               [  872,    76,  ...,   305,   892],\n               ...,\n               [  193,    83,  ...,   940,   404],\n               [  987,    69,  ...,   368,   413]],\n\n              ...,\n\n              [[  839,   704,  ...,   218,   229],\n               [  363,   605,  ...,   857,   928],\n               ...,\n               [  708,   781,  ...,   231,   277],\n               [   72,   148,  ...,   781,  1009]],\n\n              [[  369,   372,  ...,   786,   868],\n               [  874,   957,  ...,   158,   258],\n               ...,\n               [  660,   839,  ...,   592,   448],\n               [  276,   587,  ...,   880,   695]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::UINT16, layout=Layout::TILE)",
    (
        ttnn.uint32,
        ttnn.TILE_LAYOUT,
    ): "ttnn.Tensor([[[[  684,   559,  ...,   777,   916],\n               [  115,   976,  ...,   459,   882],\n               ...,\n               [  649,   773,  ...,   778,   555],\n               [  955,   414,  ...,   389,   378]],\n\n              [[  856,   273,  ...,   632,     2],\n               [  785,   143,  ...,   358,   404],\n               ...,\n               [  738,   150,  ...,   423,   609],\n               [  105,   687,  ...,   580,   862]],\n\n              ...,\n\n              [[  409,   607,  ...,   290,   816],\n               [  375,   306,  ...,   954,   218],\n               ...,\n               [  204,   718,  ...,   130,   890],\n               [  653,   250,  ...,   282,   825]],\n\n              [[  707,   273,  ...,   437,   848],\n               [  591,    14,  ...,   882,   546],\n               ...,\n               [  670,   571,  ...,   178,    24],\n               [    0,  1017,  ...,   664,   815]]],\n\n             [[[   27,     5,  ...,   401,   490],\n               [  136,   533,  ...,   688,   427],\n               ...,\n               [  827,  1018,  ...,   595,   431],\n               [  649,   238,  ...,   872,   741]],\n\n              [[  143,   142,  ...,   440,   812],\n               [  872,    76,  ...,   305,   892],\n               ...,\n               [  193,    83,  ...,   940,   404],\n               [  987,    69,  ...,   368,   413]],\n\n              ...,\n\n              [[  839,   704,  ...,   218,   229],\n               [  363,   605,  ...,   857,   928],\n               ...,\n               [  708,   781,  ...,   231,   277],\n               [   72,   148,  ...,   781,  1009]],\n\n              [[  369,   372,  ...,   786,   868],\n               [  874,   957,  ...,   158,   258],\n               ...,\n               [  660,   839,  ...,   592,   448],\n               [  276,   587,  ...,   880,   695]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::UINT32, layout=Layout::TILE)",
    (
        ttnn.float32,
        ttnn.TILE_LAYOUT,
    ): "ttnn.Tensor([[[[ 0.49626,  0.76822,  ...,  0.30510,  0.93200],\n               [ 0.17591,  0.26983,  ...,  0.20382,  0.65105],\n               ...,\n               [ 0.76926,  0.42571,  ...,  0.84923,  0.56027],\n               [ 0.44989,  0.81796,  ...,  0.82632,  0.29092]],\n\n              [[ 0.23870,  0.35561,  ...,  0.60709,  0.26819],\n               [ 0.30522,  0.16529,  ...,  0.58980,  0.36324],\n               ...,\n               [ 0.23448,  0.04438,  ...,  0.79019,  0.79197],\n               [ 0.40821,  0.77287,  ...,  0.61930,  0.06359]],\n\n              ...,\n\n              [[ 0.83083,  0.25181,  ...,  0.57106,  0.58434],\n               [ 0.36629,  0.82161,  ...,  0.59307,  0.03059],\n               ...,\n               [ 0.19764,  0.29350,  ...,  0.57648,  0.84179],\n               [ 0.63157,  0.61360,  ...,  0.61183,  0.73247]],\n\n              [[ 0.14732,  0.71010,  ...,  0.23446,  0.66704],\n               [ 0.80021,  0.18268,  ...,  0.80993,  0.10013],\n               ...,\n               [ 0.34751,  0.79996,  ...,  0.52534,  0.68817],\n               [ 0.58313,  0.48791,  ...,  0.25724,  0.24742]]],\n\n             [[[ 0.66742,  0.24011,  ...,  0.76113,  0.69809],\n               [ 0.64527,  0.37637,  ...,  0.88212,  0.59121],\n               ...,\n               [ 0.46611,  0.94733,  ...,  0.03122,  0.86672],\n               [ 0.19755,  0.84151,  ...,  0.17895,  0.65135]],\n\n              [[ 0.84791,  0.20442,  ...,  0.11282,  0.25896],\n               [ 0.79491,  0.29383,  ...,  0.44655,  0.89416],\n               ...,\n               [ 0.15174,  0.32483,  ...,  0.57135,  0.12307],\n               [ 0.12457,  0.01929,  ...,  0.79574,  0.12551]],\n\n              ...,\n\n              [[ 0.30748,  0.69975,  ...,  0.72877,  0.30830],\n               [ 0.16573,  0.45456,  ...,  0.94799,  0.36468],\n               ...,\n               [ 0.94468,  0.93938,  ...,  0.91499,  0.09071],\n               [ 0.57001,  0.48939,  ...,  0.71654,  0.78021]],\n\n              [[ 0.04604,  0.35653,  ...,  0.90001,  0.45373],\n               [ 0.09087,  0.64209,  ...,  0.97529,  0.16585],\n               ...,\n               [ 0.29423,  0.02880,  ...,  0.09598,  0.24148],\n               [ 0.29158,  0.08274,  ...,  0.43615,  0.71519]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::FLOAT32, layout=Layout::TILE)",
    (
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
    ): "ttnn.Tensor([[[[ 0.67188,  0.18359,  ...,  0.03516,  0.57812],\n               [ 0.44922,  0.81250,  ...,  0.79297,  0.44531],\n               ...,\n               [ 0.53516,  0.01953,  ...,  0.03906,  0.16797],\n               [ 0.73047,  0.61719,  ...,  0.51953,  0.47656]],\n\n              [[ 0.34375,  0.06641,  ...,  0.46875,  0.00781],\n               [ 0.06641,  0.55859,  ...,  0.39844,  0.57812],\n               ...,\n               [ 0.88281,  0.58594,  ...,  0.65234,  0.37891],\n               [ 0.41016,  0.68359,  ...,  0.26562,  0.36719]],\n\n              ...,\n\n              [[ 0.59766,  0.37109,  ...,  0.13281,  0.18750],\n               [ 0.46484,  0.19531,  ...,  0.72656,  0.85156],\n               ...,\n               [ 0.79688,  0.80469,  ...,  0.50781,  0.47656],\n               [ 0.55078,  0.97656,  ...,  0.10156,  0.22266]],\n\n              [[ 0.76172,  0.06641,  ...,  0.70703,  0.31250],\n               [ 0.30859,  0.05469,  ...,  0.44531,  0.13281],\n               ...,\n               [ 0.61719,  0.23047,  ...,  0.69531,  0.09375],\n               [ 0.00000,  0.97266,  ...,  0.59375,  0.18359]]],\n\n             [[[ 0.10547,  0.01953,  ...,  0.56641,  0.91406],\n               [ 0.53125,  0.08203,  ...,  0.68750,  0.66797],\n               ...,\n               [ 0.23047,  0.97656,  ...,  0.32422,  0.68359],\n               [ 0.53516,  0.92969,  ...,  0.40625,  0.89453]],\n\n              [[ 0.55859,  0.55469,  ...,  0.71875,  0.17188],\n               [ 0.40625,  0.29688,  ...,  0.19141,  0.48438],\n               ...,\n               [ 0.75391,  0.32422,  ...,  0.67188,  0.57812],\n               [ 0.85547,  0.26953,  ...,  0.43750,  0.61328]],\n\n              ...,\n\n              [[ 0.27734,  0.75000,  ...,  0.85156,  0.89453],\n               [ 0.41797,  0.36328,  ...,  0.34766,  0.62500],\n               ...,\n               [ 0.76562,  0.05078,  ...,  0.90234,  0.08203],\n               [ 0.28125,  0.57812,  ...,  0.05078,  0.94141]],\n\n              [[ 0.44141,  0.45312,  ...,  0.07031,  0.39062],\n               [ 0.41406,  0.73828,  ...,  0.61719,  0.00781],\n               ...,\n               [ 0.57812,  0.27734,  ...,  0.31250,  0.75000],\n               [ 0.07812,  0.29297,  ...,  0.43750,  0.71484]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::BFLOAT16, layout=Layout::TILE)",
    (
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
    ): "ttnn.Tensor([[[[ 0.50000,  0.76562,  ...,  0.30469,  0.92969],\n               [ 0.17969,  0.27344,  ...,  0.20312,  0.64844],\n               ...,\n               [ 0.76562,  0.42188,  ...,  0.85156,  0.56250],\n               [ 0.45312,  0.82031,  ...,  0.82812,  0.28906]],\n\n              [[ 0.24219,  0.35938,  ...,  0.60938,  0.26562],\n               [ 0.30469,  0.16406,  ...,  0.58594,  0.35938],\n               ...,\n               [ 0.23438,  0.04688,  ...,  0.78906,  0.78906],\n               [ 0.40625,  0.77344,  ...,  0.61719,  0.06250]],\n\n              ...,\n\n              [[ 0.82812,  0.25000,  ...,  0.57031,  0.58594],\n               [ 0.36719,  0.82031,  ...,  0.59375,  0.03125],\n               ...,\n               [ 0.19531,  0.29688,  ...,  0.57812,  0.84375],\n               [ 0.63281,  0.61719,  ...,  0.60938,  0.73438]],\n\n              [[ 0.14844,  0.71094,  ...,  0.23438,  0.66406],\n               [ 0.79688,  0.17969,  ...,  0.81250,  0.10156],\n               ...,\n               [ 0.34375,  0.79688,  ...,  0.52344,  0.68750],\n               [ 0.58594,  0.48438,  ...,  0.25781,  0.25000]]],\n\n             [[[ 0.66406,  0.24219,  ...,  0.75781,  0.69531],\n               [ 0.64844,  0.37500,  ...,  0.88281,  0.59375],\n               ...,\n               [ 0.46875,  0.94531,  ...,  0.03125,  0.86719],\n               [ 0.19531,  0.84375,  ...,  0.17969,  0.64844]],\n\n              [[ 0.85156,  0.20312,  ...,  0.10938,  0.25781],\n               [ 0.79688,  0.29688,  ...,  0.44531,  0.89062],\n               ...,\n               [ 0.14844,  0.32812,  ...,  0.57031,  0.12500],\n               [ 0.12500,  0.01562,  ...,  0.79688,  0.12500]],\n\n              ...,\n\n              [[ 0.30469,  0.70312,  ...,  0.72656,  0.30469],\n               [ 0.16406,  0.45312,  ...,  0.94531,  0.36719],\n               ...,\n               [ 0.94531,  0.93750,  ...,  0.91406,  0.09375],\n               [ 0.57031,  0.49219,  ...,  0.71875,  0.78125]],\n\n              [[ 0.04688,  0.35938,  ...,  0.89844,  0.45312],\n               [ 0.09375,  0.64062,  ...,  0.97656,  0.16406],\n               ...,\n               [ 0.29688,  0.03125,  ...,  0.09375,  0.24219],\n               [ 0.28906,  0.08594,  ...,  0.43750,  0.71875]]]], shape=Shape([2, 16, 64, 32]), dtype=DataType::BFLOAT8_B, layout=Layout::TILE)",
}


@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.uint16,
        ttnn.uint32,
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("profile", ["empty", "short"])
@pytest.mark.parametrize("deallocate", [False, True])
def test_print(device, dtype, layout, profile, deallocate):
    if layout == ttnn.ROW_MAJOR_LAYOUT and dtype == ttnn.bfloat8_b:
        pytest.skip("This combination is not valid")

    torch.manual_seed(0)

    ttnn.set_printoptions(profile=profile)

    torch_dtype = tt_dtype_to_torch_dtype[dtype]
    shape = (2, 16, 64, 32)

    if torch_dtype in {torch.int16, torch.int32}:
        torch_tensor = torch.randint(0, 1024, shape, dtype=torch_dtype)
    else:
        torch_tensor = torch.rand(shape, dtype=torch_dtype)

    tensor = ttnn.from_torch(torch_tensor, layout=layout, dtype=dtype, device=device)
    if deallocate:
        ttnn.deallocate(tensor)

    tensor_as_string = str(tensor)

    if deallocate:
        assert (
            tensor_as_string
            == f"ttnn.Tensor(<buffer is not allocated>, shape=Shape({list(shape)}), dtype=DataType::{dtype.name}, layout=Layout::{layout.name})"
        )
    elif profile == "empty":
        assert (
            tensor_as_string
            == f"ttnn.Tensor(..., shape=Shape({list(shape)}), dtype=DataType::{dtype.name}, layout=Layout::{layout.name})"
        )
    else:
        # To generate golden output, use the following line
        # print("\\n".join(str(tensor).split("\n")))

        assert tensor_as_string == GOLDEN_TENSOR_STRINGS[(dtype, layout)]


def test_print_0d(device):
    torch_tensor = torch.ones((), dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    assert str(tensor) == "ttnn.Tensor( 1.00000, shape=Shape([]), dtype=DataType::BFLOAT16, layout=Layout::TILE)"


def test_print_short_profile_limit(device):
    ttnn.set_printoptions(profile="short")  # This is the default profile
    torch_tensor = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    tensor_as_string = str(tensor)

    # Check that ellipsis is NOT used for dimensions of size 4 with short profile
    assert "..." not in tensor_as_string

    # Check the full string representation
    expected_string = (
        "ttnn.Tensor([[ 0.00000,  1.00000,  2.00000,  3.00000],\n"
        "             [ 4.00000,  5.00000,  6.00000,  7.00000],\n"
        "             [ 8.00000,  9.00000, 10.00000, 11.00000],\n"
        "             [12.00000, 13.00000, 14.00000, 15.00000]], shape=Shape([4, 4]), dtype=DataType::BFLOAT16, layout=Layout::TILE)"
    )
    assert tensor_as_string == expected_string
