# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc, tt_dtype_to_torch_dtype

bfloat4_pcc = 0.960
torch.manual_seed(0)


def is_ttnn_float_type(tt_dtype) -> bool:
    match tt_dtype:
        case ttnn.bfloat16 | ttnn.float32 | ttnn.bfloat8_b | ttnn.bfloat4_b:
            return True
        case _:
            return False


ALL_TYPES = [dtype for dtype, _ in ttnn.DataType.__entries.values() if dtype != ttnn.DataType.INVALID]
FLOAT_TYPES = [dtype for dtype, _ in ttnn.DataType.__entries.values() if is_ttnn_float_type(dtype)]


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
@pytest.mark.parametrize("to_dtype", ALL_TYPES)
@pytest.mark.parametrize("from_dtype", ALL_TYPES)
def test_to_dtype(height, width, from_dtype, to_dtype):
    torch_input_tensor = torch.randint(0, 10, (height, width), dtype=tt_dtype_to_torch_dtype[from_dtype])

    input_tensor = ttnn.from_torch(torch_input_tensor)
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_dtype(input_tensor, to_dtype)

    assert output_tensor.dtype == to_dtype
    assert tuple(output_tensor.shape) == (height, width)
    if to_dtype == ttnn.bfloat8_b or to_dtype == ttnn.bfloat4_b:
        assert output_tensor.layout == ttnn.TILE_LAYOUT
    else:
        assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch_input_tensor.dtype)
    assert_with_pcc(torch_input_tensor, output_tensor, bfloat4_pcc if to_dtype == ttnn.bfloat4_b else 0.9999)


@pytest.mark.parametrize("shape", [(4, 4), (32, 32)])
@pytest.mark.parametrize(
    "ttnn_dtype",
    [
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.bfloat16,
        ttnn.uint8,
        ttnn.int32,
        ttnn.uint32,
    ],
)
@pytest.mark.parametrize(
    "torch_dtype",
    [
        torch.float64,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.int32,
        torch.uint8,
    ],
)
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("convert_with_device", [True, False])
def test_dtype_conversion_on_device(device, shape, ttnn_dtype, torch_dtype, ttnn_layout, convert_with_device):
    ttnn_dtype_requires_tile = ttnn_dtype in [ttnn.bfloat8_b, ttnn.bfloat4_b]
    ttnn_dtype_has_random = ttnn_dtype not in [ttnn.uint8, ttnn.int32]
    ttnn_is_float = ttnn_dtype in [ttnn.float32, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b]
    torch_is_float = torch_dtype in [torch.float16, torch.float32, torch.float64]

    conversion_pcc = None

    if (ttnn_dtype == ttnn.float32 and torch_dtype == torch.float32) or (
        ttnn_dtype == ttnn.int32 and torch_dtype == torch.int32
    ):
        conversion_pcc = 1

    elif ttnn_is_float != torch_is_float:
        conversion_pcc = 0.98

    elif torch_dtype == torch.bfloat16:
        if ttnn_dtype == ttnn.bfloat16 or ttnn_dtype == ttnn.bfloat8_b:
            conversion_pcc = 0.9999
        elif ttnn_dtype == ttnn.bfloat4_b:
            conversion_pcc = 0.989
        else:
            conversion_pcc = 0.999

    else:
        conversion_pcc = 0.999

    # print("")

    if ttnn_dtype_has_random:
        for store_input_on_device in [True, False]:
            ttnn_input_tensor = ttnn.rand(
                shape,
                dtype=ttnn_dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT if ttnn_dtype_requires_tile else ttnn_layout,
            )

            if not store_input_on_device:
                ttnn_input_tensor = ttnn.from_device(ttnn_input_tensor)

            torch_result_tensor = ttnn.to_torch(
                ttnn_input_tensor, dtype=torch_dtype, device=device if convert_with_device else None
            )
            assert (
                torch_result_tensor.dtype == torch_dtype
            ), f"Expected result {torch_dtype}, got result tensor {torch_result_tensor.dtype} when converting TTNN tensor {ttnn_input_tensor.dtype}"

        assert_with_pcc(
            expected_pytorch_result=torch_result_tensor,
            actual_pytorch_result=ttnn_input_tensor.cpu().to_torch(),
            pcc=conversion_pcc,
        )

    if torch_is_float:
        torch_input_tensor = torch.rand(shape, dtype=torch_dtype) * 10

    else:
        torch_input_tensor = torch.randint(0, 100, shape, dtype=torch_dtype)

    ttnn_result_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device if convert_with_device else None,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT if ttnn_dtype_requires_tile else ttnn_layout,
    )

    assert (
        ttnn_result_tensor.dtype == ttnn_dtype
    ), f"Expected result {ttnn_dtype}, got result tensor {ttnn_result_tensor.dtype} when converting torch tensor {torch_input_tensor.dtype}"

    assert_with_pcc(
        expected_pytorch_result=torch_input_tensor,
        actual_pytorch_result=ttnn_result_tensor.cpu().to_torch(),
        pcc=conversion_pcc,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (32, 32),
    ],
)
@pytest.mark.parametrize("ttnn_dtype_from", ALL_TYPES)
@pytest.mark.parametrize("ttnn_dtype_to", ALL_TYPES)
def test_typecast_accuracy(shape, device, ttnn_dtype_from, ttnn_dtype_to):
    if ttnn_dtype_from == ttnn.uint8 or ttnn_dtype_to == ttnn.uint8:
        pytest.skip("uint8 is not supported by the typecast directly")

    float_types = [ttnn.float32, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b]
    ttnn_from_is_float = ttnn_dtype_from in float_types
    ttnn_to_is_float = ttnn_dtype_to in float_types

    conversion_pcc = None

    if ttnn_dtype_from == ttnn_dtype_to:
        conversion_pcc = 1

    elif ttnn_dtype_to == ttnn.bfloat4_b:
        conversion_pcc = 0.960

    elif ttnn_dtype_to == ttnn.bfloat8_b:
        conversion_pcc = 0.999

    else:
        conversion_pcc = 0.9999

    if ttnn_from_is_float:
        input_tensor = ttnn.rand(shape, device=device, dtype=ttnn_dtype_from) * 100

    else:
        input_tensor = ttnn.rand(shape, device=device, dtype=ttnn_dtype_from, low=0, high=100)

    input_torch_tensor = torch.Tensor(input_tensor.to_list())
    output_tensor = ttnn.typecast(input_tensor, dtype=ttnn_dtype_to)
    output_torch_tensor = torch.Tensor(output_tensor.to_list())

    pcc_passed, pcc_message = comp_pcc(input_torch_tensor, output_torch_tensor, conversion_pcc)
    assert pcc_passed, f"""
{pcc_message}
input_tensor:
{input_tensor}
input_torch_tensor:
{input_torch_tensor}
output_tensor:
{output_tensor}
output_torch_tensor:
{output_torch_tensor}
    """


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
@pytest.mark.parametrize("to_dtype", FLOAT_TYPES)
@pytest.mark.parametrize("from_dtype", FLOAT_TYPES)
def test_to_float_dtype(height, width, from_dtype, to_dtype):
    torch_input_tensor = torch.rand((height, width), dtype=tt_dtype_to_torch_dtype[from_dtype])

    input_tensor = ttnn.from_torch(torch_input_tensor)
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_dtype(input_tensor, to_dtype)

    assert output_tensor.dtype == to_dtype
    assert tuple(output_tensor.shape) == (height, width)
    if to_dtype == ttnn.bfloat8_b or to_dtype == ttnn.bfloat4_b:
        assert output_tensor.layout == ttnn.TILE_LAYOUT
    else:
        assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch_input_tensor.dtype)
    assert_with_pcc(torch_input_tensor, output_tensor, bfloat4_pcc if to_dtype == ttnn.bfloat4_b else 0.9999)


@pytest.mark.parametrize("height", [36])
@pytest.mark.parametrize("width", [36])
@pytest.mark.parametrize("to_dtype", ALL_TYPES)
@pytest.mark.parametrize("from_dtype", ALL_TYPES)
def test_to_dtype_unaligned_shape(height, width, from_dtype, to_dtype):
    if (
        from_dtype == ttnn.bfloat4_b
        or from_dtype == ttnn.bfloat8_b
        or to_dtype == ttnn.bfloat4_b
        or to_dtype == ttnn.bfloat8_b
    ):
        pytest.skip("bfloat4_b and bfloat8_b require align shape divisible by tile")

    torch_input_tensor = torch.randint(0, 10, (height, width), dtype=tt_dtype_to_torch_dtype[from_dtype])

    input_tensor = ttnn.from_torch(torch_input_tensor)
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_dtype(input_tensor, to_dtype)

    assert output_tensor.dtype == to_dtype
    assert tuple(output_tensor.shape) == (height, width)
    assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch_input_tensor.dtype)
    assert_with_pcc(torch_input_tensor, output_tensor, bfloat4_pcc if to_dtype == ttnn.bfloat4_b else 0.9999)


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [32])
@pytest.mark.parametrize("to_dtype", ALL_TYPES)
@pytest.mark.parametrize("from_dtype", ALL_TYPES)
def test_to_dtype_with_tile_layout(height, width, from_dtype, to_dtype):
    torch_input_tensor = torch.randint(0, 10, (height, width), dtype=tt_dtype_to_torch_dtype[from_dtype])

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT)
    assert input_tensor.layout == ttnn.TILE_LAYOUT

    output_tensor = ttnn.to_dtype(input_tensor, to_dtype)

    assert output_tensor.dtype == to_dtype
    assert tuple(output_tensor.shape) == (height, width)
    assert output_tensor.layout == ttnn.TILE_LAYOUT

    output_tensor = ttnn.to_torch(output_tensor, dtype=torch_input_tensor.dtype)
    assert_with_pcc(torch_input_tensor, output_tensor, bfloat4_pcc if to_dtype == ttnn.bfloat4_b else 0.9999)
