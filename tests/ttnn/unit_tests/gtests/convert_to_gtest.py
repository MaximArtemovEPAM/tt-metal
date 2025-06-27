import re


def parse_shape(spec):
    m = re.search(r"Shape\(\{([^\}]*)\}\)", spec)
    return "{" + m.group(1).strip() + "}" if m else "{}"


def parse_dtype(spec):
    m = re.search(r"dtype\s*=\s*DataType::(\w+)", spec)
    return f"::ttnn::DataType::{m.group(1)}" if m else "::ttnn::DataType::BFLOAT16"


def parse_tile(spec):
    m = re.search(r"tile_shape=\{([^\}]*)\}", spec)
    return "{" + m.group(1).strip() + "}" if m else "{32, 32}"


def parse_face(spec):
    m = re.search(r"face_shape=\{([^\}]*)\}", spec)
    return "{" + m.group(1).strip() + "}" if m else "{16, 16}"


def parse_num_faces(spec):
    m = re.search(r"num_faces=(\d+)", spec)
    return m.group(1) if m else "4"


def extract_paren_block(text, key):
    # Find the start of key=MemoryConfig(
    start = text.find(key + "=MemoryConfig(")
    if start == -1:
        return ""
    start = start + len(key) + 1  # position at 'M' in 'MemoryConfig'
    paren_count = 0
    end = start
    for i, c in enumerate(text[start:], start=start):
        if c == "(":
            paren_count += 1
        elif c == ")":
            paren_count -= 1
            if paren_count == 0:
                end = i + 1
                break
    return text[start:end].strip()


def extract_shard_spec_paren_block(text):
    # Find the start of ShardSpec(
    start = text.find("ShardSpec(")
    if start == -1:
        return ""
    # Find the opening parenthesis after 'ShardSpec'
    open_paren = text.find("(", start)
    if open_paren == -1:
        return ""
    start = open_paren + 1
    paren_count = 1
    for i in range(start, len(text)):
        if text[i] == "(":
            paren_count += 1
        elif text[i] == ")":
            paren_count -= 1
            if paren_count == 0:
                return text[start:i]
    return ""


def parse_mem_config(spec, key="memory_config"):
    # Try to extract the full MemoryConfig(...) block for MakeTensorSpec
    mem_config_block = extract_paren_block(spec, key)
    if mem_config_block:
        m_layout = re.search(r"memory_layout\s*=\s*TensorMemoryLayout::(\w+)", mem_config_block)
        layout = f"TensorMemoryLayout::{m_layout.group(1)}" if m_layout else "TensorMemoryLayout::INTERLEAVED"
        m_buf = re.search(r"buffer_type\s*=\s*BufferType::(\w+)", mem_config_block)
        buf_type = f"BufferType::{m_buf.group(1)}" if m_buf else "BufferType::DRAM"
        shard_spec_val = "std::nullopt"
        if "shard_spec=ShardSpec(" in mem_config_block:
            shard_spec_val = extract_shard_spec_paren_block(mem_config_block)
        # # [DEBUG]"[DEBUG] shard_spec_val:", shard_spec_val)
        if shard_spec_val != "std::nullopt" and "grid=" in shard_spec_val and "shape=" in shard_spec_val:
            grid = extract_brace_block(shard_spec_val, "grid") or "{}"
            shape = extract_brace_block(shard_spec_val, "shape") or "{}"
            orientation = (
                "ShardOrientation::ROW_MAJOR"
                if "orientation=ShardOrientation::ROW_MAJOR" in shard_spec_val
                else "ShardOrientation::ROW_MAJOR"
            )
            cpp = f"MakeMemoryConfig({layout}, {buf_type}, MakeShardSpec({grid}, {shape}, {orientation}))"
        else:
            cpp = f"MakeMemoryConfig({layout}, {buf_type}, std::nullopt)"
        # # [DEBUG]"[DEBUG] input_spec:", spec)
        # # [DEBUG]"[DEBUG] generated_cpp:", cpp)
        return cpp
    return f"MakeMemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt)"
    return cpp


def extract_brace_block(text, key):
    # Robustly extract the brace block after key= (tolerant to whitespace, field order, and trailing commas/parentheses)
    import re

    match = re.search(rf"{re.escape(key)}\s*=\s*\{{", text)
    if not match:
        return "{}"
    start = match.end() - 1  # position at the opening brace
    brace_count = 0
    block_start = None
    for i in range(start, len(text)):
        if text[i] == "{":
            if brace_count == 0:
                block_start = i
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0 and block_start is not None:
                result = text[block_start : i + 1]
                # [DEBUG]f"[DEBUG] extract_brace_block: found for key='{key}': {result}")
                return result
    # [DEBUG]f"[DEBUG] extract_brace_block: unmatched braces for key='{key}'")
    return "{}"


def parse_shard_spec(spec):
    grid = extract_brace_block(spec, "grid")
    shape = extract_brace_block(spec, "shape")
    return grid, shape


def parse_other_field(line):
    # Remove comment and keep only the value after '='
    parts = line.split("=")
    if len(parts) < 2:
        return ""
    value = "=".join(parts[1:]).strip().rstrip(",")
    # Remove any trailing or leading comment marks
    if value.startswith("*/"):
        value = value[2:].strip()
    return value


def parse_bias_spec(line):
    # Remove comment and keep only the value after '='
    parts = line.split("=")
    if len(parts) < 2:
        return "std::nullopt"
    value = "=".join(parts[1:]).strip().rstrip(",")
    if value == "std::nullopt":
        return "std::nullopt"
    # Otherwise, parse like input_spec/weight_spec
    shape = parse_shape(value)
    dtype = parse_dtype(value)
    tile = parse_tile(value)
    face = parse_face(value)
    num_faces = parse_num_faces(value)
    mem_config = parse_mem_config(value)
    return f"MakeTensorSpec({shape}, {dtype}, Layout::TILE, {tile}, {face}, {num_faces}, {mem_config})"


def parse_original_weights_shape(line):
    # print('[debug ENTER parse_original_weights_shape]', repr(line))
    line = line.strip()
    # Remove leading comment if present
    if line.startswith("/*"):
        line = line.split("*/", 1)[-1].strip()
    parts = line.split("=")
    if len(parts) < 2:
        # print('[debug RETURN empty - no =]')
        return "{}"
    value = "=".join(parts[1:]).strip().rstrip(",")
    # Remove trailing comment if present
    if "*/" in value:
        value = value.split("*/", 1)[0].strip()
    # print('[debug original_weights_shape value]', repr(value))
    if value.startswith("{") and value.endswith("}"):
        # print('[debug RETURN value]', repr(value))
        return value
    # print('[debug RETURN empty - not braces]', repr(value))
    return "{}"


def main(DEBUG_MODE=False):
    with open("params_gtest.inc" if not DEBUG_MODE else "debug.txt") as f:
        content = f.read()

    # Split into blocks using a brace counter for robustness
    def extract_blocks(text):
        blocks = []
        brace_level = 0
        current = []
        for line in text.splitlines():
            if "{" in line and brace_level == 0:
                current = []
            brace_level += line.count("{")
            if brace_level > 0:
                current.append(line)
            brace_level -= line.count("}")
            if brace_level == 0 and current:
                blocks.append("\n".join(current))
                current = []
        return blocks

    blocks = extract_blocks(content)
    output = []
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        try:
            input_spec = next(l for l in lines if "input_spec" in l)
            # Try to find original_weights_shape, fallback to None if missing
            original_weights_shape_line = next((l for l in lines if "original_weights_shape" in l), None)
            weight_spec = next(l for l in lines if "weight_spec" in l)
            in_channels = next(l for l in lines if "in_channels" in l)
            out_channels = next(l for l in lines if "out_channels" in l)
            batch_size = next(l for l in lines if "batch_size" in l)
            input_height = next(l for l in lines if "input_height" in l)
            input_width = next(l for l in lines if "input_width" in l)
            kernel_size = next(l for l in lines if "kernel_size" in l)
            stride = next(l for l in lines if "stride" in l)
            padding = next(l for l in lines if "padding" in l)
            dilation = next(l for l in lines if "dilation" in l)
            groups = next(l for l in lines if "groups" in l)
            bias = next(l for l in lines if "bias" in l)
            conv2d_config = next(l for l in lines if "conv2d_config" in l)
            output_mem_config = next(l for l in lines if "output_mem_config" in l)
        except StopIteration:
            continue

        input_shape = parse_shape(input_spec)
        input_dtype = parse_dtype(input_spec)
        input_tile = parse_tile(input_spec)
        input_face = parse_face(input_spec)
        input_num_faces = parse_num_faces(input_spec)
        input_mem_config = parse_mem_config(input_spec)

        # print('[debug original_weights_shape_line]', repr(original_weights_shape_line))
        if original_weights_shape_line:
            original_weights_shape_cpp = parse_original_weights_shape(original_weights_shape_line)
        else:
            original_weights_shape_cpp = "{}"
        # print('[debug assigned original_weights_shape_cpp]', repr(original_weights_shape_cpp))

        weight_shape = parse_shape(weight_spec)
        weight_dtype = parse_dtype(weight_spec)
        weight_tile = parse_tile(weight_spec)
        weight_face = parse_face(weight_spec)
        weight_num_faces = parse_num_faces(weight_spec)
        weight_mem_config = parse_mem_config(weight_spec)

        grid, shape = parse_shard_spec(output_mem_config)
        mem_config = parse_mem_config(output_mem_config)

        # Debug: print block order and memory config
        # You can comment out these prints after checking correctness
        # # [DEBUG]f"\n--- Block ---\noutput_mem_config: {output_mem_config}")

        memory_config_cpp = mem_config

        cpp_block = f"""QueryOpConstraintsParams{{
    /* input_spec */
    MakeTensorSpec(
        {input_shape},
        {input_dtype},
        Layout::TILE,
        {input_tile},
        {input_face},
        {input_num_faces},
        {input_mem_config}),
    /* original_weights_shape */
    {parse_other_field(original_weights_shape_line)},
    /* weight_spec */
    MakeTensorSpec(
        {weight_shape},
        {weight_dtype},
        Layout::TILE,
        {weight_tile},
        {weight_face},
        {weight_num_faces},
        {weight_mem_config}),
    /* in_channels */ {parse_other_field(in_channels)},
    /* out_channels */ {parse_other_field(out_channels)},
    /* batch_size */ {parse_other_field(batch_size)},
    /* input_height */ {parse_other_field(input_height)},
    /* input_width */ {parse_other_field(input_width)},
    /* kernel_size */ {parse_other_field(kernel_size)},
    /* stride */ {parse_other_field(stride)},
    /* padding */ {parse_other_field(padding)},
    /* dilation */ {parse_other_field(dilation)},
    /* groups */ {parse_other_field(groups)},
    /* bias_spec */ {parse_bias_spec(bias)},

    /* conv2d_config */
    [] {{
        ::ttnn::operations::conv::conv2d::Conv2dConfig cfg;
        cfg.dtype = ::ttnn::DataType::BFLOAT16;
        cfg.weights_dtype = ::ttnn::DataType::BFLOAT16;
        cfg.activation = "";
        cfg.deallocate_activation = false;
        cfg.reallocate_halo_output = true;
        cfg.act_block_h_override = 0;
        cfg.act_block_w_div = 1;
        cfg.reshard_if_not_optimal = false;
        cfg.override_sharding_config = false;
        cfg.shard_layout = std::nullopt;
        cfg.core_grid = std::nullopt;
        cfg.transpose_shards = true;
        cfg.output_layout = ::ttnn::Layout::TILE;
        cfg.preprocess_weights_on_device = false;
        cfg.enable_act_double_buffer = false;
        cfg.enable_weights_double_buffer = false;
        cfg.enable_split_reader = false;
        cfg.enable_subblock_padding = false;
        cfg.in_place = false;
        return cfg;
    }}(),
    /* memory_config */
    {memory_config_cpp}
}}"""
        output.append(cpp_block)

    with open("out.cpp", "w") as fout:
        fout.write("::testing::Values(\n")
        fout.write(",\n".join(output))
        fout.write("\n);\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()
    main(DEBUG_MODE=args.debug)
