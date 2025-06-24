
#include "emitc.hpp"

template <typename... T>
std::vector<ttnn::Tensor> util_create_vec(T&&... t) {
    return std::vector<ttnn::Tensor>{std::forward<T>(t)...};
}

::std::vector<::ttnn::Tensor> forward(::std::vector<::ttnn::Tensor> v1) {
    ::ttnn::Tensor v2 = v1[0];
    ::ttnn::Tensor v3 = v1[1];
    ::ttnn::Tensor v4 = v1[2];
    ::ttnn::Tensor v5 = v1[3];
    ::ttnn::Tensor v6 = v1[4];
    ::ttnn::Tensor v7 = v1[5];
    ::ttnn::Tensor v8 = v1[6];
    ::ttnn::Tensor v9 = ttnn::matmul(
        v2,
        v3,
        false,
        false,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ttnn::deallocate(v3, false);
    ttnn::deallocate(v2, false);
    ::ttnn::Tensor v10 = ttnn::reshape(
        v4,
        ::std::vector<int32_t>{1, 512},
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ttnn::deallocate(v4, false);
    ::ttnn::Tensor v11 = ttnn::add(
        v9,
        v10,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ttnn::deallocate(v10, false);
    ttnn::deallocate(v9, false);
    ::ttnn::Tensor v12 = ttnn::relu(
        v11, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ttnn::deallocate(v11, false);
    ::ttnn::Tensor v13 = ttnn::matmul(
        v12,
        v5,
        false,
        false,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ttnn::deallocate(v12, false);
    ttnn::deallocate(v5, false);
    ::ttnn::Tensor v14 = ttnn::reshape(
        v6,
        ::std::vector<int32_t>{1, 512},
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ttnn::deallocate(v6, false);
    ::ttnn::Tensor v15 = ttnn::add(
        v13,
        v14,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ttnn::deallocate(v14, false);
    ttnn::deallocate(v13, false);
    ::ttnn::Tensor v16 = ttnn::relu(
        v15, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ttnn::deallocate(v15, false);
    ::ttnn::Tensor v17 = ttnn::matmul(
        v16,
        v7,
        false,
        false,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ttnn::deallocate(v16, false);
    ttnn::deallocate(v7, false);
    ::ttnn::Tensor v18 = ttnn::reshape(
        v8,
        ::std::vector<int32_t>{1, 10},
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ttnn::deallocate(v8, false);
    ::ttnn::Tensor v19 = ttnn::add(
        v17,
        v18,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ttnn::deallocate(v18, false);
    ttnn::deallocate(v17, false);
    ::std::vector<::ttnn::Tensor> v20 = util_create_vec(v19);
    return v20;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward() {
    ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
    ::ttnn::Tensor v2 = ttnn::ones(
        ::ttnn::Shape({1, 784}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ::ttnn::Tensor v3 = ttnn::to_device(
        v2,
        v1,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ::ttnn::Tensor v4 = ttnn::ones(
        ::ttnn::Shape({784, 512}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ::ttnn::Tensor v5 = ttnn::to_device(
        v4,
        v1,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ::ttnn::Tensor v6 = ttnn::ones(
        ::ttnn::Shape({512}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ::ttnn::Tensor v7 = ttnn::to_device(
        v6,
        v1,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ::ttnn::Tensor v8 = ttnn::ones(
        ::ttnn::Shape({512, 512}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ::ttnn::Tensor v9 = ttnn::to_device(
        v8,
        v1,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ::ttnn::Tensor v10 = ttnn::ones(
        ::ttnn::Shape({512}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ::ttnn::Tensor v11 = ttnn::to_device(
        v10,
        v1,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ::ttnn::Tensor v12 = ttnn::ones(
        ::ttnn::Shape({512, 10}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ::ttnn::Tensor v13 = ttnn::to_device(
        v12,
        v1,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ::ttnn::Tensor v14 = ttnn::ones(
        ::ttnn::Shape({10}),
        ::ttnn::DataType::FLOAT32,
        ::ttnn::Layout::TILE,
        ::std::nullopt,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ::ttnn::Tensor v15 = ttnn::to_device(
        v14,
        v1,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
    ::std::vector<::ttnn::Tensor> v16 = util_create_vec(v3, v5, v7, v9, v11, v13, v15);
    return v16;
}

void time_run(
    const std::function<std::vector<ttnn::Tensor>(std::vector<ttnn::Tensor>)>& fn,
    const std::string& run_name,
    std::vector<ttnn::Tensor> inputs) {
    // Measure execution time
    //
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<ttnn::Tensor> res = fn(inputs);
    for (auto& t : res) {
        ttnn::Tensor host_tensor = ttnn::from_device(t, /*blocking=*/true);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Print results
    //
    std::cout << duration.count() << " seconds for run: " << run_name << std::endl;
}

TEST(EmitC, ProgramCache) {
    ttnn::MeshDevice* device = ttnn::DeviceGetter::getInstance();

    ::std::vector<::ttnn::Tensor> v1 = create_inputs_for_forward();

    time_run(forward, "run 0", v1);
    time_run(forward, "run 1", v1);
    time_run(forward, "run 2", v1);
    time_run(forward, "run 3", v1);
    time_run(forward, "run 4", v1);

    std::cout << "Enabling program cache" << std::endl;

    device->enable_program_cache();

    time_run(forward, "run 5", v1);
    time_run(forward, "run 6", v1);
    time_run(forward, "run 7", v1);
    time_run(forward, "run 8", v1);
    time_run(forward, "run 9", v1);
}
