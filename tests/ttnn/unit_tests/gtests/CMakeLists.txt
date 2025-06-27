set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/ttnn)

function(setup_ttnn_test_target target_name)
    target_link_libraries(
        ${target_name}
        PUBLIC
            test_common_libs
            TTNN::CPP
    )
    target_include_directories(
        ${target_name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/tests
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
endfunction()

# unit_tests_ttnn

add_library(unit_tests_ttnn_smoke OBJECT)
add_library(TTNN::Test::Smoke ALIAS unit_tests_ttnn_smoke)
TT_ENABLE_UNITY_BUILD(unit_tests_ttnn_smoke)
target_sources(
    unit_tests_ttnn_smoke
    PRIVATE
        test_matmul_benchmark.cpp
        test_multiprod_queue.cpp
        test_multi_cq_multi_dev.cpp
        test_reflect.cpp
        test_to_and_from_json.cpp
        test_sliding_window_infra.cpp
)
target_include_directories(unit_tests_ttnn_smoke PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(
    unit_tests_ttnn_smoke
    PRIVATE
        test_common_libs
        TTNN::CPP
)

add_library(unit_tests_ttnn_basic OBJECT)
add_library(TTNN::Test::Basic ALIAS unit_tests_ttnn_basic)
TT_ENABLE_UNITY_BUILD(unit_tests_ttnn_basic)
target_sources(
    unit_tests_ttnn_basic
    PRIVATE
        common_test_utils.cpp
        test_add.cpp
        test_async_runtime.cpp # TODO: Fix memory leak (LSan) then shift-left
        test_broadcast_to.cpp
        test_conv2d.cpp # TODO: Fix misaligned memory load (UBSan) then shift-left
        test_generic_op.cpp
        test_graph_add.cpp
        test_graph_basic.cpp
        test_graph_capture_arguments_morehdot.cpp
        test_graph_capture_arguments_transpose.cpp
        test_graph_query_op_constraints.cpp
        test_graph_query_op_runtime.cpp
        captures.cpp
        test_launch_operation.cpp # TODO: Fix data race (TSan) then shift-left
)
target_include_directories(unit_tests_ttnn_basic PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(
    unit_tests_ttnn_basic
    PRIVATE
        test_common_libs
        TTNN::CPP
        common
)

add_executable(unit_tests_ttnn)
target_link_libraries(
    unit_tests_ttnn
    PRIVATE
        TTNN::Test::Smoke
        TTNN::Test::Basic
)

# unit_tests_ttnn_ccl
