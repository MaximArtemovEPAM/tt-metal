// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/fabric_host_interface.h>
#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <utility>
#include <variant>
#include <vector>
#include <random>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"
#include "tests/tt_metal/tt_fabric/common/utils.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/fabric.hpp>
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "umd/device/tt_core_coordinates.h"

namespace tt::tt_fabric {
namespace fabric_router_tests {

std::random_device rd;  // Non-deterministic seed source
std::mt19937 global_rng(rd());

struct WorkerMemMap {
    uint32_t packet_header_address;
    uint32_t source_l1_buffer_address;
    uint32_t packet_payload_size_bytes;
    uint32_t test_results_address;
    uint32_t target_address;
    uint32_t test_results_size_bytes;
};

// Utility function reused across tests to get address params
WorkerMemMap generate_worker_mem_map(tt_metal::IDevice* device, Topology topology) {
    constexpr uint32_t PACKET_HEADER_RESERVED_BYTES = 45056;
    constexpr uint32_t DATA_SPACE_RESERVED_BYTES = 851968;
    constexpr uint32_t TEST_RESULTS_SIZE_BYTES = 128;

    uint32_t base_addr = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
    uint32_t packet_header_address = base_addr;
    uint32_t source_l1_buffer_address = base_addr + PACKET_HEADER_RESERVED_BYTES;
    uint32_t test_results_address = source_l1_buffer_address + DATA_SPACE_RESERVED_BYTES;
    uint32_t target_address = source_l1_buffer_address;

    uint32_t packet_payload_size_bytes = (topology == Topology::Mesh) ? 2048 : 4096;

    return {
        packet_header_address,
        source_l1_buffer_address,
        packet_payload_size_bytes,
        test_results_address,
        target_address,
        TEST_RESULTS_SIZE_BYTES};
}

std::vector<uint32_t> get_random_numbers_from_range(uint32_t start, uint32_t end, uint32_t count) {
    std::vector<uint32_t> range(end - start + 1);

    // generate the range
    std::iota(range.begin(), range.end(), start);

    // shuffle the range
    std::shuffle(range.begin(), range.end(), global_rng);

    return std::vector<uint32_t>(range.begin(), range.begin() + count);
}

std::shared_ptr<tt_metal::Program> create_receiver_program(
    const std::vector<uint32_t>& compile_time_args,
    const std::vector<uint32_t>& runtime_args,
    const CoreCoord& logical_core) {
    auto recv_program = std::make_shared<tt_metal::Program>();
    auto recv_kernel = tt_metal::CreateKernel(
        *recv_program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
        {logical_core},
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_time_args});
    tt_metal::SetRuntimeArgs(*recv_program, recv_kernel, logical_core, runtime_args);
    return recv_program;
}

void RandomizedInterMeshUnicast(BaseFabricFixture* fixture) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    constexpr uint32_t num_packets = 100;

    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();
    const auto& edm_config = fabric_context.get_fabric_router_config();

    auto devices = fixture->get_devices();
    auto num_devices = devices.size();
    const auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();

    auto distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (*(distributed_context->rank()) == 0) {
        // The following code runs on the sender host
        // Synchronize seeds across hosts (sender and receiver must use the same seed for randomization)
        uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();
        distributed_context->send(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&time_seed), sizeof(time_seed)),
            tt::tt_metal::distributed::multihost::Rank{1},  // send to receiver host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange seed over tag 0
        );
        std::cout << "Sender using time seed: " << time_seed << std::endl;

        auto random_dev_list = get_random_numbers_from_range(0, devices.size() - 1, devices.size());
        auto src_physical_device_id = devices[random_dev_list[0]]->id();
        auto src_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(src_physical_device_id);
        auto mesh_shape = control_plane.get_physical_mesh_shape(src_fabric_node_id.mesh_id);
        auto* sender_device = DevicePool::instance().get_active_device(src_physical_device_id);

        const auto& worker_grid_size = sender_device->compute_with_storage_grid_size();
        auto sender_x = get_random_numbers_from_range(0, worker_grid_size.x - 2, worker_grid_size.x)[0];
        auto sender_y = get_random_numbers_from_range(0, worker_grid_size.y - 2, worker_grid_size.y)[0];
        CoreCoord sender_logical_core = {sender_x, sender_y};
        CoreCoord receiver_logical_core = {0, 0};
        distributed_context->recv(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&receiver_logical_core), sizeof(receiver_logical_core)),
            tt::tt_metal::distributed::multihost::Rank{1},  // receive from receiver host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange logical core over tag 0
        );
        std::cout << "Sender logical core: (" << sender_x << ", " << sender_x << ")" << std::endl;
        FabricNodeId dst_fabric_node_id(MeshId{0}, 0);
        // Read for the destination fabric node id generated by the receiver host
        distributed_context->recv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&dst_fabric_node_id), sizeof(dst_fabric_node_id)),
            tt::tt_metal::distributed::multihost::Rank{1},  // receive from receiver host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange fabric node id over tag 0
        );
        chip_id_t edge_chip = 0;
        std::vector<chan_id_t> eth_chans;
        if (control_plane.has_intermesh_links(src_physical_device_id)) {
            auto intermesh_routing_direction = control_plane.get_routing_direction_between_neighboring_meshes(
                src_fabric_node_id.mesh_id, dst_fabric_node_id.mesh_id);
            auto eth_cores_and_chans =
                control_plane.get_active_intermesh_links_in_direction(src_fabric_node_id, intermesh_routing_direction);
            for (auto chan : eth_cores_and_chans) {
                if (control_plane.get_routing_plane_id(src_fabric_node_id, chan) == 0) {
                    eth_chans.push_back(chan);
                }
            }
        } else {
            for (auto chip_id : tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
                if (control_plane.has_intermesh_links(chip_id)) {
                    edge_chip = chip_id;
                    break;
                }
            }
            auto edge_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(edge_chip);
            eth_chans = control_plane.get_forwarding_eth_chans_to_chip(src_fabric_node_id, edge_fabric_node_id);
        }

        // Pick any port, for now pick the 1st one in the set
        auto edm_port = *eth_chans.begin();

        log_info(tt::LogTest, "Src MeshId {} ChipId {}", *(src_fabric_node_id.mesh_id), src_fabric_node_id.chip_id);
        log_info(tt::LogTest, "Dst MeshId {} ChipId {}", *(dst_fabric_node_id.mesh_id), dst_fabric_node_id.chip_id);

        auto edm_direction = control_plane.get_eth_chan_direction(src_fabric_node_id, edm_port);
        CoreCoord edm_eth_core = tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
            src_physical_device_id, edm_port);
        log_info(tt::LogTest, "Using edm port {} in direction {}", edm_port, edm_direction);

        CoreCoord receiver_virtual_core = sender_device->worker_core_from_logical_core(receiver_logical_core);
        auto receiver_noc_encoding = tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(
            receiver_virtual_core.x, receiver_virtual_core.y);

        // test parameters
        auto worker_mem_map = generate_worker_mem_map(sender_device, topology);
        uint32_t target_address = worker_mem_map.target_address;
        std::cout << "Write to target addr: " << target_address << std::endl;

        // common compile time args for sender and receiver
        std::vector<uint32_t> compile_time_args = {
            worker_mem_map.test_results_address, worker_mem_map.test_results_size_bytes, target_address};

        // Create the sender program
        auto sender_program = tt_metal::CreateProgram();
        auto sender_kernel = tt_metal::CreateKernel(
            sender_program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp",
            {sender_logical_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = compile_time_args,
                .defines = {}});

        std::vector<uint32_t> sender_runtime_args = {
            worker_mem_map.packet_header_address,
            worker_mem_map.source_l1_buffer_address,
            worker_mem_map.packet_payload_size_bytes,
            num_packets,
            receiver_noc_encoding,
            time_seed,
            mesh_shape[1],
            src_fabric_node_id.chip_id,
            dst_fabric_node_id.chip_id,
            *dst_fabric_node_id.mesh_id};

        // append the EDM connection rt args
        const auto sender_channel = topology == Topology::Mesh ? edm_direction : 0;
        tt::tt_fabric::SenderWorkerAdapterSpec edm_connection = {
            .edm_noc_x = edm_eth_core.x,
            .edm_noc_y = edm_eth_core.y,
            .edm_buffer_base_addr = edm_config.sender_channels_base_address[sender_channel],
            .num_buffers_per_channel = edm_config.sender_channels_num_buffers[sender_channel],
            .edm_l1_sem_addr = edm_config.sender_channels_local_flow_control_semaphore_address[sender_channel],
            .edm_connection_handshake_addr = edm_config.sender_channels_connection_semaphore_address[sender_channel],
            .edm_worker_location_info_addr = edm_config.sender_channels_worker_conn_info_base_address[sender_channel],
            .buffer_size_bytes = edm_config.channel_buffer_size_bytes,
            .buffer_index_semaphore_id = edm_config.sender_channels_buffer_index_semaphore_address[sender_channel],
            .edm_direction = edm_direction};

        auto worker_flow_control_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);
        auto worker_teardown_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);
        auto worker_buffer_index_semaphore_id = tt_metal::CreateSemaphore(sender_program, sender_logical_core, 0);

        append_worker_to_fabric_edm_sender_rt_args(
            edm_connection,
            worker_flow_control_semaphore_id,
            worker_teardown_semaphore_id,
            worker_buffer_index_semaphore_id,
            sender_runtime_args);

        tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_core, sender_runtime_args);

        // Launch sender and receiver programs and wait for them to finish
        fixture->RunProgramNonblocking(sender_device, sender_program);
        fixture->WaitForSingleProgramDone(sender_device, sender_program);

        // Validate status of sender
        std::vector<uint32_t> sender_status;

        tt_metal::detail::ReadFromDeviceL1(
            sender_device,
            sender_logical_core,
            worker_mem_map.test_results_address,
            worker_mem_map.test_results_size_bytes,
            sender_status,
            CoreType::WORKER);

        EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
        uint64_t sender_bytes =
            ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];
        uint64_t receiver_bytes = 0;
        distributed_context->send(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&sender_bytes), sizeof(sender_bytes)),
            tt::tt_metal::distributed::multihost::Rank{1},  // send to receiver host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange tests results over tag 0
        );
        distributed_context->recv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&receiver_bytes), sizeof(receiver_bytes)),
            tt::tt_metal::distributed::multihost::Rank{1},  // recv from receiver host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange tests results over tag 0
        );
        std::cout << "Sender bytes: " << sender_bytes << std::endl;
        std::cout << "Receiver bytes: " << receiver_bytes << std::endl;
        EXPECT_EQ(sender_bytes, receiver_bytes);
    } else {
        // The following code runs on the receiver host

        // Synchronize seeds across hosts (sender and receiver must use the same seed for randomization)
        uint32_t time_seed = 0;
        distributed_context->recv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&time_seed), sizeof(time_seed)),
            tt::tt_metal::distributed::multihost::Rank{0},  // recv from sender host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange seed over tag 0
        );
        std::cout << "Receiver using time seed: " << time_seed << std::endl;
        // Pick an arbitrary receiver and generate its fabric node id
        auto random_dev_list = get_random_numbers_from_range(0, devices.size() - 2, devices.size());
        auto dst_physical_device_id = devices[random_dev_list[0]]->id();
        auto dst_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(dst_physical_device_id);
        auto* receiver_device = DevicePool::instance().get_active_device(dst_physical_device_id);

        const auto& worker_grid_size = receiver_device->compute_with_storage_grid_size();
        auto recv_x = get_random_numbers_from_range(0, worker_grid_size.x - 2, worker_grid_size.x)[0];
        auto recv_y = get_random_numbers_from_range(0, worker_grid_size.y - 2, worker_grid_size.y)[0];
        CoreCoord receiver_logical_core = {recv_x, recv_x};

        std::cout << "Receiver logical core: (" << recv_x << ", " << recv_y << ")" << std::endl;

        distributed_context->send(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&receiver_logical_core), sizeof(receiver_logical_core)),
            tt::tt_metal::distributed::multihost::Rank{0},  // send to sender host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange logical core over tag 0
        );
        std::cout << "Receiver Fabric Node ID: "
                  << "MeshId: " << *(dst_fabric_node_id.mesh_id) << ", ChipId: " << dst_fabric_node_id.chip_id
                  << std::endl;
        // Forward the receiver's fabric node id to the sender host, so it can send packets to the correct destination
        distributed_context->send(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&dst_fabric_node_id), sizeof(dst_fabric_node_id)),
            tt::tt_metal::distributed::multihost::Rank{0},  // send to sender host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange fabric node id over tag 0
        );

        // test parameters
        auto worker_mem_map = generate_worker_mem_map(receiver_device, topology);
        uint32_t target_address = worker_mem_map.target_address;
        std::cout << "Read from target addr: " << target_address << std::endl;

        std::vector<uint32_t> compile_time_args = {
            worker_mem_map.test_results_address,
            worker_mem_map.test_results_size_bytes,
            target_address,
            0 /* mcast_mode */,
            topology == Topology::Mesh,
            fabric_config == tt_metal::FabricConfig::FABRIC_2D_DYNAMIC};

        std::vector<uint32_t> receiver_runtime_args = {
            worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};
        auto receiver_program = tt_metal::CreateProgram();
        auto receiver_kernel = tt_metal::CreateKernel(
            receiver_program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_rx.cpp",
            {receiver_logical_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = compile_time_args});

        tt_metal::SetRuntimeArgs(receiver_program, receiver_kernel, receiver_logical_core, receiver_runtime_args);

        // Launch sender and receiver programs and wait for them to finish
        fixture->RunProgramNonblocking(receiver_device, receiver_program);
        fixture->WaitForSingleProgramDone(receiver_device, receiver_program);
        std::vector<uint32_t> receiver_status;

        tt_metal::detail::ReadFromDeviceL1(
            receiver_device,
            receiver_logical_core,
            worker_mem_map.test_results_address,
            worker_mem_map.test_results_size_bytes,
            receiver_status,
            CoreType::WORKER);
        EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
        uint64_t receiver_bytes =
            ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | receiver_status[TT_FABRIC_WORD_CNT_INDEX];
        uint64_t sender_bytes = 0;
        distributed_context->recv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&sender_bytes), sizeof(sender_bytes)),
            tt::tt_metal::distributed::multihost::Rank{0},  // recv from sender host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange tests results over tag 0
        );
        distributed_context->send(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&receiver_bytes), sizeof(receiver_bytes)),
            tt::tt_metal::distributed::multihost::Rank{0},  // send to sender host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange tests results over tag 0
        );
        EXPECT_EQ(sender_bytes, receiver_bytes);
    }
}

void InterMeshLineMcast(
    BaseFabricFixture* fixture,
    FabricNodeId mcast_request_node,
    FabricNodeId mcast_start_node,
    const std::vector<McastRoutingInfo>& mcast_routing_info,
    const std::vector<FabricNodeId>& mcast_group_node_ids) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    constexpr uint32_t num_packets = 100;

    const auto& fabric_context = control_plane.get_fabric_context();
    const auto topology = fabric_context.get_fabric_topology();
    const auto& edm_config = fabric_context.get_fabric_router_config();

    auto distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    if (*(distributed_context->rank()) == 0) {
        // Synchronize seeds across hosts (sender and receiver must use the same seed for randomization)
        uint32_t time_seed = std::chrono::system_clock::now().time_since_epoch().count();
        distributed_context->send(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&time_seed), sizeof(time_seed)),
            tt::tt_metal::distributed::multihost::Rank{1},  // send to receiver host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange seed over tag 0
        );
        std::cout << "Sender using time seed: " << time_seed << std::endl;

        auto request_phys_id = control_plane.get_physical_chip_id_from_fabric_node_id(mcast_request_node);
        auto request_device = DevicePool::instance().get_active_device(request_phys_id);
        const auto& worker_grid_size = request_device->compute_with_storage_grid_size();
        auto sender_x = get_random_numbers_from_range(0, worker_grid_size.x - 2, worker_grid_size.x)[0];
        auto sender_y = get_random_numbers_from_range(0, worker_grid_size.y - 2, worker_grid_size.y)[0];
        CoreCoord sender_logical_core = {sender_x, sender_y};
        CoreCoord receiver_logical_core = {0, 0};

        distributed_context->recv(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&receiver_logical_core), sizeof(receiver_logical_core)),
            tt::tt_metal::distributed::multihost::Rank{1},  // receive from receiver host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange logical core over tag 0
        );
        std::cout << "Sender logical core: (" << sender_x << ", " << sender_y << ")" << std::endl;

        CoreCoord receiver_virtual_core = request_device->worker_core_from_logical_core(receiver_logical_core);
        auto receiver_noc_encoding = tt::tt_metal::MetalContext::instance().hal().noc_xy_encoding(
            receiver_virtual_core.x, receiver_virtual_core.y);

        std::cout << "Request Mcast from: " << request_phys_id << std::endl;

        auto worker_mem_map = generate_worker_mem_map(request_device, topology);
        uint32_t target_address = worker_mem_map.target_address;

        std::cout << "Sender using seed: " << time_seed << std::endl;
        std::cout << "Writing to: " << target_address << std::endl;
        // common compile time args for sender and receiver
        std::vector<uint32_t> compile_time_args = {
            worker_mem_map.test_results_address, worker_mem_map.test_results_size_bytes, target_address};

        auto mcast_req_program = tt_metal::CreateProgram();
        auto mcast_req_kernel = tt_metal::CreateKernel(
            mcast_req_program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_line_mcast_tx.cpp",
            {sender_logical_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = compile_time_args,
                .defines = {}});

        std::vector<uint32_t> sender_runtime_args = {
            worker_mem_map.packet_header_address,
            worker_mem_map.source_l1_buffer_address,
            worker_mem_map.packet_payload_size_bytes,
            num_packets,
            receiver_noc_encoding,
            time_seed,
            mcast_start_node.chip_id,
            *mcast_start_node.mesh_id};

        std::vector<uint32_t> mcast_header_rtas(4, 0);
        for (const auto& routing_info : mcast_routing_info) {
            mcast_header_rtas[static_cast<uint32_t>(control_plane.routing_direction_to_eth_direction(
                routing_info.mcast_dir))] = routing_info.num_mcast_hops;
        }

        sender_runtime_args.insert(sender_runtime_args.end(), mcast_header_rtas.begin(), mcast_header_rtas.end());

        std::vector<chan_id_t> eth_chans;
        chan_id_t edm_port;
        if (control_plane.has_intermesh_links(request_phys_id)) {
            auto intermesh_routing_direction = control_plane.get_routing_direction_between_neighboring_meshes(
                mcast_request_node.mesh_id, mcast_start_node.mesh_id);
            auto eth_cores_and_chans =
                control_plane.get_active_intermesh_links_in_direction(mcast_request_node, intermesh_routing_direction);
            for (auto chan : eth_cores_and_chans) {
                if (control_plane.get_routing_plane_id(mcast_request_node, chan) == 0) {
                    eth_chans.push_back(chan);
                }
            }
        } else {
            chip_id_t edge_chip = 0;
            for (auto chip_id : tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
                if (control_plane.has_intermesh_links(chip_id)) {
                    edge_chip = chip_id;
                    break;
                }
            }
            auto edge_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(edge_chip);
            eth_chans = control_plane.get_forwarding_eth_chans_to_chip(mcast_request_node, edge_fabric_node_id);
        }

        edm_port = *eth_chans.begin();
        auto edm_direction = control_plane.get_eth_chan_direction(mcast_request_node, edm_port);
        CoreCoord edm_eth_core = tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
            request_phys_id, edm_port);
        log_info(tt::LogTest, "Using edm port {} in direction {}", edm_port, edm_direction);

        const auto sender_channel = topology == Topology::Mesh ? edm_direction : 0;
        tt::tt_fabric::SenderWorkerAdapterSpec edm_connection = {
            .edm_noc_x = edm_eth_core.x,
            .edm_noc_y = edm_eth_core.y,
            .edm_buffer_base_addr = edm_config.sender_channels_base_address[sender_channel],
            .num_buffers_per_channel = edm_config.sender_channels_num_buffers[sender_channel],
            .edm_l1_sem_addr = edm_config.sender_channels_local_flow_control_semaphore_address[sender_channel],
            .edm_connection_handshake_addr = edm_config.sender_channels_connection_semaphore_address[sender_channel],
            .edm_worker_location_info_addr = edm_config.sender_channels_worker_conn_info_base_address[sender_channel],
            .buffer_size_bytes = edm_config.channel_buffer_size_bytes,
            .buffer_index_semaphore_id = edm_config.sender_channels_buffer_index_semaphore_address[sender_channel],
            .edm_direction = edm_direction};

        auto worker_flow_control_semaphore_id = tt_metal::CreateSemaphore(mcast_req_program, sender_logical_core, 0);
        auto worker_teardown_semaphore_id = tt_metal::CreateSemaphore(mcast_req_program, sender_logical_core, 0);
        auto worker_buffer_index_semaphore_id = tt_metal::CreateSemaphore(mcast_req_program, sender_logical_core, 0);

        append_worker_to_fabric_edm_sender_rt_args(
            edm_connection,
            worker_flow_control_semaphore_id,
            worker_teardown_semaphore_id,
            worker_buffer_index_semaphore_id,
            sender_runtime_args);

        tt_metal::SetRuntimeArgs(mcast_req_program, mcast_req_kernel, sender_logical_core, sender_runtime_args);

        log_info(tt::LogTest, "Run Sender on: {}", request_device->id());
        fixture->RunProgramNonblocking(request_device, mcast_req_program);
        fixture->WaitForSingleProgramDone(request_device, mcast_req_program);

        std::vector<uint32_t> sender_status;
        tt_metal::detail::ReadFromDeviceL1(
            request_device,
            sender_logical_core,
            worker_mem_map.test_results_address,
            worker_mem_map.test_results_size_bytes,
            sender_status,
            CoreType::WORKER);
        EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
        uint64_t sender_bytes =
            ((uint64_t)sender_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) | sender_status[TT_FABRIC_WORD_CNT_INDEX];
        distributed_context->send(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&sender_bytes), sizeof(sender_bytes)),
            tt::tt_metal::distributed::multihost::Rank{1},  // send to receiver host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange test results over tag 0
        );
        for (std::size_t recv_idx = 0; recv_idx < mcast_group_node_ids.size() + 1; recv_idx++) {
            uint64_t recv_bytes = 0;
            distributed_context->recv(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&recv_bytes), sizeof(recv_bytes)),
                tt::tt_metal::distributed::multihost::Rank{1},  // recv from receiver host
                tt::tt_metal::distributed::multihost::Tag{0}    // exchange test results over tag 0
            );
            EXPECT_EQ(sender_bytes, recv_bytes);
        }
    } else {
        // Synchronize seeds across hosts (sender and receiver must use the same seed for randomization)
        uint32_t time_seed = 0;
        distributed_context->recv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&time_seed), sizeof(time_seed)),
            tt::tt_metal::distributed::multihost::Rank{0},  // recv from sender host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange seed over tag 0
        );
        auto mcast_start_phys_id = control_plane.get_physical_chip_id_from_fabric_node_id(mcast_start_node);
        auto mcast_start_device = DevicePool::instance().get_active_device(mcast_start_phys_id);

        const auto& worker_grid_size = mcast_start_device->compute_with_storage_grid_size();
        auto recv_x = get_random_numbers_from_range(0, worker_grid_size.x - 2, worker_grid_size.x)[0];
        auto recv_y = get_random_numbers_from_range(0, worker_grid_size.y - 2, worker_grid_size.y)[0];
        CoreCoord receiver_logical_core = {recv_x, recv_x};

        std::cout << "Receiver logical core: (" << recv_x << ", " << recv_y << ")" << std::endl;

        distributed_context->send(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&receiver_logical_core), sizeof(receiver_logical_core)),
            tt::tt_metal::distributed::multihost::Rank{0},  // send to sender host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange logical core over tag 0
        );

        std::vector<tt_metal::IDevice*> mcast_group_devices = {};
        for (auto mcast_node_id : mcast_group_node_ids) {
            mcast_group_devices.push_back(DevicePool::instance().get_active_device(
                control_plane.get_physical_chip_id_from_fabric_node_id(mcast_node_id)));
        }
        const auto topology = control_plane.get_fabric_context().get_fabric_topology();
        auto worker_mem_map = generate_worker_mem_map(mcast_start_device, topology);
        uint32_t target_address = worker_mem_map.target_address;
        std::cout << "Receiver using seed: " << time_seed << std::endl;
        std::cout << "Reading from: " << target_address << std::endl;

        std::vector<uint32_t> compile_time_args = {
            worker_mem_map.test_results_address, worker_mem_map.test_results_size_bytes, target_address};

        std::vector<uint32_t> receiver_runtime_args = {
            worker_mem_map.packet_payload_size_bytes, num_packets, time_seed};
        std::unordered_map<tt_metal::IDevice*, std::shared_ptr<tt_metal::Program>> recv_programs;

        recv_programs[mcast_start_device] =
            create_receiver_program(compile_time_args, receiver_runtime_args, receiver_logical_core);
        for (const auto& dev : mcast_group_devices) {
            recv_programs[dev] =
                create_receiver_program(compile_time_args, receiver_runtime_args, receiver_logical_core);
        }

        for (auto& [dev, recv_program] : recv_programs) {
            log_info(tt::LogTest, "Run receiver on: {}", dev->id());
            fixture->RunProgramNonblocking(dev, *recv_program);
        }

        for (auto& [dev, recv_program] : recv_programs) {
            fixture->WaitForSingleProgramDone(dev, *recv_program);
        }
        uint64_t sender_bytes = 0;
        distributed_context->recv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&sender_bytes), sizeof(sender_bytes)),
            tt::tt_metal::distributed::multihost::Rank{0},  // recv from sender host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange tests results over tag 0
        );

        for (auto& [dev, _] : recv_programs) {
            std::vector<uint32_t> receiver_status;
            std::cout << "Verify mcast recv: " << dev->id() << std::endl;
            tt_metal::detail::ReadFromDeviceL1(
                dev,
                receiver_logical_core,
                worker_mem_map.test_results_address,
                worker_mem_map.test_results_size_bytes,
                receiver_status,
                CoreType::WORKER);

            EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS);
            uint64_t receiver_bytes = ((uint64_t)receiver_status[TT_FABRIC_WORD_CNT_INDEX + 1] << 32) |
                                      receiver_status[TT_FABRIC_WORD_CNT_INDEX];
            distributed_context->send(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&receiver_bytes), sizeof(receiver_bytes)),
                tt::tt_metal::distributed::multihost::Rank{0},  // send to sender host
                tt::tt_metal::distributed::multihost::Tag{0}    // exchange test results over tag 0
            );
            std::cout << "Sender bytes: " << sender_bytes << ", Receiver bytes: " << receiver_bytes << std::endl;
            EXPECT_EQ(sender_bytes, receiver_bytes);
        }
    }
}

std::map<FabricNodeId, chip_id_t> get_physical_chip_mapping_from_eth_coords_mapping(
    const std::vector<std::vector<eth_coord_t>>& mesh_graph_eth_coords) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    std::map<FabricNodeId, chip_id_t> physical_chip_ids_mapping;
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto local_mesh_id = control_plane.get_local_mesh_id_binding();
    for (std::uint32_t mesh_id = 0; mesh_id < mesh_graph_eth_coords.size(); mesh_id++) {
        if (mesh_id == *(local_mesh_id.value())) {
            for (std::uint32_t chip_id = 0; chip_id < mesh_graph_eth_coords[mesh_id].size(); chip_id++) {
                const auto& eth_coord = mesh_graph_eth_coords[mesh_id][chip_id];
                physical_chip_ids_mapping.insert(
                    {FabricNodeId(MeshId{mesh_id}, chip_id), cluster.get_physical_chip_id_from_eth_coord(eth_coord)});
            }
        }
    }
    return physical_chip_ids_mapping;
}

class Custom2x4Fabric2DDynamicFixture : public BaseFabricFixture {
public:
    void SetUp() override {
        static const std::tuple<std::string, std::vector<std::vector<eth_coord_t>>> multi_mesh_2x4_chip_mappings =
            std::tuple{
                "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.yaml",
                std::vector<std::vector<eth_coord_t>>{
                    {{0, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 0, 0}},
                    {{0, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 0, 0}}}};

        tt::tt_metal::MetalContext::instance().set_custom_control_plane_mesh_graph(
            std::get<0>(multi_mesh_2x4_chip_mappings),
            get_physical_chip_mapping_from_eth_coords_mapping(std::get<1>(multi_mesh_2x4_chip_mappings)));

        this->SetUpDevices(tt::tt_metal::FabricConfig::FABRIC_2D_DYNAMIC);
    }

    void TearDown() override {
        BaseFabricFixture::TearDown();
        tt::tt_metal::MetalContext::instance().set_default_control_plane_mesh_graph();
    }
};

TEST_F(Custom2x4Fabric2DDynamicFixture, RandomizedInterMeshUnicast) {
    for (uint32_t i = 0; i < 500; i++) {
        RandomizedInterMeshUnicast(this);
    }
}

TEST_F(Custom2x4Fabric2DDynamicFixture, MultiMeshMulticast) {
    std::vector<FabricNodeId> mcast_req_nodes = {
        FabricNodeId(MeshId{0}, 1), FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 3), FabricNodeId(MeshId{0}, 2)};
    std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{1}, 0)};
    std::vector<McastRoutingInfo> routing_info = {
        McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1}};
    std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
        {FabricNodeId(MeshId{1}, 3)}, {FabricNodeId(MeshId{1}, 1)}};
    for (uint32_t i = 0; i < 500; i++) {
        InterMeshLineMcast(
            this, mcast_req_nodes[i % 4], mcast_start_nodes[i % 2], routing_info, mcast_group_node_ids[i % 2]);
    }
}

TEST_F(Custom2x4Fabric2DDynamicFixture, MultiMeshSouthMulticast) {
    std::vector<FabricNodeId> mcast_req_nodes = {FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 1)};
    std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 1)};
    std::vector<McastRoutingInfo> routing_info = {
        McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1}};
    std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
        {FabricNodeId(MeshId{1}, 2)}, {FabricNodeId(MeshId{1}, 3)}};
    for (uint32_t i = 0; i < 500; i++) {
        InterMeshLineMcast(
            this, mcast_req_nodes[i % 2], mcast_start_nodes[i % 2], routing_info, mcast_group_node_ids[i % 2]);
    }
}

TEST_F(Custom2x4Fabric2DDynamicFixture, MultiMeshNorthMulticast) {
    std::vector<FabricNodeId> mcast_req_nodes = {FabricNodeId(MeshId{0}, 3), FabricNodeId(MeshId{0}, 3)};
    std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{1}, 3)};
    std::vector<McastRoutingInfo> routing_info = {
        McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1}};
    std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
        {FabricNodeId(MeshId{1}, 0)}, {FabricNodeId(MeshId{1}, 1)}};
    for (uint32_t i = 0; i < 500; i++) {
        InterMeshLineMcast(
            this, mcast_req_nodes[i % 2], mcast_start_nodes[i % 2], routing_info, mcast_group_node_ids[i % 2]);
    }
}
}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
