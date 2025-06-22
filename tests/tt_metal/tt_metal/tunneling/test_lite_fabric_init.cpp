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
#include "dispatch_fixture.hpp"
#include "utils.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/kernels/kernel_impl.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"

// Tests in this file are to verify the initialization / handshake sequence for setting up Lite Fabric kernel
// on remote chips. The tests don't require remote chips, if all chips have PCIe, remote chips are spoofed.

namespace tt::tt_metal {
namespace tunneling {

TEST_F(DispatchFixture, SingleRemoteChipInit) {  // make this device fixture
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping test for Wormhole B0, as it does not support tunneling yet";
    }
    // Start with first device and find its connected chip, connected chip will be the remote chip
    chip_id_t mmio_chip_id = MetalContext::instance().get_cluster().get_associated_mmio_device(0);
    tt::tt_metal::IDevice* mmio_device = nullptr;
    tt::tt_metal::IDevice* remote_device = nullptr;
    std::optional<CoreCoord> mmio_chip_eth = std::nullopt;
    std::optional<CoreCoord> remote_chip_eth = std::nullopt;
    for (auto device : devices_) {
        if (device->id() == MetalContext::instance().get_cluster().get_associated_mmio_device(device->id())) {
            mmio_device = device;
            // whichever chip this is connected to will be considered the remote chip
            for (const auto& active_eth : device->get_active_ethernet_cores()) {
                if (MetalContext::instance().get_cluster().is_ethernet_link_up(mmio_device->id(), active_eth)) {
                    mmio_chip_eth = active_eth;
                    auto connected_chip_eth = MetalContext::instance().get_cluster().get_connected_ethernet_core(
                        {mmio_device->id(), active_eth});
                    auto remote_device_id = std::get<0>(connected_chip_eth);
                    remote_chip_eth = std::get<1>(connected_chip_eth);
                    for (auto potential_remote_device : devices_) {
                        if (potential_remote_device->id() == remote_device_id) {
                            remote_device = potential_remote_device;
                            break;
                        }
                    }
                    break;
                }
            }
            if (remote_device != nullptr and remote_chip_eth.has_value()) {
                break;
            }
        }
    }
    if (mmio_device == nullptr || remote_device == nullptr || !mmio_chip_eth.has_value() ||
        !remote_chip_eth.has_value()) {
        GTEST_SKIP() << "Skipping test, could not find connected devices to act as mmio and remote";
    }

    std::cout << "MMIO Device: " << mmio_device->id() << ", Remote Device: " << remote_device->id() << std::endl;
    std::cout << "MMIO Chip Eth: " << mmio_chip_eth->str() << ", Remote Chip Eth: " << remote_chip_eth->str()
              << std::endl;

    // Create a program on the MMIO device with the kernel that is responsible for loading itself onto the remote eth.
    // This kernel will stall until it receives a signal from the remote eth core.
    // Remote eth core will complete the handshake only after all ethernets on its chip have been initialized.
    tt_metal::Program mmio_program = tt_metal::Program();

    auto mmio_eth_kernel = tt_metal::CreateKernel(
        mmio_program,
        "tests/tt_metal/tt_metal/tunneling/kernels/lite_fabric_handshake.cpp",
        mmio_chip_eth.value(),
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0});

    // Compile the program because we need to write the binary into mmio eth core so it can send it over
    tt_metal::detail::CompileProgram(mmio_device, mmio_program);

    // Extract the binary and write it to the mmio eth core
    const auto& kernels = mmio_program.get_kernels(static_cast<uint32_t>(HalProgrammableCoreType::ACTIVE_ETH));
    auto eth_kernel = kernels.at(mmio_eth_kernel);

    const ll_api::memory& binary_mem =
        *tt_metal::KernelImpl::from(*eth_kernel)
             .binaries(BuildEnvManager::get_instance().get_device_build_env(mmio_device->build_id()).build_key)[0];
    auto binary_address = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);

    auto num_spans = binary_mem.num_spans();
    uint32_t erisc_core_type =
        MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::ACTIVE_ETH);
    uint32_t processor_class_idx = magic_enum::enum_integer(HalProcessorClassType::DM);
    int processor_type_idx = magic_enum::enum_integer(std::get<EthernetConfig>(eth_kernel->config()).processor);

    TT_FATAL(
        binary_mem.num_spans() == 1,
        "Expected 1 binary span for lite fabric handshake kernel, got {}",
        binary_mem.num_spans());
    uint64_t local_init_addr = tt::tt_metal::MetalContext::instance()
                                   .hal()
                                   .get_jit_build_config(erisc_core_type, processor_class_idx, processor_type_idx)
                                   .local_init_addr;
    uint32_t dst_binary_address;
    uint32_t binary_size_bytes;
    binary_mem.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len_words) {
        uint32_t relo_addr = tt::tt_metal::MetalContext::instance().hal().relocate_dev_addr(addr, local_init_addr);
        dst_binary_address = relo_addr;
        binary_size_bytes = len_words * sizeof(uint32_t);
    });

    auto virtual_eth_core = mmio_device->ethernet_core_from_logical_core(mmio_chip_eth.value());
    llrt::write_binary_to_address(binary_mem, mmio_device->id(), virtual_eth_core, binary_address);

    std::cout << "virtual_eth_core " << virtual_eth_core.str() << std::endl;
    std::cout << "dst_binary_address: " << dst_binary_address << " binary_size_bytes " << binary_size_bytes
              << std::endl;
    tt_metal::SetRuntimeArgs(
        mmio_program,
        mmio_eth_kernel,
        mmio_chip_eth.value(),
        {
            0,
            binary_address,      // Address where the binary is written on mmio core
            dst_binary_address,  // where active eth binaries should be written ...
            binary_size_bytes,   // size of the binary .... how to get this?
        });

    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(
        mmio_device->id());  // don't need launch program should do
    tt_metal::detail::LaunchProgram(mmio_device, mmio_program);
}

}  // namespace tunneling
}  // namespace tt::tt_metal
