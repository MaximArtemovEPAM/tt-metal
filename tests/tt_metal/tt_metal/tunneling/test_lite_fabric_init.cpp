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
#include "device_fixture.hpp"
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

struct MmmioAndEthDeviceDesc {
    IDevice* mmio_device = nullptr;
    IDevice* eth_device = nullptr;
    std::optional<CoreCoord> mmio_eth = std::nullopt;
    std::optional<CoreCoord> eth_to_init = std::nullopt;
};

void get_mmio_device_and_eth_device_to_init(const std::vector<IDevice*>& devices, MmmioAndEthDeviceDesc& desc) {
    for (auto device : devices) {
        if (device->id() == MetalContext::instance().get_cluster().get_associated_mmio_device(device->id())) {
            desc.mmio_device = device;
            // whichever chip this is connected to will be considered the remote chip
            for (const auto& active_eth : device->get_active_ethernet_cores()) {
                if (MetalContext::instance().get_cluster().is_ethernet_link_up(desc.mmio_device->id(), active_eth)) {
                    desc.mmio_eth = active_eth;
                    auto connected_chip_eth = MetalContext::instance().get_cluster().get_connected_ethernet_core(
                        {desc.mmio_device->id(), active_eth});
                    auto remote_device_id = std::get<0>(connected_chip_eth);
                    desc.eth_to_init = std::get<1>(connected_chip_eth);
                    for (auto potential_remote_device : devices) {
                        if (potential_remote_device->id() == remote_device_id) {
                            desc.eth_device = potential_remote_device;
                            break;
                        }
                    }
                    break;
                }
            }
            if (desc.eth_device != nullptr and desc.eth_to_init.has_value()) {
                break;
            }
        }
    }

    if (desc.mmio_device == nullptr || desc.eth_device == nullptr || !desc.mmio_eth.has_value() ||
        !desc.eth_to_init.has_value()) {
        GTEST_SKIP() << "Skipping test, could not find connected devices to act as mmio and eth connected device";
    }

    std::cout << "MMIO Device: " << desc.mmio_device->id() << ", Remote Device: " << desc.eth_device->id() << std::endl;
    std::cout << "MMIO Chip Eth: " << desc.mmio_eth->str() << ", Remote Chip Eth: " << desc.eth_to_init->str()
              << std::endl;
}

Program create_eth_init_program(const MmmioAndEthDeviceDesc& desc, bool init_all_eth_cores) {
    // Create a program on the MMIO device with the kernel that is responsible for loading itself onto the remote eth.
    // This kernel will stall until it receives a signal from the remote eth core.
    // Remote eth core will complete the handshake only after all ethernets on its chip have been initialized.
    tt_metal::Program mmio_program = tt_metal::Program();

    std::unordered_map<CoreCoord, KernelHandle> mmio_eth_to_kernel;
    for (const auto& core : desc.mmio_device->get_active_ethernet_cores()) {
        if (!init_all_eth_cores && core != desc.mmio_eth.value()) {
            continue;  // Skip other eth cores if we are initializing only one
        }
        auto kernel_handle = tt_metal::CreateKernel(
            mmio_program,
            "tests/tt_metal/tt_metal/tunneling/kernels/lite_fabric_handshake.cpp",
            desc.mmio_eth.value(),
            tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0});
        mmio_eth_to_kernel[core] = kernel_handle;
    }

    // Compile the program because we need to write the binary into mmio eth core so it can send it over
    tt_metal::detail::CompileProgram(desc.mmio_device, mmio_program);

    // Extract the binary and write it to the mmio eth core
    const auto& kernels = mmio_program.get_kernels(static_cast<uint32_t>(HalProgrammableCoreType::ACTIVE_ETH));
    auto eth_kernel = kernels.at(mmio_eth_to_kernel.at(desc.mmio_eth.value()));

    const ll_api::memory& binary_mem =
        *tt_metal::KernelImpl::from(*eth_kernel)
             .binaries(BuildEnvManager::get_instance().get_device_build_env(desc.mmio_device->build_id()).build_key)[0];

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

    std::cout << "dst_binary_address: " << dst_binary_address << " binary_size_bytes " << binary_size_bytes
              << std::endl;

    for (const auto& [core, kernel_handle] : mmio_eth_to_kernel) {
        uint32_t initial_state = (core == desc.mmio_eth.value()) ? 0 : 1;

        tt_metal::SetRuntimeArgs(
            mmio_program,
            kernel_handle,
            core,
            {initial_state, dst_binary_address, binary_size_bytes, init_all_eth_cores});
    }

    return mmio_program;
}

TEST_F(DeviceFixture, MmioEthCoreInitSingleEthCore) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping test for Wormhole B0, as it does not support tunneling yet";
    }
    // if (devices_.size() != 2) {
    //     GTEST_SKIP() << "Only expect to be initializing 1 eth device per MMIO chip. Test should ";
    // }

    MmmioAndEthDeviceDesc desc;
    get_mmio_device_and_eth_device_to_init(devices_, desc);

    auto mmio_program = create_eth_init_program(desc, false);

    auto virtual_eth_core = desc.mmio_device->ethernet_core_from_logical_core(desc.mmio_eth.value());
    std::cout << "virtual_eth_core " << virtual_eth_core.str() << std::endl;

    std::cout << "Number of active ethernet cores on mmio device: "
              << desc.mmio_device->get_active_ethernet_cores().size() << std::endl;

    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(
        desc.mmio_device->id());  // don't need launch program should do
    tt_metal::detail::LaunchProgram(desc.mmio_device, mmio_program);
}

TEST_F(DeviceFixture, MmioEthCoreInitAllEthCores) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping test for Wormhole B0, as it does not support tunneling yet";
    }
    if (devices_.size() != 2) {
        GTEST_SKIP() << "Only expect to be initializing 1 eth device per MMIO chip. Test should ";
    }

    MmmioAndEthDeviceDesc desc;
    get_mmio_device_and_eth_device_to_init(devices_, desc);

    auto mmio_program = create_eth_init_program(desc, true);

    auto virtual_eth_core = desc.mmio_device->ethernet_core_from_logical_core(desc.mmio_eth.value());
    std::cout << "virtual_eth_core " << virtual_eth_core.str() << std::endl;

    std::cout << "Number of active ethernet cores on mmio device: "
              << desc.mmio_device->get_active_ethernet_cores().size() << std::endl;

    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(
        desc.mmio_device->id());  // don't need launch program should do
    tt_metal::detail::LaunchProgram(desc.mmio_device, mmio_program);
}

}  // namespace tunneling
}  // namespace tt::tt_metal
