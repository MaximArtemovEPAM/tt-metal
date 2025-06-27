
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gtest/gtest.h"
#include <vector>

#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"
#include "intermesh_routing_test_utils.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {

class Custom2x4Fabric2DDynamicFixture : public BaseFabricFixture {
public:
    void SetUp() override {
        local_binding_manager_.validate_local_mesh_id_and_host_rank();

        static const std::tuple<std::string, std::vector<std::vector<eth_coord_t>>> multi_mesh_2x4_chip_mappings =
            std::tuple{
                "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.yaml",
                std::vector<std::vector<eth_coord_t>>{
                    {{0, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 0, 0}},
                    {{0, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 0, 0}}}};
        auto chip_to_eth_coord_mapping = multihost_utils::get_physical_chip_mapping_from_eth_coords_mapping(std::get<1>(multi_mesh_2x4_chip_mappings), 
            local_binding_manager_.get_local_mesh_id());
    
        tt::tt_metal::MetalContext::instance().set_custom_control_plane_mesh_graph(
            std::get<0>(multi_mesh_2x4_chip_mappings),
            chip_to_eth_coord_mapping);
        
        this->SetUpDevices(tt::tt_metal::FabricConfig::FABRIC_2D_DYNAMIC);
    }

    void TearDown() override {
        BaseFabricFixture::TearDown();
        local_binding_manager_.clear_bindings();
        tt::tt_metal::MetalContext::instance().set_default_control_plane_mesh_graph();
        local_binding_manager_.set_bindings();
    }

private:
    multihost_utils::LocalBindingManager local_binding_manager_;
};

} // namespace fabric_router_tests
} // namespace tt::tt_fabric

