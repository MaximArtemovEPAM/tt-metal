// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <map>

#include <tt-metalium/device.hpp>
#include "umd/device/types/cluster_descriptor_types.h"
#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {

namespace multihost_utils {

class LocalBindingManager {
public:
    void validate_local_mesh_id_and_host_rank() {
        const char* mesh_id_str = std::getenv("TT_MESH_ID");
        const char* host_rank_str = std::getenv("TT_HOST_RANK");
        TT_FATAL(mesh_id_str and host_rank_str,
                "TT_MESH_ID and TT_HOST_RANK environment variables must be set for Multi-Host Fabric Tests.");

        local_mesh_id_ = std::string(mesh_id_str);
        local_host_rank_ = std::string(host_rank_str);
    }

    void clear_bindings() {
        unsetenv("TT_MESH_ID");
        unsetenv("TT_HOST_RANK");
    }

    void set_bindings() {
        setenv("TT_MESH_ID", local_mesh_id_.c_str(), 1);
        setenv("TT_HOST_RANK", local_host_rank_.c_str(), 1);
    }

    uint32_t get_local_mesh_id() const {
        return std::stoi(local_mesh_id_);
    }

    uint32_t get_local_host_rank() const {
        return std::stoi(local_host_rank_);
    }

private:
    std::string local_mesh_id_;
    std::string local_host_rank_;
};

void RandomizedInterMeshUnicast(BaseFabricFixture* fixture);

void InterMeshLineMcast(
    BaseFabricFixture* fixture,
    FabricNodeId mcast_sender_node,
    FabricNodeId mcast_start_node,
    const std::vector<McastRoutingInfo>& mcast_routing_info,
    const std::vector<FabricNodeId>& mcast_group_node_ids);

std::map<FabricNodeId, chip_id_t> get_physical_chip_mapping_from_eth_coords_mapping(
    const std::vector<std::vector<eth_coord_t>>& mesh_graph_eth_coords, uint32_t local_mesh_id);

} // namespace multihost_utils

} // namespace fabric_router_tests
}  // namespace tt::tt_fabric