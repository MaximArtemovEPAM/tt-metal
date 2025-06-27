#pragma once
#include "hostdevcommon/common_values.hpp"
#include "ttnn_test_fixtures.hpp"
#include <cstddef>
#include <memory>

// Suite-level device fixture: opens device once per suite, closes once after all tests
class TTNNSuiteDeviceFixture : public ttnn::TTNNFixtureBase {
protected:
    static std::shared_ptr<tt::tt_metal::distributed::MeshDevice> static_device_holder_;
    static tt::tt_metal::distributed::MeshDevice* static_device_;
    tt::tt_metal::distributed::MeshDevice* device_ = nullptr;

    static size_t L1_SMALL_SIZE;
    static size_t TRACE_REGION_SIZE;
    static void SetUpTestSuite() {
        size_t numDevices = ::tt::tt_metal::GetNumAvailableDevices();
        size_t numPCIeDevices = ::tt::tt_metal::GetNumPCIeDevices();
        ::tt::tt_metal::DispatchCoreType dispatchCoreType = numDevices == numPCIeDevices
                                                                ? ::tt::tt_metal::DispatchCoreType::WORKER
                                                                : ::tt::tt_metal::DispatchCoreType::ETH;

        ::tt::tt_metal::distributed::MeshShape shape{1, 1};
        static_device_holder_ = ::tt::tt_metal::distributed::MeshDevice::create(
            ::tt::tt_metal::distributed::MeshDeviceConfig{shape},
            /* l1_small_size = */ L1_SMALL_SIZE,
            /* trace_region_size = */ TRACE_REGION_SIZE,
            /* num_hw_cqs = */ 1,
            dispatchCoreType);
        static_device_ = static_device_holder_.get();
    }
    static void TearDownTestSuite() {
        if (static_device_) {
            static_device_->close();
        }
        static_device_holder_.reset();
        static_device_ = nullptr;
    }

    void SetUp() override { device_ = static_device_; }
    void TearDown() override {}

    TTNNSuiteDeviceFixture() : ttnn::TTNNFixtureBase() {}
    TTNNSuiteDeviceFixture(int trace_region_size, int l1_small_size) :
        ttnn::TTNNFixtureBase(trace_region_size, l1_small_size) {}
};

std::shared_ptr<tt::tt_metal::distributed::MeshDevice> TTNNSuiteDeviceFixture::static_device_holder_ = nullptr;
tt::tt_metal::distributed::MeshDevice* TTNNSuiteDeviceFixture::static_device_ = nullptr;
size_t TTNNSuiteDeviceFixture::L1_SMALL_SIZE = 1 << 15;
size_t TTNNSuiteDeviceFixture::TRACE_REGION_SIZE = 500000;
