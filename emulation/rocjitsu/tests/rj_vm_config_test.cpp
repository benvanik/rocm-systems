// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/rj_vm_config.h"
#include "test_paths.h"

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>

namespace {

using ConfigOwner = std::unique_ptr<rj_vm_config_t, decltype(&rj_vm_config_destroy)>;
using VmOwner = std::unique_ptr<rj_vm_t, decltype(&rj_vm_destroy)>;

std::string read_config(const char *name, uint32_t count = 1) {
  std::ifstream input(std::string(CONFIG_DIR) + "/" + name);
  std::string json{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  const auto position = json.find("\"gpu\": {");
  if (position != std::string::npos)
    json.insert(position + std::strlen("\"gpu\": {"),
                "\"num_gpus\":" + std::to_string(count) + ",");
  // The real topology and identity remain intact; this bounds host workers.
  json.insert(1, "\"cpu_thread_budget\":1,");
  return json;
}

struct ConfigCase {
  const char *file; ///< Canonical config source.
  uint32_t gpu_id;  ///< Base KFD GPU identity.
  uint32_t gfx;     ///< Compiler target identity.
};

class VmConfigTest : public testing::TestWithParam<ConfigCase> {};

TEST_P(VmConfigTest, EnumeratesIndependentGpuIdentitiesWithoutAVm) {
  auto json = read_config(GetParam().file, 4);
  rj_vm_config_t *raw = nullptr;
  ASSERT_EQ(rj_vm_config_create_from_string(json.c_str(), &raw), ROCJITSU_STATUS_SUCCESS);
  ConfigOwner config(raw, rj_vm_config_destroy);
  uint32_t count = 0;
  ASSERT_EQ(rj_vm_config_gpu_count(config.get(), &count), ROCJITSU_STATUS_SUCCESS);
  ASSERT_EQ(count, 4u);
  uint64_t first_unique_id = 0;
  for (uint32_t index = 0; index < count; ++index) {
    rj_vm_gpu_info_t info{};
    ASSERT_EQ(rj_vm_config_gpu_info(config.get(), index, &info), ROCJITSU_STATUS_SUCCESS);
    EXPECT_EQ(info.present, 1u);
    EXPECT_EQ(info.gpu_id, GetParam().gpu_id + index);
    EXPECT_EQ(info.gfx_target_version, GetParam().gfx);
    EXPECT_EQ(info.location_id, 0x0300u + (index << 8));
    EXPECT_EQ(info.drm_render_minor, 128u + index);
    EXPECT_EQ(info.num_xcc, 1u);
    EXPECT_EQ(info.wave_front_size, 32u);
    if (index == 0)
      first_unique_id = info.unique_id;
    EXPECT_EQ(info.unique_id, first_unique_id + index);
  }
}

TEST_P(VmConfigTest, CreatesFromOwnedSnapshotAndReleasesConfigurationBeforeVm) {
  auto json = read_config(GetParam().file);
  rj_vm_config_t *raw = nullptr;
  ASSERT_EQ(rj_vm_config_create_from_string(json.c_str(), &raw), ROCJITSU_STATUS_SUCCESS);
  ConfigOwner config(raw, rj_vm_config_destroy);
  rj_vm_gpu_info_t expected{};
  ASSERT_EQ(rj_vm_config_gpu_info(config.get(), 0, &expected), ROCJITSU_STATUS_SUCCESS);
  json.assign("not JSON");
  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(rj_vm_create_from_config(config.get(), RJ_VM_MODE_LOCAL, &raw_vm),
            ROCJITSU_STATUS_SUCCESS);
  VmOwner vm(raw_vm, rj_vm_destroy);
  config.reset();
  rj_vm_gpu_info_t observed{};
  ASSERT_EQ(rj_vm_gpu_info(vm.get(), &observed), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(observed.gpu_id, expected.gpu_id);
  EXPECT_EQ(observed.unique_id, expected.unique_id);
  EXPECT_EQ(observed.location_id, expected.location_id);
  EXPECT_EQ(observed.drm_render_minor, expected.drm_render_minor);
  EXPECT_EQ(observed.gfx_target_version, expected.gfx_target_version);
  EXPECT_EQ(observed.num_xcc, expected.num_xcc);
  EXPECT_EQ(observed.simd_count, expected.simd_count);
  EXPECT_EQ(observed.simd_per_cu, expected.simd_per_cu);
  EXPECT_EQ(observed.num_shader_engines, expected.num_shader_engines);
  EXPECT_EQ(observed.num_shader_arrays_per_engine, expected.num_shader_arrays_per_engine);
  EXPECT_EQ(observed.num_cu_per_sh, expected.num_cu_per_sh);
  EXPECT_EQ(observed.max_waves_per_simd, expected.max_waves_per_simd);
  EXPECT_EQ(observed.lds_size_kb, expected.lds_size_kb);
  EXPECT_STREQ(observed.marketing_name, expected.marketing_name);
}

TEST_P(VmConfigTest, FailedQueriesPreserveOutputs) {
  const auto json = read_config(GetParam().file);
  rj_vm_config_t *raw = nullptr;
  ASSERT_EQ(rj_vm_config_create_from_string(json.c_str(), &raw), ROCJITSU_STATUS_SUCCESS);
  ConfigOwner config(raw, rj_vm_config_destroy);
  rj_vm_gpu_info_t info;
  std::memset(&info, 0x5a, sizeof(info));
  const auto before = info;
  EXPECT_EQ(rj_vm_config_gpu_info(config.get(), 1, &info), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(std::memcmp(&before, &info, sizeof(info)), 0);
  EXPECT_EQ(rj_vm_config_gpu_info(nullptr, 0, &info), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(std::memcmp(&before, &info, sizeof(info)), 0);
  EXPECT_EQ(rj_vm_config_gpu_info(config.get(), 0, nullptr), ROCJITSU_STATUS_INVALID_ARGUMENT);
  uint32_t count = 99;
  EXPECT_EQ(rj_vm_config_gpu_count(nullptr, &count), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(count, 99u);
  EXPECT_EQ(rj_vm_config_gpu_count(config.get(), nullptr), ROCJITSU_STATUS_INVALID_ARGUMENT);
}

TEST_P(VmConfigTest, ReplicatedIdentitiesStayWithinBusAddressSpace) {
  auto json = read_config(GetParam().file, 253);
  rj_vm_config_t *raw = nullptr;
  ASSERT_EQ(rj_vm_config_create_from_string(json.c_str(), &raw), ROCJITSU_STATUS_SUCCESS);
  ConfigOwner config(raw, rj_vm_config_destroy);
  rj_vm_gpu_info_t info{};
  ASSERT_EQ(rj_vm_config_gpu_info(config.get(), 252, &info), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(info.location_id, 0xff00u);
  json = read_config(GetParam().file, 254);
  auto *unchanged = raw;
  EXPECT_EQ(rj_vm_config_create_from_string(json.c_str(), &unchanged),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(unchanged, raw);
}

INSTANTIATE_TEST_SUITE_P(Targets, VmConfigTest,
                         testing::Values(ConfigCase{"gfx1151.json", 5510, 110501},
                                         ConfigCase{"gfx1201_r9700.json", 8716, 120001}));

TEST(VmConfigArgumentsTest, FailedConstructionPreservesOutput) {
  rj_vm_config_t *const sentinel = reinterpret_cast<rj_vm_config_t *>(uintptr_t{1});
  auto *config = sentinel;
  EXPECT_EQ(rj_vm_config_create_from_string(nullptr, &config), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(config, sentinel);
  EXPECT_EQ(rj_vm_config_create_from_string("not JSON", &config), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(config, sentinel);
  EXPECT_EQ(rj_vm_config_create_from_string("{}", &config), ROCJITSU_STATUS_UNSUPPORTED);
  EXPECT_EQ(config, sentinel);
  EXPECT_EQ(rj_vm_config_create_from_string("{}", nullptr), ROCJITSU_STATUS_INVALID_ARGUMENT);
  rj_vm_config_destroy(nullptr);
}

TEST(VmConfigArgumentsTest, IndependentConfigurationsKeepTheirOwnSnapshots) {
  rj_vm_config_t *first_raw = nullptr;
  ASSERT_EQ(rj_vm_config_create_from_string(read_config("gfx1151.json", 4).c_str(), &first_raw),
            ROCJITSU_STATUS_SUCCESS);
  ConfigOwner first(first_raw, rj_vm_config_destroy);
  rj_vm_config_t *second_raw = nullptr;
  ASSERT_EQ(rj_vm_config_create_from_string(read_config("gfx1201_r9700.json").c_str(), &second_raw),
            ROCJITSU_STATUS_SUCCESS);
  ConfigOwner second(second_raw, rj_vm_config_destroy);
  first.reset();
  uint32_t count = 0;
  EXPECT_EQ(rj_vm_config_gpu_count(second.get(), &count), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(count, 1u);
  rj_vm_gpu_info_t info{};
  EXPECT_EQ(rj_vm_config_gpu_info(second.get(), 0, &info), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(info.gfx_target_version, 120001u);
}

} // namespace
