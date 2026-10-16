// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/rj_vm_config.h"

#include "embedded_schema.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/gpu_info.h"
#include "util/log.h"

#include <algorithm>
#include <memory>
#include <new>
#include <string>
#include <vector>

struct rj_vm_config_t {
  std::string json; ///< Owned input snapshot used for every VM construction.
  std::vector<rocjitsu::Sysfs::GpuInfo> devices; ///< Immutable metadata in GPU ordinal order.
};

rj_status_t rj_vm_config_create_from_string(const char *json, rj_vm_config_t **config) {
  if (!json || !config)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  try {
    auto owned = std::make_unique<rj_vm_config_t>();
    owned->json = json;
    const auto identity =
        rocjitsu::config::load_device_identity_from_string(owned->json, rocjitsu::kEmbeddedSchema);
    if (identity.num_gpus == 0 || identity.num_xcc == 0)
      return ROCJITSU_STATUS_UNSUPPORTED;
    owned->devices.reserve(identity.num_gpus);
    for (uint32_t index = 0; index < identity.num_gpus; ++index)
      owned->devices.push_back(rocjitsu::gpu_info_from_config(
          rocjitsu::config::device_identity_at(identity, index), identity.num_xcc));
    *config = owned.release();
    return ROCJITSU_STATUS_SUCCESS;
  } catch (const std::bad_alloc &) {
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  } catch (const std::exception &error) {
    util::Logger::warn("rj_vm_config_create_from_string failed: ", error.what());
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  }
}

void rj_vm_config_destroy(rj_vm_config_t *config) { delete config; }

rj_status_t rj_vm_config_gpu_count(const rj_vm_config_t *config, uint32_t *count) {
  if (!config || !count)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  *count = static_cast<uint32_t>(config->devices.size());
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_config_gpu_info(const rj_vm_config_t *config, uint32_t index,
                                  rj_vm_gpu_info_t *info) {
  if (!config || !info || index >= config->devices.size())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  *info = rocjitsu::gpu_info_to_c(config->devices[index]);
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_config_gpu_ip_version(const rj_vm_config_t *config, uint32_t index,
                                        uint32_t hardware_id, rj_vm_ip_version_t *version) {
  if (!config || !version || index >= config->devices.size())
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  const auto &versions = config->devices[index].ip_versions;
  const auto found =
      std::ranges::find(versions, hardware_id, &rocjitsu::config::DeviceIpVersion::hardware_id);
  if (found == versions.end())
    return ROCJITSU_STATUS_UNSUPPORTED;
  *version = {found->major, found->minor, found->revision};
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_create_from_config(const rj_vm_config_t *config, rj_vm_mode_t mode,
                                     rj_vm_t **vm) {
  if (!config || !vm)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  return rj_vm_create_from_string(config->json.c_str(), mode, vm);
}
