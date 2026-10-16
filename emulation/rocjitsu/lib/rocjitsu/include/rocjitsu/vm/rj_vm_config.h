// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_vm_config.h
/// @brief Passive GPU discovery from an owned VM configuration snapshot.

#ifndef ROCJITSU_VM_RJ_VM_CONFIG_H_
#define ROCJITSU_VM_RJ_VM_CONFIG_H_

#include "rocjitsu/vm/rj_vm.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Immutable JSON and device metadata, without a simulated machine.
/// @details The handle owns its input snapshot. Queries may run concurrently;
/// destruction requires exclusive access. The library must remain loaded until
/// the handle and every VM created from it have been destroyed.
typedef struct rj_vm_config_t rj_vm_config_t;

/// @brief Exact version of one configured hardware IP block.
typedef struct rj_vm_ip_version_t {
  uint32_t major;    ///< Hardware IP major version.
  uint32_t minor;    ///< Hardware IP minor version.
  uint32_t revision; ///< Hardware IP revision, independent of compiler target spelling.
} rj_vm_ip_version_t;

/// @brief Copy JSON and discover its configured GPUs without activating them.
/// @details Parses metadata without constructing simulator components, creating
/// workers, opening a simulated process, or generating sysfs files. Full VM
/// construction may still reject unsupported execution configuration. Failure
/// leaves @p config unchanged and creates no cleanup obligation.
/// @param[in] json Null-terminated JSON, borrowed only during this call.
/// @param[out] config Newly owned passive configuration handle.
RJ_API_EXPORT rj_status_t rj_vm_config_create_from_string(const char *json,
                                                          rj_vm_config_t **config);

/// @brief Destroy a passive configuration; NULL is a no-op.
RJ_API_EXPORT void rj_vm_config_destroy(rj_vm_config_t *config);

/// @brief Return the number of configured GPUs without allocation or activation.
RJ_API_EXPORT rj_status_t rj_vm_config_gpu_count(const rj_vm_config_t *config, uint32_t *count);

/// @brief Copy metadata for one zero-based GPU ordinal.
/// @details Performs no allocation or activation. An invalid argument or index
/// leaves @p info unchanged and returns ROCJITSU_STATUS_INVALID_ARGUMENT.
RJ_API_EXPORT rj_status_t rj_vm_config_gpu_info(const rj_vm_config_t *config, uint32_t index,
                                                rj_vm_gpu_info_t *info);

/// @brief Query an exact hardware IP version by its discovery hardware ID.
/// @details For example, hardware ID 11 selects GC and 42 selects SDMA0.
/// Missing IP metadata returns ROCJITSU_STATUS_UNSUPPORTED. Every failure
/// leaves @p version unchanged. This query performs no allocation or activation.
RJ_API_EXPORT rj_status_t rj_vm_config_gpu_ip_version(const rj_vm_config_t *config, uint32_t index,
                                                      uint32_t hardware_id,
                                                      rj_vm_ip_version_t *version);

/// @brief Create a VM from the configuration's retained JSON snapshot.
/// @details The VM owns its construction state independently of @p config;
/// the configuration may be destroyed after this call returns. Mode and VM
/// lifetime follow rj_vm_create_from_string. Failure leaves @p vm unchanged.
RJ_API_EXPORT rj_status_t rj_vm_create_from_config(const rj_vm_config_t *config, rj_vm_mode_t mode,
                                                   rj_vm_t **vm);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROCJITSU_VM_RJ_VM_CONFIG_H_
