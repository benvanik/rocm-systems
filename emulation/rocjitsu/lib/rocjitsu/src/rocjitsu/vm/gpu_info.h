// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_GPU_INFO_H_
#define ROCJITSU_VM_GPU_INFO_H_

#include "rocjitsu/kmd/linux/sysfs.h"
#include "rocjitsu/vm/rj_vm.h"

namespace rocjitsu {

/// @brief Copy the common passive and activated GPU metadata into its C representation.
rj_vm_gpu_info_t gpu_info_to_c(const Sysfs::GpuInfo &gpu);

} // namespace rocjitsu

#endif // ROCJITSU_VM_GPU_INFO_H_
