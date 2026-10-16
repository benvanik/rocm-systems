// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/host_access_guard.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>

TEST(HostAccessGuardTest, ValidAccessRunsOnce) {
  const std::array<unsigned char, 4> source{1, 2, 3, 4};
  std::array<unsigned char, 4> destination{};
  unsigned calls = 0;
  bool armed_during_access = false;
  EXPECT_TRUE(rocjitsu::with_host_access_guard([&] {
    ++calls;
    armed_during_access = rocjitsu::host_access_guard_state().armed;
    std::memcpy(destination.data(), source.data(), source.size());
  }));
  EXPECT_EQ(calls, 1u);
  EXPECT_EQ(destination, source);
  EXPECT_FALSE(rocjitsu::host_access_guard_state().armed);
#if defined(RJ_DISABLE_HOST_FAULT_RECOVERY)
  EXPECT_FALSE(armed_during_access);
#else
  EXPECT_TRUE(armed_during_access);
#endif
}

#if defined(RJ_DISABLE_HOST_FAULT_RECOVERY)
TEST(HostAccessGuardTest, DisabledRecoveryPreservesHostActions) {
  for (int signal_number : {SIGSEGV, SIGBUS}) {
    struct sigaction before {};
    ASSERT_EQ(sigaction(signal_number, nullptr, &before), 0);
    rocjitsu::install_host_access_guard();
    struct sigaction after {};
    ASSERT_EQ(sigaction(signal_number, nullptr, &after), 0);
    EXPECT_EQ(after.sa_sigaction, before.sa_sigaction);
    EXPECT_EQ(after.sa_flags, before.sa_flags);
    for (int member = 1; member < NSIG; ++member)
      EXPECT_EQ(sigismember(&after.sa_mask, member), sigismember(&before.sa_mask, member));
  }
}
#endif
