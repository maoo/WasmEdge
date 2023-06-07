// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2022 Second State INC

#include "common/defines.h"
#include "runtime/instance/module.h"
#include "tensorflowlite_func.h"
#include "tensorflowlite_module.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {
// WasmEdge::Runtime::CallingFrame DummyCallFrame(nullptr, nullptr);

WasmEdge::Runtime::Instance::ModuleInstance *createModule() {
  using namespace std::literals::string_view_literals;
  WasmEdge::Plugin::Plugin::load(std::filesystem::u8path(
      "../../../plugins/wasmedge_tensorflowlite/"
      "libwasmedgePluginWasmEdgeTensorflowLite" WASMEDGE_LIB_EXTENSION));
  if (const auto *Plugin =
          WasmEdge::Plugin::Plugin::find("wasmedge_tensorflowlite"sv)) {
    if (const auto *Module = Plugin->findModule("wasmedge_tensorflowlite"sv)) {
      return Module->create().release();
    }
  }
  return nullptr;
}

/*
void fillMemContent(WasmEdge::Runtime::Instance::MemoryInstance &MemInst,
                    uint32_t Offset, uint32_t Cnt, uint8_t C = 0) noexcept {
  std::fill_n(MemInst.getPointer<uint8_t *>(Offset), Cnt, C);
}

void fillMemContent(WasmEdge::Runtime::Instance::MemoryInstance &MemInst,
                    uint32_t Offset, const std::string &Str) noexcept {
  char *Buf = MemInst.getPointer<char *>(Offset);
  std::copy_n(Str.c_str(), Str.length(), Buf);
}
*/
} // namespace

/*
TEST(WasmEdgeProcessTest, SetProgName) {
  // Create the wasmedge_process module instance.
  auto *ProcMod =
      dynamic_cast<WasmEdge::Host::WasmEdgeProcessModule *>(createModule());
  ASSERT_TRUE(ProcMod != nullptr);

  // Create the calling frame with memory instance.
  WasmEdge::Runtime::Instance::ModuleInstance Mod("");
  Mod.addHostMemory(
      "memory", std::make_unique<WasmEdge::Runtime::Instance::MemoryInstance>(
                    WasmEdge::AST::MemoryType(1)));
  auto *MemInstPtr = Mod.findMemoryExports("memory");
  ASSERT_TRUE(MemInstPtr != nullptr);
  auto &MemInst = *MemInstPtr;
  WasmEdge::Runtime::CallingFrame CallFrame(nullptr, &Mod);

  // Clear the memory[0, 64].
  fillMemContent(MemInst, 0, 64);
  // Set the memory[0, 4] as string "echo".
  fillMemContent(MemInst, 0, std::string("echo"));

  // Get the function "wasmedge_process_set_prog_name".
  auto *FuncInst = ProcMod->findFuncExports("wasmedge_process_set_prog_name");
  EXPECT_NE(FuncInst, nullptr);
  EXPECT_TRUE(FuncInst->isHostFunction());
  auto &HostFuncInst =
      dynamic_cast<WasmEdge::Host::WasmEdgeProcessSetProgName &>(
          FuncInst->getHostFunc());

  // Test: Run function successfully.
  EXPECT_TRUE(HostFuncInst.run(
      CallFrame,
      std::initializer_list<WasmEdge::ValVariant>{UINT32_C(0), UINT32_C(4)},
      {}));
  EXPECT_EQ(ProcMod->getEnv().Name, "echo");

  // Test: Run function with nullptr memory instance -- fail
  EXPECT_FALSE(HostFuncInst.run(
      DummyCallFrame,
      std::initializer_list<WasmEdge::ValVariant>{UINT32_C(0), UINT32_C(4)},
      {}));

  delete ProcMod;
}
*/

TEST(WasmEdgeTensorflowTest, Module) {
  // Create the wasmedge_tensorflow module instance.
  auto *TFLiteMod =
      dynamic_cast<WasmEdge::Host::WasmEdgeTensorflowLiteModule *>(
          createModule());
  EXPECT_FALSE(TFLiteMod == nullptr);
  EXPECT_EQ(TFLiteMod->getFuncExportNum(), 7U);
  EXPECT_NE(TFLiteMod->findFuncExports("create_session"), nullptr);
  EXPECT_NE(TFLiteMod->findFuncExports("delete_session"), nullptr);
  EXPECT_NE(TFLiteMod->findFuncExports("run_session"), nullptr);
  EXPECT_NE(TFLiteMod->findFuncExports("get_output_tensor"), nullptr);
  EXPECT_NE(TFLiteMod->findFuncExports("get_tensor_len"), nullptr);
  EXPECT_NE(TFLiteMod->findFuncExports("get_tensor_data"), nullptr);
  EXPECT_NE(TFLiteMod->findFuncExports("append_input"), nullptr);
  delete TFLiteMod;
}

GTEST_API_ int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
