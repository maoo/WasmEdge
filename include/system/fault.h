// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2022 Second State INC

//===-- wasmedge/system/fault.h - Memory and arithmetic exception ---------===//
//
// Part of the WasmEdge Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the software exception handler for various operating
/// system.
///
//===----------------------------------------------------------------------===//
#pragma once

#include "common/errcode.h"

namespace WasmEdge {

class Fault {
public:
  Fault();

  ~Fault() noexcept;

  [[noreturn]] static void emitFault(ErrCode Error);

private:
  Fault *Prev = nullptr;
};

} // namespace WasmEdge
