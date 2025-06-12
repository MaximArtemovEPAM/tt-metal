// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

constexpr const char* TERM_RED = "\033[31m";
constexpr const char* TERM_GREEN = "\033[32m";
constexpr const char* TERM_YELLOW = "\033[33m";
constexpr const char* TERM_BLUE = "\033[34m";
constexpr const char* TERM_MAGENTA = "\033[35m";
constexpr const char* TERM_CYAN = "\033[36m";
constexpr const char* TERM_WHITE = "\033[37m";
constexpr const char* TERM_RESET = "\033[0m";

constexpr const char* TERM_COMPUTE = TERM_CYAN;
constexpr const char* TERM_READER = TERM_YELLOW;
constexpr const char* TERM_WRITER = TERM_GREEN;
