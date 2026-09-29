#pragma once

#include "runtime/contract/request.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ninfer::runtime {

class GenerationBudget {
public:
    GenerationBudget(std::uint32_t effective_tokens, FinishReason limit_reason)
        : remaining_(effective_tokens), limit_reason_(limit_reason) {
        if (limit_reason_ != FinishReason::OutputLimit &&
            limit_reason_ != FinishReason::ContextCapacity) {
            // LOCAL DIAG (budget-abort-name): this abort used to be completely silent. The engine
            // has been dying with 0xC0000409 and no output; name the offending reason instead.
            std::fprintf(stderr,
                         "[ninfer] budget abort: constructor limit_reason=%u effective_tokens=%u\n",
                         static_cast<unsigned>(limit_reason_), effective_tokens);
            std::fflush(stderr);
            std::abort();
        }
    }

    [[nodiscard]] std::uint32_t remaining() const noexcept { return remaining_; }

    [[nodiscard]] FinishReason limit_reason() const noexcept { return limit_reason_; }

    void commit(std::uint32_t tokens) noexcept {
        if (tokens > remaining_) {
            std::fprintf(stderr,
                         "[ninfer] budget abort: commit overrun tokens=%u remaining=%u "
                         "limit_reason=%u\n",
                         tokens, remaining_, static_cast<unsigned>(limit_reason_));
            std::fflush(stderr);
            std::abort();
        }
        remaining_ -= tokens;
    }

private:
    std::uint32_t remaining_   = 0;
    FinishReason limit_reason_ = FinishReason::None;
};

} // namespace ninfer::runtime
