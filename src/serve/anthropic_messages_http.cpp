#include "serve/http_server.h"

#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"
#include "serve/request_json.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {

namespace {

// LOCAL DIAGNOSTIC (reuse-fingerprint): FNV-1a/64 over as-received JSON sections.
// ordered_json preserves key order, so serializer jitter (e.g. reshuffled tool
// schemas) changes the hash exactly when it would change the token stream.
std::string fnv1a_hex(const std::string& bytes) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : bytes) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    char out[17];
    std::snprintf(out, sizeof(out), "%016llx", static_cast<unsigned long long>(hash));
    return std::string(out);
}

PromptFingerprint make_prompt_fingerprint(const RequestJson& body) {
    PromptFingerprint fingerprint;
    const auto tools = body.find("tools");
    fingerprint.tools =
        fnv1a_hex(tools == body.end() ? std::string("null") : tools->dump());
    const auto system = body.find("system");
    fingerprint.system =
        fnv1a_hex(system == body.end() ? std::string("null") : system->dump());
    const auto messages = body.find("messages");
    if (messages != body.end() && messages->is_array()) {
        fingerprint.messages.reserve(messages->size());
        for (const auto& message : *messages) {
            fingerprint.messages.push_back(fnv1a_hex(message.dump()));
        }
    }
    return fingerprint;
}

} // namespace

void HttpServer::handle_count_tokens(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = new_anthropic_request_id();
    res.set_header("request-id", request_id);
    try {
        const AnthropicCountTokensRequest request =
            parse_anthropic_count_tokens_request(parse_json_body(req));
        const int input_tokens = service_->count_prompt_tokens(
            request.generation, [&req] { return client_disconnected(req); });
        res.set_content(make_anthropic_count_tokens_response(input_tokens), "application/json");
    } catch (const ApiException& exception) {
        write_anthropic_error(res, exception.error(), request_id);
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "anthropic_count_tokens",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()), request_id);
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
    }
}

void HttpServer::handle_messages(const httplib::Request& req, httplib::Response& res) {
    const std::string request_id = new_anthropic_request_id();
    res.set_header("request-id", request_id);

    AnthropicMessagesRequest request;
    PromptFingerprint prompt_fingerprint;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        const RequestJson body = parse_json_body(req);
        request                = parse_anthropic_messages_request(body, limits);
        prompt_fingerprint     = make_prompt_fingerprint(body);
    } catch (const ApiException& exception) {
        write_anthropic_error(res, exception.error(), request_id);
        return;
    } catch (const std::exception& exception) {
        operational_log_.http_failure(
            "anthropic_messages",
            make_internal_request_failure(RequestFailurePhase::Http, exception.what()), request_id);
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogMetadata metadata{.model                  = request.model,
                                      .stream                 = request.stream,
                                      .output_tokens_explicit = request.output_tokens_explicit};
    PreparedRequest prepared;
    try {
        // LOCAL (prefill-heartbeat): stream prompt progress on streaming requests so a long
        // root prefill emits bytes while it runs. DSH's dsh-llm-pi-ai kills a stream after
        // 300 s idle and the engine is silent for 10-20 min of prefill otherwise; that exact
        // pairing is what killed every first turn. The OpenAI path gates this on a client
        // return_progress flag that ZCode never sends, so the Anthropic path gates on stream
        // alone. Bytes are written as SSE comments (see on_progress below), which every SSE
        // parser ignores, so no client contract changes.
        const ninfer::GenerationObservationOptions observation{
            .phase_timings   = true,
            .prompt_progress = request.stream,
        };
        prepared = service_->prepare(request.generation,
                                     request.stream ? GenerationConsumerMode::Streaming
                                                    : GenerationConsumerMode::Aggregate,
                                     observation, [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        const ApiError error = normalize_anthropic_error(exception.error());
        record_request_rejected(make_request_rejection_log_context(
            req_id, "anthropic_messages", request.generation, metadata, error));
        write_anthropic_error(res, error, request_id);
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "anthropic_messages", request.generation, metadata, error));
        write_anthropic_error(res, error, request_id);
        return;
    }

    const AnthropicResponseIdentity identity =
        make_anthropic_response_identity(request_id, request.model);
    const int input_tokens = prepared.prompt_tokens;

    auto lifecycle = begin_request(make_request_log_context(
        req_id, "anthropic_messages", request.generation, metadata, prepared,
        std::move(prompt_fingerprint)));

    if (!request.stream) {
        GenerationOutcome outcome;
        try {
            outcome = service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            const ApiError error = normalize_anthropic_error(exception.error());
            lifecycle->failure(make_generation_request_failure(error));
            write_anthropic_error(res, error, request_id);
            return;
        } catch (const std::exception& exception) {
            lifecycle->failure(
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
            ApiError error;
            error.status  = 500;
            error.message = exception.what();
            write_anthropic_error(res, error, request_id);
            return;
        }
        lifecycle->done(outcome);
        try {
            set_owned_json_content(res, make_anthropic_messages_response(identity, outcome),
                                   prepared.lifetime);
        } catch (const ApiException& exception) {
            const ApiError error = normalize_anthropic_error(exception.error());
            lifecycle->response_failure(
                make_request_failure(RequestFailurePhase::ResponseRender, error));
            write_anthropic_error(res, error, request_id);
        } catch (const std::exception& exception) {
            lifecycle->response_failure(make_internal_request_failure(
                RequestFailurePhase::ResponseRender, exception.what()));
            ApiError error;
            error.status  = 500;
            error.message = exception.what();
            write_anthropic_error(res, error, request_id);
        }
        return;
    }

    try {
        auto stream  = std::make_shared<HttpGenerationStream>(std::move(prepared));
        auto encoder = std::make_shared<AnthropicMessagesStream>(identity, input_tokens);

        prepare_sse_response(res);
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, stream, encoder, lifecycle](std::size_t, httplib::DataSink& sink) -> bool {
                if (stream->started.exchange(true, std::memory_order_acq_rel)) {
                    sink.done();
                    return true;
                }
                SseTransport transport(sink, stream->cancelled);
                const auto send_error = [&](const ApiError& error) {
                    try {
                        if (!encoder->started()) {
                            render_and_write(transport, [&] { return encoder->start(); });
                        }
                        render_and_write(transport, [&] { return encoder->error(error); });
                        sink.done();
                        return true;
                    } catch (const ClientDisconnected&) {
                        lifecycle->response_failure(
                            make_client_disconnected_failure(RequestFailurePhase::Transport));
                        return false;
                    } catch (const ResponseRenderFailure& exception) {
                        lifecycle->response_failure(make_internal_request_failure(
                            RequestFailurePhase::ResponseRender, exception.what()));
                        return false;
                    }
                };

                GenerationOutcome outcome;
                try {
                    StreamSink output;
                    output.on_start = [&](const ninfer::GenerationStart& start) {
                        render_and_write(transport, [&] { return encoder->start(start); });
                    };
                    output.on_reasoning = [&](const std::string& text) {
                        render_and_write(transport, [&] { return encoder->reasoning_delta(text); });
                    };
                    output.on_content = [&](const std::string& text) {
                        render_and_write(transport, [&] { return encoder->content_delta(text); });
                    };
                    output.is_cancelled = [&] { return transport.poll(); };
                    // LOCAL (prefill-heartbeat): the 5 s poll() heartbeat only fires when the
                    // engine asks, and dense prefill units can go minutes without asking. Prompt
                    // progress is published per prefill chunk, so hook it to an SSE comment: all
                    // parsers ignore ':' lines, but every idle timer sees the bytes. Written
                    // straight to the transport, bypassing the event encoder, because the
                    // encoder's state machine throws on out-of-order use and a keep-alive must
                    // never be able to fail the request. Disconnects propagate as
                    // ClientDisconnected through the same catch as every other transport write.
                    output.on_progress = [&](const ninfer::PromptProgress&) {
                        transport.write(SseTransport::kHeartbeatComment);
                    };

                    outcome = service_->run(stream->prepared, &output);
                } catch (const ClientDisconnected&) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                } catch (const ResponseRenderFailure& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.message = exception.what();
                    return send_error(error);
                } catch (const ApiException& exception) {
                    const ApiError error = normalize_anthropic_error(exception.error());
                    lifecycle->failure(make_generation_request_failure(error));
                    return send_error(error);
                } catch (const std::exception& exception) {
                    lifecycle->failure(make_internal_request_failure(
                        RequestFailurePhase::Generation, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.message = exception.what();
                    return send_error(error);
                }

                lifecycle->done(outcome);
                std::vector<std::string> terminal;
                try {
                    terminal = encoder->finish(outcome);
                } catch (const std::exception& exception) {
                    lifecycle->response_failure(make_internal_request_failure(
                        RequestFailurePhase::ResponseRender, exception.what()));
                    ApiError error;
                    error.status  = 500;
                    error.message = exception.what();
                    return send_error(error);
                }
                try {
                    transport.write(terminal);
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) {
                    lifecycle->response_failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                    return false;
                }
            },
            [stream, lifecycle](bool successful) {
                stream->cancelled.store(true, std::memory_order_release);
                if (!successful || !stream->started.load(std::memory_order_acquire)) {
                    lifecycle->failure(
                        make_client_disconnected_failure(RequestFailurePhase::Transport));
                }
            });
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::ResponseRender, exception.what()));
        ApiError error;
        error.status  = 500;
        error.message = exception.what();
        write_anthropic_error(res, error, request_id);
    }
}

} // namespace ninfer::serve
