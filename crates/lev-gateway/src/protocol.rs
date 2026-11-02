//! Per-protocol identity: response IDs and error formats.

use axum::Json;
use axum::response::{IntoResponse, Response};
use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
use serde_json::json;
use uuid::Uuid;

use crate::error::ApiError;

/// An error response with a `Retry-After` header when the error has one.
fn with_retry_after(error: &ApiError, response: Response) -> Response {
    let mut response = response;
    if let Some(after) = error.retry_after {
        let seconds = after.as_secs_f64().ceil().max(1.0) as u64;
        if let Ok(value) = axum::http::HeaderValue::from_str(&seconds.to_string()) {
            response
                .headers_mut()
                .insert(axum::http::header::RETRY_AFTER, value);
        }
    }
    response
}

/// What differs between the three API protocols beyond recipe's conversion.
pub trait ProtocolInfo {
    /// Short name used in metrics, logs and the audit log.
    const NAME: &'static str;

    /// The response ID clients see for request `id`.
    fn response_id(id: &Uuid) -> String;

    /// A complete error response.
    fn error_response(error: &ApiError) -> Response;

    /// The SSE frame that reports a failure after streaming has started.
    fn stream_error_frame(error: &ApiError) -> String;
}

impl ProtocolInfo for ChatCompletionRequest {
    const NAME: &'static str = "chat_completions";

    fn response_id(id: &Uuid) -> String {
        format!("chatcmpl-{}", id.simple())
    }

    fn error_response(error: &ApiError) -> Response {
        with_retry_after(
            error,
            (error.status, Json(error.openai_body())).into_response(),
        )
    }

    fn stream_error_frame(error: &ApiError) -> String {
        format!("data: {}\n\n", error.openai_body())
    }
}

impl ProtocolInfo for ResponsesRequest {
    const NAME: &'static str = "responses";

    fn response_id(id: &Uuid) -> String {
        format!("resp_{}", id.simple())
    }

    fn error_response(error: &ApiError) -> Response {
        with_retry_after(
            error,
            (error.status, Json(error.openai_body())).into_response(),
        )
    }

    fn stream_error_frame(error: &ApiError) -> String {
        let data = json!({
            "type": "error",
            "code": error.code,
            "message": error.message,
            "param": null,
        });
        format!("event: error\ndata: {data}\n\n")
    }
}

impl ProtocolInfo for MessagesRequest {
    const NAME: &'static str = "messages";

    fn response_id(id: &Uuid) -> String {
        format!("msg_{}", id.simple())
    }

    fn error_response(error: &ApiError) -> Response {
        with_retry_after(
            error,
            (error.status, Json(error.anthropic_body())).into_response(),
        )
    }

    fn stream_error_frame(error: &ApiError) -> String {
        format!("event: error\ndata: {}\n\n", error.anthropic_body())
    }
}
