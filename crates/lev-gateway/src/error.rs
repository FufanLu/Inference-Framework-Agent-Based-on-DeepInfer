//! API errors, rendered in the error format of the protocol that failed.

use axum::http::StatusCode;
use deepseek_recipe::request::ConversionError;
use serde_json::json;

/// What kind of failure an [`ApiError`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidRequest,
    Authentication,
    PermissionDenied,
    NotFound,
    RateLimited,
    /// No worker could take the request.
    Overloaded,
    /// A worker failed while serving the request.
    Upstream,
    Internal,
}

impl ErrorKind {
    /// The OpenAI `error.type`.
    pub fn openai_type(self) -> &'static str {
        match self {
            ErrorKind::InvalidRequest | ErrorKind::NotFound => "invalid_request_error",
            ErrorKind::Authentication => "authentication_error",
            ErrorKind::PermissionDenied => "permission_error",
            ErrorKind::RateLimited => "rate_limit_error",
            ErrorKind::Overloaded => "overloaded_error",
            ErrorKind::Upstream | ErrorKind::Internal => "server_error",
        }
    }

    /// The Anthropic `error.type`.
    pub fn anthropic_type(self) -> &'static str {
        match self {
            ErrorKind::InvalidRequest => "invalid_request_error",
            ErrorKind::Authentication => "authentication_error",
            ErrorKind::PermissionDenied => "permission_error",
            ErrorKind::NotFound => "not_found_error",
            ErrorKind::RateLimited => "rate_limit_error",
            ErrorKind::Overloaded => "overloaded_error",
            ErrorKind::Upstream | ErrorKind::Internal => "api_error",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub kind: ErrorKind,
    /// Machine-readable code, e.g. `model_not_found`.
    pub code: &'static str,
    pub message: String,
    /// For 429: when the client may retry.
    pub retry_after: Option<std::time::Duration>,
}

impl ApiError {
    pub fn new(
        status: StatusCode,
        kind: ErrorKind,
        code: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self {
            status,
            kind,
            code,
            message: message.into(),
            retry_after: None,
        }
    }

    pub fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            ErrorKind::InvalidRequest,
            code,
            message,
        )
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorKind::Internal,
            "internal_error",
            message,
        )
    }

    pub fn upstream(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_GATEWAY,
            ErrorKind::Upstream,
            "upstream_error",
            message,
        )
    }

    pub fn overloaded(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            ErrorKind::Overloaded,
            "no_available_worker",
            message,
        )
    }

    /// The OpenAI error body: `{"error": {"message", "type", "param", "code"}}`.
    pub fn openai_body(&self) -> serde_json::Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.kind.openai_type(),
                "param": null,
                "code": self.code,
            }
        })
    }

    /// The Anthropic error body: `{"type": "error", "error": {"type", "message"}}`.
    pub fn anthropic_body(&self) -> serde_json::Value {
        json!({
            "type": "error",
            "error": {
                "type": self.kind.anthropic_type(),
                "message": self.message,
            }
        })
    }
}

impl From<ConversionError> for ApiError {
    fn from(error: ConversionError) -> Self {
        let status =
            StatusCode::from_u16(error.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        match error {
            ConversionError::BadRequest { detail } => {
                let code = if detail.contains("maximum context length") {
                    "context_length_exceeded"
                } else {
                    "invalid_request_error"
                };
                Self::new(status, ErrorKind::InvalidRequest, code, detail)
            }
            ConversionError::Internal { detail } => {
                Self::new(status, ErrorKind::Internal, "internal_error", detail)
            }
        }
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({}): {}", self.status, self.code, self.message)
    }
}

impl std::error::Error for ApiError {}
