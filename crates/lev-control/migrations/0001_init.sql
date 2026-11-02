-- Leviathan control plane schema. Idempotent: applied at startup.
CREATE TABLE IF NOT EXISTS tenants (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    rpm                 BIGINT,
    tpm                 BIGINT,
    monthly_token_quota BIGINT,
    disabled            BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS api_keys (
    id           UUID PRIMARY KEY,
    tenant_id    TEXT NOT NULL REFERENCES tenants (id),
    name         TEXT NOT NULL,
    prefix       TEXT NOT NULL,
    sha256       BYTEA NOT NULL UNIQUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS usage_ledger (
    id                BIGSERIAL PRIMARY KEY,
    tenant_id         TEXT NOT NULL,
    request_id        TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt_tokens     BIGINT NOT NULL,
    cached_tokens     BIGINT NOT NULL,
    completion_tokens BIGINT NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS usage_ledger_tenant_time ON usage_ledger (tenant_id, created_at);
