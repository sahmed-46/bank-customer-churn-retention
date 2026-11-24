-- Supabase/PostgreSQL schema for the bank churn scoring pipeline.
-- Keep personally identifiable information out of dashboard-facing exports.

create extension if not exists pgcrypto;

create table if not exists public.customers (
    customer_id uuid primary key default gen_random_uuid(),
    external_customer_id text not null unique,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.customer_snapshots (
    snapshot_id bigint generated always as identity primary key,
    customer_id uuid not null references public.customers(customer_id),
    observed_at timestamptz not null default now(),
    gender text,
    marital_status text,
    number_of_dependents integer,
    occupation text,
    income numeric(14,2),
    education_level text,
    customer_tenure numeric(8,2),
    customer_segment text,
    preferred_communication_channel text,
    credit_score numeric(6,2),
    credit_history_length numeric(8,2),
    outstanding_loans numeric(14,2),
    balance numeric(14,2),
    num_of_products integer,
    num_complaints integer,
    raw_payload jsonb,
    data_quality_flags jsonb
);

create table if not exists public.scoring_runs (
    run_id uuid primary key default gen_random_uuid(),
    model_name text not null,
    model_version text not null,
    source_watermark timestamptz,
    started_at timestamptz not null default now(),
    completed_at timestamptz,
    rows_read integer,
    rows_scored integer,
    status text not null default 'running' check (status in ('running', 'completed', 'failed')),
    error_message text
);

create table if not exists public.customer_predictions (
    prediction_id bigint generated always as identity primary key,
    customer_id uuid not null references public.customers(customer_id),
    snapshot_id bigint not null references public.customer_snapshots(snapshot_id),
    run_id uuid not null references public.scoring_runs(run_id),
    churn_score numeric(8,6) not null check (churn_score between 0 and 1),
    clv_est numeric(14,2),
    expected_lost_value numeric(14,2),
    segment text,
    recommended_action text,
    predicted_at timestamptz not null default now(),
    model_version text not null,
    unique (customer_id, snapshot_id, model_version)
);

create table if not exists public.retention_actions (
    retention_action_id bigint generated always as identity primary key,
    prediction_id bigint not null references public.customer_predictions(prediction_id),
    action_type text not null,
    action_status text not null default 'recommended',
    contacted_at timestamptz,
    outcome text,
    outcome_at timestamptz,
    notes text
);

create index if not exists idx_snapshots_customer_time
    on public.customer_snapshots (customer_id, observed_at desc);
create index if not exists idx_predictions_customer_time
    on public.customer_predictions (customer_id, predicted_at desc);
create index if not exists idx_predictions_expected_loss
    on public.customer_predictions (expected_lost_value desc);
create index if not exists idx_predictions_score
    on public.customer_predictions (churn_score desc);

create or replace view public.dashboard_customer_priority as
select distinct on (p.customer_id)
    c.external_customer_id,
    s.customer_segment,
    p.churn_score,
    p.clv_est,
    p.expected_lost_value,
    p.segment,
    p.recommended_action,
    p.predicted_at,
    p.model_version
from public.customer_predictions p
join public.customers c on c.customer_id = p.customer_id
join public.customer_snapshots s on s.snapshot_id = p.snapshot_id
order by p.customer_id, p.predicted_at desc;

-- The dashboard is currently published as static files, so the scoring worker
-- uses a database connection string. Do not expose that credential to clients.
