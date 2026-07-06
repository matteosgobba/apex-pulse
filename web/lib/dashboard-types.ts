export type DashboardStatus = "complete" | "partial" | "empty" | "invalid";

export type LifecycleState =
  | "no_event_available"
  | "practice_in_progress"
  | "ready_to_forecast"
  | "forecast_available"
  | "awaiting_qualifying_targets"
  | "settled"
  | "blocked"
  | "legacy_descriptive_only";

export interface DashboardEnvelope<TData extends Record<string, unknown>> {
  schema_version: "1.0";
  artifact_type: string;
  generated_at_utc: string;
  source_artifacts: string[];
  source_fingerprints: Record<string, SourceFingerprint>;
  status: DashboardStatus;
  data: TData;
}

export interface SourceFingerprint {
  available: boolean;
  required: boolean;
  sha256?: string | null;
  reason?: string | null;
}

export interface AvailabilityValue<TValue> {
  available: boolean;
  reason: string;
  value: TValue | null;
}

export interface EventIdentity {
  season: number | null;
  event: string | null;
  event_slug: string | null;
  event_order: number | null;
}

export interface LifecycleInfo {
  state: LifecycleState;
  display_label: string;
  reason: string;
}

export interface MonitoringProtocol {
  checkpoint?: string | null;
  monitor_season?: number | null;
  policy_recommendation?: string | null;
  protocol_fingerprint?: string | null;
  protocol_name?: string | null;
  protocol_version?: string | null;
  train_seasons?: number[] | null;
}

export interface RegistryLineage {
  eligible_for_valid_prospective_evidence?: boolean | null;
  event_order?: number | null;
  event_order_lineage_status?: string | null;
  legacy_noncanonical?: boolean | null;
  reconciliation_action?: string | null;
  reconciliation_reason?: string | null;
}

export interface PreflightStatus {
  available?: boolean | null;
  blocking_check_count?: number | null;
  forecast_allowed?: boolean | null;
  next_required_command?: string | null;
  preflight_run_id?: string | null;
  runbook_path?: string | null;
  status?: string | null;
  warning_check_count?: number | null;
}

export interface ForecastStatus {
  available?: boolean | null;
  checkpoint?: string | null;
  forecast_created_at_utc?: string | null;
  forecasted_driver_count?: number | null;
}

export interface SettlementStatus {
  available?: boolean | null;
  excluded_driver_count?: number | null;
  scored_driver_count?: number | null;
  settled_at_utc?: string | null;
  settlement_valid?: boolean | null;
}

export interface LegacyStatus {
  display_label?: string | null;
  eligible_for_valid_prospective_evidence?: boolean | null;
  legacy_noncanonical?: boolean | null;
  reason?: string | null;
}

export interface FreshnessInfo {
  absent_values_are_unavailable_not_zero?: boolean | null;
  dashboard_source_generated_at_utc?: Record<string, string | null>;
}

export interface SummaryKpis {
  actual_pole_driver?: string | null;
  forecast_checkpoint?: string | null;
  forecasted_driver_count?: number | null;
  interval_availability_rate?: number | null;
  predicted_pole_driver?: string | null;
  settlement_mae_gap_sec?: number | null;
}

export interface CurrentEventData extends Record<string, unknown> {
  event_identity?: EventIdentity | null;
  lifecycle?: LifecycleInfo | null;
  freshness?: FreshnessInfo | null;
  monitoring_protocol?: MonitoringProtocol | null;
  registry_lineage?: RegistryLineage | null;
  preflight?: PreflightStatus | null;
  forecast_status?: ForecastStatus | null;
  settlement_status?: SettlementStatus | null;
  legacy_status?: LegacyStatus | null;
  summary_kpis?: SummaryKpis | null;
}

export interface SessionStatus {
  session: "FP1" | "FP2" | "FP3" | "Q" | string;
  available: boolean;
  status: string;
  artifact_available: boolean;
  last_known_timestamp: string | null;
  reason: string | null;
}

export interface MonitoringReadiness {
  chronological_order_status?: string | null;
  forecastable_event_count?: number | null;
  settleable_event_count?: number | null;
  status?: string | null;
  target_isolation_status?: string | null;
}

export interface PracticeStatusData extends Record<string, unknown> {
  event_identity?: EventIdentity | null;
  lifecycle_state?: LifecycleState | null;
  sessions?: SessionStatus[];
  monitoring_readiness?: MonitoringReadiness | null;
  preflight?: PreflightStatus | null;
  notes?: string[];
}

export interface ManifestData extends Record<string, unknown> {
  current_event_reference?: {
    artifact?: string | null;
    event_identity?: EventIdentity | null;
    lifecycle_state?: LifecycleState | null;
  } | null;
  available_pages?: Record<string, AvailabilityValue<string>>;
  event_count?: number | null;
  eligible_prospective_event_count?: number | null;
  legacy_descriptive_event_count?: number | null;
  dashboard_contract_capabilities?: Record<string, boolean>;
}

export interface HealthResponse {
  status: "ok";
  service: string;
  api_version: "v1";
  dashboard_artifact_status:
    | "complete"
    | "partial"
    | "empty"
    | "invalid"
    | "unavailable";
}

export interface DashboardApiErrorPayload {
  detail: {
    code: string;
    message: string;
    artifact_type?: string;
  };
}

export type CurrentEventEnvelope = DashboardEnvelope<CurrentEventData>;
export type PracticeStatusEnvelope = DashboardEnvelope<PracticeStatusData>;
export type ManifestEnvelope = DashboardEnvelope<ManifestData>;

export interface CurrentEventPageData {
  health: HealthResponse | null;
  manifest: ManifestEnvelope | null;
  currentEvent: CurrentEventEnvelope | null;
  practiceStatus: PracticeStatusEnvelope | null;
  error: SafeDashboardError | null;
}

export interface SafeDashboardError {
  code: string;
  message: string;
  status?: number;
  artifactType?: string;
}
