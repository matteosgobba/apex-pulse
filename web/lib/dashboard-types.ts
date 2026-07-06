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

export type DashboardRows<TRow> = TRow[] | AvailabilityValue<TRow[]>;

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
  forecast_only_driver_count?: number | null;
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

export interface ForecastMethodIdentity extends Record<string, unknown> {
  family?: string | null;
  feature_group?: string | null;
  model_name?: string | null;
  temporal_weighting_policy?: string | null;
}

export interface ForecastProvenance extends Record<string, unknown> {
  forecast_id?: string | null;
  forecast_integrity_status?: string | null;
  live_policy_selected?: boolean | null;
  prediction_role?: string | null;
  source_lineage_valid?: boolean | null;
}

export interface ForecastMetadata extends Record<string, unknown> {
  candidate_or_policy_identity?: ForecastMethodIdentity | null;
  checkpoint?: string | null;
  forecast_timestamp?: string | null;
  prediction_target?: string | null;
  preflight_run_id?: string | null;
  preflight_status?: string | null;
  protocol_fingerprint?: string | null;
  protocol_name?: string | null;
  uncertainty_method?: string | null;
}

export interface ForecastLeaderboardRow extends Record<string, unknown> {
  predicted_position?: number | null;
  driver?: string | null;
  driver_code?: string | null;
  team?: string | null;
  team_key?: string | null;
  predicted_gap_to_pole_sec?: number | null;
  interval_lower_sec?: number | null;
  interval_upper_sec?: number | null;
  interval_available?: boolean | null;
  selected_method?: ForecastMethodIdentity | null;
  provenance?: ForecastProvenance | null;
  actual_position?: number | null;
  actual_gap_to_pole_sec?: number | null;
  absolute_gap_error_sec?: number | null;
  forecast_eligible_driver?: boolean | null;
  forecast_only_driver?: boolean | null;
  forecast_only_reason?: string | null;
  settlement_evaluable_driver?: boolean | null;
}

export interface ForecastSummary extends Record<string, unknown> {
  checkpoint?: string | null;
  forecasted_driver_count?: number | null;
  forecast_only_driver_count?: number | null;
  interval_availability_rate?: number | null;
  predicted_pole_driver?: string | null;
  settlement_evaluable_driver_count?: number | null;
}

export interface ForecastData extends Record<string, unknown> {
  event_identity?: EventIdentity | null;
  lifecycle_state?: LifecycleState | null;
  forecast_metadata?: ForecastMetadata | null;
  leaderboard?: DashboardRows<ForecastLeaderboardRow>;
  qualifying_eligible_forecast_rows?: DashboardRows<ForecastLeaderboardRow>;
  forecast_only_rows?: DashboardRows<ForecastLeaderboardRow>;
  settlement_evaluable_rows?: DashboardRows<ForecastLeaderboardRow>;
  summary?: ForecastSummary | null;
}

export interface SettlementMetadata extends Record<string, unknown> {
  checkpoint?: string | null;
  protocol_fingerprint?: string | null;
  protocol_name?: string | null;
  settled_at_utc?: string | null;
  settlement_valid?: boolean | null;
}

export interface SettlementSummaryMetrics extends Record<string, unknown> {
  actual_pole_driver?: string | null;
  driver_count?: number | null;
  mae_gap_sec?: number | null;
  mean_absolute_position_error?: number | null;
  median_absolute_gap_error_sec?: number | null;
  rmse_gap_sec?: number | null;
  scored_driver_count?: number | null;
  settlement_evaluable_driver_count?: number | null;
  excluded_driver_count?: number | null;
  top_10_agreement?: number | null;
  top_3_agreement?: number | null;
  top_5_agreement?: number | null;
}

export interface SettlementDriverComparisonRow extends Record<string, unknown> {
  absolute_gap_error_sec?: number | null;
  absolute_position_error?: number | null;
  actual_gap_to_pole_sec?: number | null;
  actual_position?: number | null;
  driver?: string | null;
  driver_code?: string | null;
  included_in_metrics?: boolean | null;
  predicted_gap_to_pole_sec?: number | null;
  predicted_position?: number | null;
  settlement_evaluable?: boolean | null;
  settlement_exclusion_reason?: string | null;
  team?: string | null;
  team_key?: string | null;
}

export interface SettlementData extends Record<string, unknown> {
  driver_comparison?: DashboardRows<SettlementDriverComparisonRow>;
  event_identity?: EventIdentity | null;
  interval_diagnostics?: AvailabilityValue<Record<string, unknown>>;
  lifecycle_state?: LifecycleState | null;
  settlement_metadata?: SettlementMetadata | null;
  settlement_evaluable_rows?: DashboardRows<SettlementDriverComparisonRow>;
  forecast_only_rows?: DashboardRows<SettlementDriverComparisonRow>;
  summary_metrics?: SettlementSummaryMetrics | null;
}

export type HistoricalMonitoringAggregateMetrics = AvailabilityValue<Record<string, unknown>>;

export interface MonitoringHistoryEvent extends Record<string, unknown> {
  event_identity?: EventIdentity | null;
  lifecycle_state?: LifecycleState | null;
  forecast_available?: boolean | null;
  settlement_available?: boolean | null;
  forecast_checkpoint?: string | null;
  mae_gap_sec?: number | null;
  interval_availability_rate?: number | null;
  eligible_for_valid_prospective_evidence?: boolean | null;
}

export interface ValidProspectiveMonitoring extends Record<string, unknown> {
  aggregate_metrics?: HistoricalMonitoringAggregateMetrics | null;
  event_count?: number | null;
  events?: MonitoringHistoryEvent[];
  forecasted_event_count?: number | null;
  settled_event_count?: number | null;
}

export interface LegacyDescriptiveRecord extends Record<string, unknown> {
  descriptive_metrics?: {
    available?: boolean | null;
    excluded_rows?: number | null;
    forecast_rows?: number | null;
    mae_gap_sec?: number | null;
    scored_rows?: number | null;
  } | null;
  eligible_for_valid_prospective_evidence?: boolean | null;
  event_identity?: EventIdentity | null;
  exclusion_reason?: string | null;
  legacy_noncanonical?: boolean | null;
  lifecycle_state?: LifecycleState | null;
}

export interface BacktestContext extends Record<string, unknown> {
  available?: boolean | null;
  champion_selection_mode?: string | null;
  n_events?: number | null;
  n_folds_successful?: number | null;
  preferred_backtest_strategy?: string | null;
}

export interface HistoricalMonitoringData extends Record<string, unknown> {
  valid_prospective_monitoring?: ValidProspectiveMonitoring | null;
  legacy_descriptive_records?: LegacyDescriptiveRecord[];
  backtest_context?: BacktestContext | null;
}

export interface ModelSummaryData extends Record<string, unknown> {
  backtest_summary?: Record<string, unknown> | null;
  current_policy_summary?: Record<string, unknown> | null;
  limitations?: string[];
  model_status?: Record<string, unknown> | null;
  supported_forecast_outputs?: Record<string, boolean>;
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
export type ForecastEnvelope = DashboardEnvelope<ForecastData>;
export type SettlementEnvelope = DashboardEnvelope<SettlementData>;
export type HistoricalMonitoringEnvelope = DashboardEnvelope<HistoricalMonitoringData>;
export type ModelSummaryEnvelope = DashboardEnvelope<ModelSummaryData>;

export interface CurrentEventPageData {
  health: HealthResponse | null;
  manifest: ManifestEnvelope | null;
  currentEvent: CurrentEventEnvelope | null;
  practiceStatus: PracticeStatusEnvelope | null;
  forecast: ForecastEnvelope | null;
  settlement: SettlementEnvelope | null;
  historicalMonitoring: HistoricalMonitoringEnvelope | null;
  error: SafeDashboardError | null;
}

export interface ForecastPageData {
  health: HealthResponse | null;
  currentEvent: CurrentEventEnvelope | null;
  forecast: ForecastEnvelope | null;
  error: SafeDashboardError | null;
}

export interface PracticePageData {
  health: HealthResponse | null;
  currentEvent: CurrentEventEnvelope | null;
  practiceStatus: PracticeStatusEnvelope | null;
  error: SafeDashboardError | null;
}

export interface SettlementPageData {
  health: HealthResponse | null;
  currentEvent: CurrentEventEnvelope | null;
  forecast: ForecastEnvelope | null;
  settlement: SettlementEnvelope | null;
  error: SafeDashboardError | null;
}

export interface MonitoringHistoryPageData {
  health: HealthResponse | null;
  historicalMonitoring: HistoricalMonitoringEnvelope | null;
  modelSummary: ModelSummaryEnvelope | null;
  error: SafeDashboardError | null;
}

export interface SafeDashboardError {
  code: string;
  message: string;
  status?: number;
  artifactType?: string;
}
