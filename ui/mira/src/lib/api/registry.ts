import { fetchJson, patchJson, postJson } from "./http"

// Kory Source work and decision registry (fork-owned; see FORK.md).

export interface RegistryAnchor {
  repo: string
  path: string
  qualified_name: string
  symbol_kind: string
  anchor_id: string
  resolution_status: string
}

export interface RegistryRelationship {
  id: string
  source_id: string
  target_id: string
  type: string
  status: string
  relation: string
  new_clause: string
  existing_clause: string
  session?: string
  answer?: string | null
  answered_by?: string
}

export interface RegistryObject {
  id: string
  kind: string
  title: string
  summary: string
  body: string
  scope: string
  repo: string
  status: string
  status_reason: string
  realization: string
  confidence: number
  author_kind: string
  author: string
  model: string
  decided_by: string
  session: string
  branch: string
  pr: string
  merge_commit: string
  source_url: string
  basis: string
  conflict_check: string
  goal: string
  work_scope: string
  plan: string
  owner: string
  delivery_state: string
  tested_revision: string
  method: string
  result: string
  artifact_url: string
  source_system: string
  source_id: string
  sync_status: string
  created_at: number
  restored_at: number
  proposal: boolean
  unresolved?: string[]
  conflicts?: RegistryRelationship[]
  member_count?: number
  requirement_id?: string
}

export interface RegistryDetail extends RegistryObject {
  anchors: RegistryAnchor[]
  concepts: string[]
  relationships: RegistryRelationship[]
}

export interface AnchorGroup extends RegistryAnchor {
  objects: { id: string; title: string; status: string; kind: string }[]
}

export interface WorkView {
  work: RegistryObject
  requirements: RegistryObject[]
  decisions: RegistryObject[]
  children: RegistryObject[]
  dependencies: {
    parent: RegistryObject | null
    blocks: RegistryObject[]
    blocked_by: RegistryObject[]
    relates: RegistryObject[]
  }
  evidence: RegistryObject[]
  delivery_state: string
  implementation: {
    realization_counts: Record<string, number>
    prs: string[]
    merge_commits: string[]
  }
  verification: {
    requirement: RegistryObject
    evidence: RegistryObject[]
    latest_result: string
    latest_revision: string
  }[]
}

export interface NewWork {
  title: string
  goal: string
  work_scope: string
  plan: string
  owner: string
}

// implementation contract: provenance, history and cited questions. Only approved
// complete-corpus readers get data; everyone else gets the same 404.

export interface Attribution {
  role: string
  actor_kind: string
  original_identity: string | null
  original_provider: string | null
  original_bot: string | null
  original_model: string | null
  original_client: string | null
  original_environment: string | null
  normalized_identity: string | null
  provider_family: string | null
  model_family: string | null
  model_version: string | null
  client_interface: string | null
  execution_environment: string | null
  source_session_id: string | null
  source_run_id: string | null
  occurred_at: number | null
  evidence_id: string
  reason: string
  field_states?: Record<string, { state: string; reason: string; pointer: string }>
  unknown_fields?: string[]
  inferred_fields?: string[]
}

export interface EvidenceQuote {
  id?: string
  evidence_id: string
  evidence_role?: string
  role?: string
  quote: string
  quote_start?: number
  quote_end?: number
  section?: string
  stated_or_inferred?: string
}

export interface SourceRevision {
  evidence_id: string
  source_system?: string
  source_type: string
  source_url: string
  source_version: string
  source_created_at: number
  commit_sha: string
  base_sha: string
  head_sha: string
  content_sha256: string
  import_run_id: string
  original_text?: string
}

export interface AnchorEvidence {
  repository: string
  observed_commit: string
  path: string
  qualified_name: string
  heading: string
  line_start: number
  line_end: number
  resolution_status: string
  resolved_path: string
}

export interface DecisionEvidence {
  object_id: string
  legacy: boolean
  legacy_projection: Record<string, { value: unknown; state: string; reason: string }> | null
  quotes: EvidenceQuote[]
  revisions: SourceRevision[]
  attributions: Attribution[]
  versions: { revision: number; claim_text: string; approval_status: string; authority_tier: string; authority_reasons: string[] }[]
  anchors: AnchorEvidence[]
}

export interface HistoryEdge {
  id: string
  type: string
  source_id: string
  target_id: string
  status: string
  relation: string
  ancestry: boolean
  symmetric: boolean
}

export interface HistoryGraph {
  start: string[]
  nodes: Record<string, { id: string; title: string; status: string; authority_tier: string; source_url: string }>
  edges: HistoryEdge[]
  history_order: string[]
  heads: string[]
  hidden_links: number
  truncated: string[]
  complete: boolean
}

export interface Citation {
  quotes: EvidenceQuote[]
  revisions: SourceRevision[]
  attribution: Attribution[]
  provenance: Attribution[]
  legacy: boolean
  anchors: AnchorEvidence[]
}

export interface QuestionAnswer {
  kind: "pr_decisions" | "decision_history" | "needs_context"
  verdict: string
  needs?: string
  repo?: string
  pr?: { number: number; url?: string; title?: string; base_sha?: string; head_sha?: string; merge_base_sha?: string }
  findings?: {
    decision: { id: string; title: string; status: string; authority_tier: string; source_url: string }
    verdict: string
    why_considered: string
    reasons: string[]
  }[]
  narrative?: { text: string; evidence: string[]; verdict?: string }[]
  citations?: Record<string, Citation>
  limits?: string[]
  graph?: HistoryGraph
}

export interface ImportRun {
  run_id: string
  source_system: string
  stream: string
  status: string
  importer: string
  importer_version: string
  coverage: Record<string, number | null>
  started_at: number
  finished_at: number
}

const base = "/api/registry"
const id = (value: string) => encodeURIComponent(value)

export const registryApi = {
  queue: () => fetchJson<RegistryObject[]>(`${base}/queue`),
  expired: () => fetchJson<RegistryObject[]>(`${base}/expired`),
  concepts: () => fetchJson<RegistryObject[]>(`${base}/concepts`),
  anchors: () => fetchJson<AnchorGroup[]>(`${base}/anchors`),
  conceptMembers: (concept: string) =>
    fetchJson<RegistryObject[]>(`${base}/objects?concept=${id(concept)}`),
  detail: (objectId: string) => fetchJson<RegistryDetail>(`${base}/objects/${id(objectId)}`),
  chain: (objectId: string) => fetchJson<RegistryObject[]>(`${base}/objects/${id(objectId)}/chain`),
  approve: (objectId: string) => postJson<RegistryObject>(`${base}/objects/${id(objectId)}/approve`, {}),
  reject: (objectId: string, reason = "") =>
    postJson<RegistryObject>(`${base}/objects/${id(objectId)}/reject`, { reason }),
  restore: (objectId: string) => postJson<RegistryObject>(`${base}/objects/${id(objectId)}/restore`, {}),
  resolve: (objectId: string, choice: "keep" | "supersede" | "defer") =>
    postJson<RegistryObject>(`${base}/objects/${id(objectId)}/resolve`, { choice }),
  sessionConflicts: (session: string) =>
    fetchJson<{ session: string; unresolved: boolean; conflicts: RegistryRelationship[] }>(
      `${base}/session-conflicts?session=${id(session)}`,
    ),
  conflictPrompts: () =>
    fetchJson<{
      prompts: number
      answers: { keep: number; supersede: number; defer: number }
      by_session: { session: string; prompts: number }[]
    }>(`${base}/conflict-prompts`),
  works: () => fetchJson<RegistryObject[]>(`${base}/work`),
  work: (workId: string) => fetchJson<WorkView>(`${base}/work/${id(workId)}`),
  createWork: (body: NewWork) => postJson<RegistryObject>(`${base}/work`, { ...body, scope: "org" }),
  setDeliveryState: (workId: string, delivery_state: string) =>
    patchJson<RegistryObject>(`${base}/work/${id(workId)}`, { delivery_state }),
  linkWork: (workId: string, target_id: string, type: string) =>
    postJson<unknown>(`${base}/work/${id(workId)}/links`, { target_id, type }),
  evidence: (objectId: string) =>
    fetchJson<DecisionEvidence>(`${base}/objects/${id(objectId)}/evidence`),
  history: (objectId: string) => fetchJson<HistoryGraph>(`${base}/objects/${id(objectId)}/history`),
  sourceEvidence: (evidenceId: string) =>
    fetchJson<SourceRevision>(`${base}/evidence/${id(evidenceId)}`),
  coverage: () => fetchJson<ImportRun[]>(`${base}/coverage`),
  ask: (body: { question: string; repo?: string; pr_number?: number; paths?: string[]; symbols?: string[] }) =>
    postJson<QuestionAnswer>(`${base}/questions`, body),
  prDecisions: (repo: string, number: number) => {
    const [owner, name] = repo.split("/")
    return fetchJson<QuestionAnswer>(`${base}/prs/${id(owner)}/${id(name)}/${number}/decisions`)
  },
}
