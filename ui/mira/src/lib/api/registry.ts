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
}
