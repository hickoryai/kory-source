import { fetchJson, postJson } from "./http"
import type { AnchorModel, AnchorsResponse } from "./types"

// Symbol anchors that a rename or delete left unresolved, plus index freshness.
export const anchorsApi = {
  getAnchors: () => fetchJson<AnchorsResponse>("/api/anchors"),

  confirmAnchor: (owner: string, repo: string, id: string) =>
    postJson<AnchorModel>(
      `/api/anchors/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/${encodeURIComponent(id)}/confirm`,
      {}
    ),

  retireAnchor: (owner: string, repo: string, id: string) =>
    postJson<AnchorModel>(
      `/api/anchors/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/${encodeURIComponent(id)}/retire`,
      {}
    ),
}
