import { useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { api, type AnchorModel } from "@/lib/api"
import { useAuth } from "@/lib/auth"
import { useAsync, useDocumentTitle } from "@/lib/hooks"

function indexedLabel(value: string | null): string {
  if (!value) return "Not indexed"
  return `Indexed ${value.slice(0, 16).replace("T", " ")} UTC`
}

function countLabel(count: number, singular: string): string {
  return `${count} ${count === 1 ? singular : `${singular}s`}`
}

export function AnchorsPage() {
  useDocumentTitle("Anchors")
  const { user } = useAuth()
  const [refreshKey, setRefreshKey] = useState(0)
  const [actionError, setActionError] = useState<string | null>(null)
  const { data, loading, error } = useAsync(() => api.getAnchors(), [refreshKey])

  const run = async (action: () => Promise<unknown>) => {
    setActionError(null)
    try {
      await action()
      setRefreshKey((key) => key + 1)
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Request failed")
    }
  }

  const pending = data?.anchors.filter((anchor) => anchor.resolution_status === "pending") ?? []
  const dangling = data?.anchors.filter((anchor) => anchor.resolution_status === "dangling") ?? []

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Symbol anchors</h1>
        <p className="text-sm text-muted-foreground">
          Links from the registry to code. A rename keeps the link. An
          unresolved link stays on this page until someone confirms or retires
          it.
        </p>
      </div>

      {loading && <p className="text-sm text-muted-foreground">Loading...</p>}
      {error && <p className="text-sm text-destructive">{error}</p>}
      {actionError && <p className="text-sm text-destructive">{actionError}</p>}

      <Card>
        <CardHeader>
          <CardTitle>Index freshness</CardTitle>
          <CardDescription>When each repository was last indexed, and how many symbols it holds.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {data && data.repos.length === 0 && (
            <p className="text-sm text-muted-foreground">No repositories indexed yet.</p>
          )}
          {data?.repos.map((repo) => (
            <div
              key={`${repo.owner}/${repo.repo}`}
              className="flex flex-wrap items-center gap-3 border-b pb-3 last:border-0"
            >
              <div className="min-w-48">
                <p className="text-sm font-medium">
                  {repo.owner}/{repo.repo}
                </p>
                <p className="text-xs text-muted-foreground">{indexedLabel(repo.last_indexed)}</p>
              </div>
              <span className="text-sm text-muted-foreground">{countLabel(repo.file_count, "file")}</span>
              <span className="text-sm text-muted-foreground">{countLabel(repo.symbol_count, "symbol")}</span>
              {repo.commit && (
                <span className="font-mono text-xs text-muted-foreground">{repo.commit.slice(0, 12)}</span>
              )}
              {repo.pending_count > 0 && <Badge variant="secondary">{repo.pending_count} pending</Badge>}
              {repo.dangling_count > 0 && <Badge variant="destructive">{repo.dangling_count} dangling</Badge>}
            </div>
          ))}
        </CardContent>
      </Card>

      <AnchorTable
        title="Pending relink"
        description="A candidate was found. Confirm applies it. Retire keeps the row and stops resolution."
        anchors={pending}
        isAdmin={Boolean(user?.is_admin)}
        onConfirm={(anchor) => run(() => api.confirmAnchor(anchor.owner, anchor.repo, anchor.id))}
        onRetire={(anchor) => run(() => api.retireAnchor(anchor.owner, anchor.repo, anchor.id))}
        showConfirm
      />
      <AnchorTable
        title="Dangling"
        description="No symbol matched. The anchor is still stored."
        anchors={dangling}
        isAdmin={Boolean(user?.is_admin)}
        onConfirm={() => undefined}
        onRetire={(anchor) => run(() => api.retireAnchor(anchor.owner, anchor.repo, anchor.id))}
        showConfirm={false}
      />
    </div>
  )
}

function AnchorTable({
  title,
  description,
  anchors,
  isAdmin,
  onConfirm,
  onRetire,
  showConfirm,
}: {
  title: string
  description: string
  anchors: AnchorModel[]
  isAdmin: boolean
  onConfirm: (anchor: AnchorModel) => void
  onRetire: (anchor: AnchorModel) => void
  showConfirm: boolean
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {anchors.length === 0 && <p className="text-sm text-muted-foreground">None.</p>}
        {anchors.map((anchor) => (
          <div key={anchor.id} className="flex flex-wrap items-start justify-between gap-3 border-b pb-3 last:border-0">
            <div>
              <p className="text-sm font-medium">
                {anchor.owner}/{anchor.repo} {anchor.qualified_name}
              </p>
              <p className="text-xs text-muted-foreground">{anchor.path}</p>
              {anchor.proposal_qualified_name && (
                <p className="text-xs text-muted-foreground">
                  Proposal: {anchor.proposal_qualified_name} in {anchor.proposal_path}
                  {anchor.proposal_score ? ` (${anchor.proposal_score.toFixed(2)})` : ""}
                  {anchor.proposal_reason ? ` — ${anchor.proposal_reason}` : ""}
                </p>
              )}
            </div>
            {isAdmin && (
              <div className="flex gap-2">
                {showConfirm && (
                  <Button size="sm" onClick={() => onConfirm(anchor)}>
                    Confirm
                  </Button>
                )}
                <Button size="sm" variant="outline" onClick={() => onRetire(anchor)}>
                  Retire
                </Button>
              </div>
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
