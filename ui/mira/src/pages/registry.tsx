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
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import {
  registryApi,
  type Attribution,
  type Citation,
  type DecisionEvidence,
  type HistoryGraph,
  type QuestionAnswer,
  type NewWork,
  type RegistryObject,
  type RegistryRelationship,
  type WorkView,
} from "@/lib/api/registry"
import { useAuth } from "@/lib/auth"
import { useAsync, useDocumentTitle } from "@/lib/hooks"

const DELIVERY_STATES = ["planned", "active", "shipped", "canceled", "archived"]
const WORK_LINKS = [
  { type: "addresses", label: "Requirement" },
  { type: "records", label: "Decision" },
  { type: "parent_of", label: "Child work" },
  { type: "blocks", label: "Blocks work" },
  { type: "relates", label: "Related" },
]

function ago(epoch: number): string {
  const days = Math.floor((Date.now() / 1000 - epoch) / 86400)
  return days <= 0 ? "today" : `${days}d ago`
}

function StatusBadges({ obj }: { obj: RegistryObject }) {
  return (
    <span className="flex flex-wrap gap-1">
      <Badge variant="outline">{obj.kind}</Badge>
      <Badge variant={obj.status === "approved" ? "default" : "secondary"}>{obj.status}</Badge>
      {obj.kind !== "work" && obj.kind !== "verification_evidence" && (
        <Badge variant={obj.realization === "merged_with_conflict" ? "destructive" : "outline"}>
          {obj.realization}
        </Badge>
      )}
      <Badge variant="outline">{obj.confidence.toFixed(2)}</Badge>
    </span>
  )
}

function ConflictCard({ conflict }: { conflict: RegistryRelationship }) {
  const answer = conflict.answer
    ? `Answer: ${conflict.answer}${conflict.answered_by ? ` by ${conflict.answered_by}` : ""}`
    : "Unanswered"
  return (
    <div className="rounded border p-2 text-xs">
      <p>
        <span className="font-medium">{conflict.relation}</span> {conflict.target_id}
      </p>
      <p>New: “{conflict.new_clause}”</p>
      <p>Existing: “{conflict.existing_clause}”</p>
      <p>
        {answer}
        {conflict.session ? ` · session ${conflict.session}` : ""}
      </p>
    </div>
  )
}

function ObjectRow({
  obj,
  onOpen,
  children,
}: {
  obj: RegistryObject
  onOpen: (id: string) => void
  children?: React.ReactNode
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3 border-b pb-3 last:border-0">
      <div className="min-w-0 space-y-1">
        <button className="text-left text-sm font-medium hover:underline" onClick={() => onOpen(obj.id)}>
          {obj.title}
        </button>
        <p className="font-mono text-xs text-muted-foreground">
          {obj.id} · {obj.author_kind} {obj.author}
          {obj.model ? ` (${obj.model})` : ""} · {ago(obj.created_at)}
          {obj.branch ? ` · ${obj.repo}@${obj.branch}` : ""}
        </p>
        <StatusBadges obj={obj} />
        {obj.unresolved && obj.unresolved.length > 0 && (
          <p className="text-xs text-destructive">Unresolved: {obj.unresolved.join(", ")}</p>
        )}
        {obj.conflicts?.map((c) => (
          <ConflictCard key={c.id} conflict={c} />
        ))}
      </div>
      {children && <div className="flex flex-wrap gap-2">{children}</div>}
    </div>
  )
}

export function RegistryPage() {
  useDocumentTitle("Registry")
  const { user } = useAuth()
  const isAdmin = Boolean(user?.is_admin)
  const [refreshKey, setRefreshKey] = useState(0)
  const [actionError, setActionError] = useState<string | null>(null)
  const [openId, setOpenId] = useState<string | null>(null)
  const [workId, setWorkId] = useState<string | null>(null)
  const [concept, setConcept] = useState<string | null>(null)

  const queue = useAsync(() => registryApi.queue(), [refreshKey])
  const expired = useAsync(() => registryApi.expired(), [refreshKey])
  const concepts = useAsync(() => registryApi.concepts(), [refreshKey])
  const anchors = useAsync(() => registryApi.anchors(), [refreshKey])
  const works = useAsync(() => registryApi.works(), [refreshKey])

  const run = async (action: () => Promise<unknown>) => {
    setActionError(null)
    try {
      await action()
      setRefreshKey((key) => key + 1)
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Request failed")
    }
  }

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Registry</h1>
        <p className="text-sm text-muted-foreground">
          Work, decisions, rules, constraints, and requirements. Agents propose. Only a person
          approves; a merge records realization, never approval. Only approved objects reach a
          review. Imported history is visible only to approved corpus readers.
        </p>
      </div>
      {actionError && <p className="text-sm text-destructive">{actionError}</p>}

      <Tabs defaultValue="queue">
        <TabsList>
          <TabsTrigger value="queue">Approval queue</TabsTrigger>
          <TabsTrigger value="work">Work</TabsTrigger>
          <TabsTrigger value="concepts">Concepts</TabsTrigger>
          <TabsTrigger value="anchors">Anchors</TabsTrigger>
          <TabsTrigger value="expired">Expired</TabsTrigger>
          <TabsTrigger value="questions">Ask the source</TabsTrigger>
        </TabsList>

        <TabsContent value="questions">
          <QuestionPanel onOpen={setOpenId} />
        </TabsContent>

        <TabsContent value="queue">
          <Card>
            <CardHeader>
              <CardTitle>Proposals</CardTitle>
              <CardDescription>
                Conflicts first, then by how far the work has shipped, then oldest first.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {queue.loading && <p className="text-sm text-muted-foreground">Loading...</p>}
              {queue.data?.length === 0 && <p className="text-sm text-muted-foreground">Nothing waiting.</p>}
              {queue.data?.map((obj) => (
                <ObjectRow key={obj.id} obj={obj} onOpen={setOpenId}>
                  {isAdmin && (obj.conflicts?.length ?? 0) > 0 ? (
                    <>
                      <Button size="sm" variant="outline" onClick={() => run(() => registryApi.resolve(obj.id, "keep"))}>
                        Keep existing
                      </Button>
                      <Button size="sm" onClick={() => run(() => registryApi.resolve(obj.id, "supersede"))}>
                        Supersede
                      </Button>
                      <Button size="sm" variant="outline" onClick={() => run(() => registryApi.resolve(obj.id, "defer"))}>
                        Defer
                      </Button>
                    </>
                  ) : isAdmin ? (
                    <>
                      <Button size="sm" onClick={() => run(() => registryApi.approve(obj.id))}>
                        Approve
                      </Button>
                      <Button size="sm" variant="outline" onClick={() => run(() => registryApi.reject(obj.id))}>
                        Reject
                      </Button>
                    </>
                  ) : null}
                </ObjectRow>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="work" className="space-y-4">
          <WorkList
            works={works.data ?? []}
            selected={workId}
            onSelect={setWorkId}
            onCreate={(body) => run(() => registryApi.createWork(body))}
          />
          {workId && (
            <WorkPanel
              key={`${workId}-${refreshKey}`}
              workId={workId}
              onOpen={setOpenId}
              onWork={setWorkId}
              run={run}
            />
          )}
        </TabsContent>

        <TabsContent value="concepts">
          <Card>
            <CardHeader>
              <CardTitle>Concepts</CardTitle>
              <CardDescription>Concepts never rename. Pick one to see what belongs to it.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-wrap gap-2">
                {concepts.data?.map((c) => (
                  <Button
                    key={c.id}
                    size="sm"
                    variant={concept === c.id ? "default" : "outline"}
                    onClick={() => setConcept(c.id)}
                  >
                    {c.title} ({c.member_count ?? 0})
                  </Button>
                ))}
                {concepts.data?.length === 0 && <p className="text-sm text-muted-foreground">No concepts yet.</p>}
              </div>
              {concept && <ConceptMembers concept={concept} onOpen={setOpenId} />}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="anchors">
          <Card>
            <CardHeader>
              <CardTitle>Anchored symbols</CardTitle>
              <CardDescription>Paths shown are where the index resolves each anchor today.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {anchors.data?.length === 0 && <p className="text-sm text-muted-foreground">No anchors yet.</p>}
              {anchors.data?.map((a) => (
                <div key={`${a.repo}:${a.path}:${a.qualified_name}`} className="border-b pb-3 last:border-0">
                  <p className="font-mono text-sm">
                    {a.repo} {a.path} {a.qualified_name}
                    {a.resolution_status && (
                      <Badge className="ml-2" variant={a.resolution_status === "dangling" ? "destructive" : "outline"}>
                        {a.resolution_status}
                      </Badge>
                    )}
                  </p>
                  {a.objects.map((o) => (
                    <button key={o.id} className="block text-left text-xs hover:underline" onClick={() => setOpenId(o.id)}>
                      {o.id} · {o.kind} · {o.status} · {o.title}
                    </button>
                  ))}
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="expired">
          <Card>
            <CardHeader>
              <CardTitle>Expired proposals</CardTitle>
              <CardDescription>
                Agent proposals with no PR 14 days after they were recorded or last restored.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {expired.data?.length === 0 && <p className="text-sm text-muted-foreground">None.</p>}
              {expired.data?.map((obj) => (
                <ObjectRow key={obj.id} obj={obj} onOpen={setOpenId}>
                  {isAdmin && (
                    <Button size="sm" onClick={() => run(() => registryApi.restore(obj.id))}>
                      Restore
                    </Button>
                  )}
                </ObjectRow>
              ))}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {openId && <ObjectPanel key={`${openId}-${refreshKey}`} objectId={openId} onOpen={setOpenId} />}
    </div>
  )
}

function ConceptMembers({ concept, onOpen }: { concept: string; onOpen: (id: string) => void }) {
  const { data } = useAsync(() => registryApi.conceptMembers(concept), [concept])
  return (
    <div className="space-y-3">
      {data?.map((obj) => <ObjectRow key={obj.id} obj={obj} onOpen={onOpen} />)}
    </div>
  )
}

function ObjectPanel({ objectId, onOpen }: { objectId: string; onOpen: (id: string) => void }) {
  const detail = useAsync(() => registryApi.detail(objectId), [objectId])
  const chain = useAsync(() => registryApi.chain(objectId), [objectId])
  const obj = detail.data
  return (
    <Card>
      <CardHeader>
        <CardTitle>{obj?.title ?? objectId}</CardTitle>
        <CardDescription className="font-mono">{objectId}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        {detail.error && <p className="text-destructive">{detail.error}</p>}
        {obj && (
          <>
            <StatusBadges obj={obj} />
            {obj.summary && <p>{obj.summary}</p>}
            {obj.body && <p className="whitespace-pre-wrap text-muted-foreground">{obj.body}</p>}
            <p className="text-xs text-muted-foreground">
              Decided by {obj.decided_by} · basis {obj.basis} · conflict check {obj.conflict_check}
              {obj.status_reason ? ` · ${obj.status_reason}` : ""}
            </p>
            {obj.pr && (
              <a className="text-xs underline" href={obj.pr} target="_blank" rel="noreferrer">
                {obj.pr}
              </a>
            )}
            {obj.source_url && <p className="font-mono text-xs">{obj.source_url}</p>}
            {obj.concepts.length > 0 && <p className="text-xs">Concepts: {obj.concepts.join(", ")}</p>}
            {obj.anchors.map((a) => (
              <p key={`${a.repo}:${a.path}:${a.qualified_name}`} className="font-mono text-xs">
                {a.repo} {a.path} {a.qualified_name}
              </p>
            ))}
            {obj.conflicts && obj.conflicts.length > 0 && (
              <div className="space-y-2">
                <p className="font-medium">Conflicts</p>
                {obj.conflicts.map((c) => (
                  <ConflictCard key={c.id} conflict={c} />
                ))}
              </div>
            )}
          </>
        )}
        <EvidenceSection objectId={objectId} />
        <HistorySection objectId={objectId} onOpen={onOpen} />
        {chain.data && chain.data.length > 1 && (
          <div className="space-y-2">
            <p className="font-medium">Supersession chain, first edge only (legacy view)</p>
            {chain.data.map((c) => (
              <div key={c.id}>
                <button
                  className={`block text-left text-xs hover:underline ${c.id === objectId ? "font-semibold" : ""}`}
                  onClick={() => onOpen(c.id)}
                >
                  {c.id} · {c.status} · {c.title}
                  {c.session ? ` · session ${c.session}` : ""}
                </button>
                {c.conflicts?.map((conflict) => (
                  <ConflictCard key={conflict.id} conflict={conflict} />
                ))}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function WorkList({
  works,
  selected,
  onSelect,
  onCreate,
}: {
  works: RegistryObject[]
  selected: string | null
  onSelect: (id: string) => void
  onCreate: (body: NewWork) => void
}) {
  const [draft, setDraft] = useState<NewWork>({ title: "", goal: "", work_scope: "", plan: "", owner: "" })
  const set = (key: keyof NewWork) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
    setDraft({ ...draft, [key]: e.target.value })
  return (
    <Card>
      <CardHeader>
        <CardTitle>Work</CardTitle>
        <CardDescription>Goal, scope, plan, owner, and delivery state.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid gap-2 md:grid-cols-2">
          <Input placeholder="Title" value={draft.title} onChange={set("title")} />
          <Input placeholder="Owner" value={draft.owner} onChange={set("owner")} />
          <Input placeholder="Goal" value={draft.goal} onChange={set("goal")} />
          <Input placeholder="Scope" value={draft.work_scope} onChange={set("work_scope")} />
          <Textarea className="md:col-span-2" placeholder="Plan" value={draft.plan} onChange={set("plan")} />
        </div>
        <Button
          size="sm"
          disabled={!draft.title.trim()}
          onClick={() => {
            onCreate(draft)
            setDraft({ title: "", goal: "", work_scope: "", plan: "", owner: "" })
          }}
        >
          Create work
        </Button>
        {works.map((w) => (
          <button
            key={w.id}
            className={`block w-full border-b pb-2 text-left text-sm last:border-0 ${w.id === selected ? "font-semibold" : ""}`}
            onClick={() => onSelect(w.id)}
          >
            {w.title} <span className="font-mono text-xs text-muted-foreground">{w.id} · {w.delivery_state}</span>
            {w.source_system && (
              <span className="text-xs text-muted-foreground">
                {" "}· {w.source_system} {w.source_id} ({w.sync_status})
              </span>
            )}
          </button>
        ))}
      </CardContent>
    </Card>
  )
}

function WorkPanel({
  workId,
  onOpen,
  onWork,
  run,
}: {
  workId: string
  onOpen: (id: string) => void
  onWork: (id: string) => void
  run: (action: () => Promise<unknown>) => Promise<void>
}) {
  const { data, error } = useAsync<WorkView>(() => registryApi.work(workId), [workId])
  const [target, setTarget] = useState("")
  const [linkType, setLinkType] = useState("addresses")
  if (error) return <p className="text-sm text-destructive">{error}</p>
  if (!data) return null
  const { work } = data
  const list = (items: RegistryObject[], open: (id: string) => void) =>
    items.length === 0 ? (
      <p className="text-xs text-muted-foreground">None.</p>
    ) : (
      items.map((o) => (
        <button key={o.id} className="block text-left text-xs hover:underline" onClick={() => open(o.id)}>
          {o.id} · {o.title} · {o.kind === "work" ? o.delivery_state : `${o.status}, ${o.realization}`}
        </button>
      ))
    )
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Card className="lg:col-span-3">
        <CardHeader>
          <CardTitle>{work.title}</CardTitle>
          <CardDescription>
            {work.goal || "No goal recorded."} {work.owner && `· Owner ${work.owner}`}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          {work.work_scope && <p>Scope: {work.work_scope}</p>}
          {work.plan && <p className="whitespace-pre-wrap">{work.plan}</p>}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-muted-foreground">Link</span>
            <select
              className="h-8 rounded border bg-transparent px-2 text-xs"
              value={linkType}
              onChange={(e) => setLinkType(e.target.value)}
            >
              {WORK_LINKS.map((l) => (
                <option key={l.type} value={l.type}>
                  {l.label}
                </option>
              ))}
            </select>
            <Input className="h-8 w-56" placeholder="reg_… id" value={target} onChange={(e) => setTarget(e.target.value)} />
            <Button size="sm" disabled={!target.trim()} onClick={() => run(() => registryApi.linkWork(workId, target.trim(), linkType))}>
              Add link
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Implementation</CardTitle>
          <CardDescription>Derived from PR events on the linked records.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 text-xs">
          {Object.entries(data.implementation.realization_counts).map(([state, n]) => (
            <p key={state}>
              {state}: {n}
            </p>
          ))}
          {data.implementation.prs.map((pr) => (
            <a key={pr} className="block underline" href={pr} target="_blank" rel="noreferrer">
              {pr}
            </a>
          ))}
          {data.implementation.prs.length === 0 && <p className="text-muted-foreground">No PRs yet.</p>}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Verification</CardTitle>
          <CardDescription>Evidence only. A merged PR does not verify a requirement.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 text-xs">
          {data.verification.length === 0 && <p className="text-muted-foreground">No requirements linked.</p>}
          {data.verification.map((v) => (
            <div key={v.requirement.id}>
              <button className="text-left hover:underline" onClick={() => onOpen(v.requirement.id)}>
                {v.requirement.title}
              </button>
              <Badge className="ml-2" variant={v.latest_result === "pass" ? "default" : v.latest_result === "fail" ? "destructive" : "outline"}>
                {v.latest_result}
              </Badge>
              {v.evidence.map((e) => (
                <p key={e.id} className="font-mono text-muted-foreground">
                  {e.result} · {e.method} @ {e.tested_revision.slice(0, 12)}
                  {e.artifact_url && (
                    <a className="ml-1 underline" href={e.artifact_url} target="_blank" rel="noreferrer">
                      artifact
                    </a>
                  )}
                </p>
              ))}
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Delivery</CardTitle>
          <CardDescription>Set by a person. A merge never changes it.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          {DELIVERY_STATES.map((state) => (
            <Button
              key={state}
              size="sm"
              variant={data.delivery_state === state ? "default" : "outline"}
              onClick={() => run(() => registryApi.setDeliveryState(workId, state))}
            >
              {state}
            </Button>
          ))}
        </CardContent>
      </Card>

      <Card className="lg:col-span-3">
        <CardContent className="grid gap-4 pt-6 md:grid-cols-2 lg:grid-cols-4">
          <div>
            <p className="mb-1 text-sm font-medium">Requirements</p>
            {list(data.requirements, onOpen)}
          </div>
          <div>
            <p className="mb-1 text-sm font-medium">Decisions</p>
            {list(data.decisions, onOpen)}
          </div>
          <div>
            <p className="mb-1 text-sm font-medium">Children</p>
            {list(data.children, onWork)}
            {data.dependencies.parent && (
              <p className="mt-2 text-xs">Parent: {data.dependencies.parent.title}</p>
            )}
          </div>
          <div>
            <p className="mb-1 text-sm font-medium">Dependencies</p>
            <p className="text-xs text-muted-foreground">Blocks</p>
            {list(data.dependencies.blocks, onWork)}
            <p className="mt-1 text-xs text-muted-foreground">Blocked by</p>
            {list(data.dependencies.blocked_by, onWork)}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

// ── HAI-12825: evidence, attribution, history, questions ─────────────────

function Field({ label, value, unknown }: { label: string; value: unknown; unknown?: boolean }) {
  if (unknown || value === null || value === undefined || value === "") {
    return (
      <span className="text-muted-foreground">
        {label}: <em>unknown</em>
      </span>
    )
  }
  return (
    <span>
      {label}: <span className="font-mono">{String(value)}</span>
    </span>
  )
}

function AttributionRow({ a }: { a: Attribution }) {
  const unknown = new Set(a.unknown_fields ?? Object.entries(a.field_states ?? {}).filter(([, s]) => s.state === "unknown").map(([k]) => k))
  const inferred = new Set(a.inferred_fields ?? Object.entries(a.field_states ?? {}).filter(([, s]) => s.state === "inferred").map(([k]) => k))
  const mark = (k: string) => (inferred.has(k) ? " (inferred)" : "")
  return (
    <div className="rounded border p-2 text-xs">
      <p className="font-medium">
        {a.role.replace("_", " ")} · {a.actor_kind}
        {mark("actor_kind")}
      </p>
      <p className="flex flex-wrap gap-x-3">
        <Field label="identity" value={a.original_identity} unknown={unknown.has("original_identity")} />
        <Field label="bot" value={a.original_bot} unknown={unknown.has("original_bot")} />
        <Field label="provider" value={a.original_provider} unknown={unknown.has("original_provider")} />
        <Field label="model" value={a.original_model} unknown={unknown.has("original_model")} />
        <Field label="client" value={a.original_client} unknown={unknown.has("original_client")} />
        <Field label="environment" value={a.original_environment} unknown={unknown.has("original_environment")} />
      </p>
      <p className="flex flex-wrap gap-x-3 text-muted-foreground">
        normalized:
        <Field label="provider" value={a.provider_family} unknown={unknown.has("provider_family")} />
        <Field label="family" value={a.model_family} unknown={unknown.has("model_family")} />
        <Field label="version" value={a.model_version} unknown={unknown.has("model_version")} />
        <Field label="interface" value={a.client_interface} unknown={unknown.has("client_interface")} />
        <Field label="session" value={a.source_session_id} unknown={unknown.has("source_session_id")} />
        <Field label="run" value={a.source_run_id} unknown={unknown.has("source_run_id")} />
      </p>
      {a.reason && <p className="text-muted-foreground">{a.reason}</p>}
    </div>
  )
}

function CitationView({ citation }: { citation: Citation | DecisionEvidence }) {
  const attribution = "attribution" in citation ? citation.attribution : citation.attributions.filter((a) => a.role !== "importer" && a.role !== "extractor")
  const provenance = "provenance" in citation ? citation.provenance : citation.attributions.filter((a) => a.role === "importer" || a.role === "extractor")
  return (
    <div className="space-y-2">
      {citation.legacy && (
        <p className="text-xs text-muted-foreground">
          Legacy row: no preserved source quote. Approval evidence is unknown.
        </p>
      )}
      {citation.quotes.map((q) => {
        const rev = citation.revisions.find((r) => r.evidence_id === q.evidence_id)
        return (
          <blockquote key={`${q.evidence_id}:${q.quote_start ?? q.quote}`} className="border-l-2 pl-2 text-xs">
            “{q.quote}”
            <span className="block text-muted-foreground">
              {q.evidence_role ?? q.role} · {rev?.source_type}{" "}
              {rev?.source_url && (
                <a className="underline" href={rev.source_url} target="_blank" rel="noreferrer">
                  source
                </a>
              )}{" "}
              · <span className="font-mono">{q.evidence_id}</span>
              {rev?.head_sha ? ` · head ${rev.head_sha.slice(0, 12)}` : ""}
              {rev?.commit_sha ? ` · commit ${rev.commit_sha.slice(0, 12)}` : ""}
            </span>
          </blockquote>
        )
      })}
      {attribution.map((a) => (
        <AttributionRow key={`${a.role}:${a.evidence_id}:${a.original_identity}`} a={a} />
      ))}
      {provenance.length > 0 && (
        <details className="text-xs">
          <summary>Importer and extractor</summary>
          {provenance.map((a) => (
            <AttributionRow key={`${a.role}:${a.evidence_id}`} a={a} />
          ))}
        </details>
      )}
      {citation.anchors.map((a) => (
        <p key={`${a.observed_commit}:${a.path}:${a.line_start}`} className="font-mono text-xs">
          {a.path}
          {a.qualified_name ? ` ${a.qualified_name}` : ""}
          {a.heading ? ` § ${a.heading}` : ""} @ {a.observed_commit.slice(0, 12)} · {a.resolution_status}
          {a.resolved_path && a.resolved_path !== a.path ? ` → ${a.resolved_path}` : ""}
        </p>
      ))}
    </div>
  )
}

function EvidenceSection({ objectId }: { objectId: string }) {
  const { data, error } = useAsync(() => registryApi.evidence(objectId), [objectId])
  if (error || !data) return null
  return (
    <div className="space-y-2">
      <p className="font-medium">Evidence and attribution</p>
      <CitationView citation={data} />
      {data.versions.map((v) => (
        <p key={v.revision} className="text-xs text-muted-foreground">
          v{v.revision} · {v.approval_status} · {v.authority_tier.replaceAll("_", " ")}
          {v.authority_reasons.length ? ` — ${v.authority_reasons.join("; ")}` : ""}
        </p>
      ))}
    </div>
  )
}

function HistoryGraphView({ graph, onOpen }: { graph: HistoryGraph; onOpen: (id: string) => void }) {
  return (
    <div className="space-y-1 text-xs">
      {graph.history_order.map((n) => (
        <button key={n} className="block text-left hover:underline" onClick={() => onOpen(n)}>
          {graph.heads.includes(n) ? "● " : "○ "}
          {n} · {graph.nodes[n].status} · {graph.nodes[n].title}
        </button>
      ))}
      {graph.edges.map((e) => (
        <p key={e.id} className="font-mono text-muted-foreground">
          {e.source_id} {e.symmetric ? "↔" : "→"} {e.target_id} {e.type} ({e.status})
          {e.relation ? ` ${e.relation}` : ""}
        </p>
      ))}
      {!graph.complete && (
        <p className="text-muted-foreground">
          Incomplete: {[...graph.truncated, graph.hidden_links ? `${graph.hidden_links} hidden links` : ""].filter(Boolean).join("; ")}
        </p>
      )}
    </div>
  )
}

function HistorySection({ objectId, onOpen }: { objectId: string; onOpen: (id: string) => void }) {
  const { data } = useAsync(() => registryApi.history(objectId), [objectId])
  if (!data || data.edges.length === 0) return null
  return (
    <div className="space-y-2">
      <p className="font-medium">Decision history (every edge, ancestry order oldest first)</p>
      <HistoryGraphView graph={data} onOpen={onOpen} />
    </div>
  )
}

function AnswerView({ answer, onOpen }: { answer: QuestionAnswer; onOpen: (id: string) => void }) {
  if (answer.kind === "needs_context") {
    return <p className="text-sm">Needs: {answer.needs}</p>
  }
  return (
    <div className="space-y-3 text-sm">
      <p>
        <Badge variant={answer.verdict === "confirmed_supersession" ? "destructive" : "outline"}>
          {answer.verdict.replaceAll("_", " ")}
        </Badge>
        {answer.pr?.head_sha && (
          <span className="ml-2 font-mono text-xs">
            base {answer.pr.base_sha?.slice(0, 12)} · head {answer.pr.head_sha.slice(0, 12)}
          </span>
        )}
      </p>
      {answer.narrative?.map((n, i) => (
        <p key={i}>
          {n.text}
          {n.evidence.length > 0 && <span className="font-mono text-xs text-muted-foreground"> [{n.evidence.join(", ")}]</span>}
        </p>
      ))}
      {answer.findings?.map((f) => (
        <Card key={f.decision.id}>
          <CardHeader>
            <CardTitle className="text-sm">
              <button className="hover:underline" onClick={() => onOpen(f.decision.id)}>
                {f.decision.title}
              </button>
            </CardTitle>
            <CardDescription>
              {f.verdict.replaceAll("_", " ")} · {f.why_considered} · {f.decision.status} ·{" "}
              {f.decision.authority_tier?.replaceAll("_", " ") || "legacy"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {f.reasons.map((r) => (
              <p key={r} className="text-xs">
                {r}
              </p>
            ))}
            {answer.citations?.[f.decision.id] && <CitationView citation={answer.citations[f.decision.id]} />}
          </CardContent>
        </Card>
      ))}
      {answer.graph && answer.kind === "decision_history" && <HistoryGraphView graph={answer.graph} onOpen={onOpen} />}
    </div>
  )
}

function QuestionPanel({ onOpen }: { onOpen: (id: string) => void }) {
  const [repo, setRepo] = useState("")
  const [number, setNumber] = useState("")
  const [question, setQuestion] = useState("")
  const [answer, setAnswer] = useState<QuestionAnswer | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const coverage = useAsync(() => registryApi.coverage(), [])

  async function go(fn: () => Promise<QuestionAnswer>) {
    setBusy(true)
    setError(null)
    try {
      setAnswer(await fn())
    } catch (e) {
      setAnswer(null)
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Does PR #X override any prior decisions?</CardTitle>
        <CardDescription>
          Read-only. Freezes the PR at its head, follows every decision relationship, and cites the
          original source. Nothing is imported, approved, or posted.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-2">
          <Input className="w-64" placeholder="owner/repo" value={repo} onChange={(e) => setRepo(e.target.value)} />
          <Input className="w-28" placeholder="PR number" value={number} onChange={(e) => setNumber(e.target.value)} />
          <Button
            size="sm"
            disabled={busy || !repo.includes("/") || !/^\d+$/.test(number)}
            onClick={() => go(() => registryApi.prDecisions(repo.trim(), Number(number)))}
          >
            Check PR
          </Button>
        </div>
        <Textarea
          placeholder="Or ask: why does src/app/auth.py login keep sessions seven days?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <Button
          size="sm"
          variant="outline"
          disabled={busy || !question.trim()}
          onClick={() => go(() => registryApi.ask({ question: question.trim(), repo: repo.trim() || undefined }))}
        >
          Ask
        </Button>
        {error && <p className="text-sm text-destructive">{error}</p>}
        {answer && <AnswerView answer={answer} onOpen={onOpen} />}
        {coverage.data && coverage.data.length > 0 && (
          <details className="text-xs">
            <summary>Import coverage ({coverage.data.length} runs)</summary>
            {coverage.data.map((r) => (
              <p key={r.run_id} className="font-mono">
                {r.stream} · {r.status} · {Object.entries(r.coverage).map(([k, v]) => `${k}=${v ?? "unknown"}`).join(" ")}
              </p>
            ))}
          </details>
        )}
      </CardContent>
    </Card>
  )
}
