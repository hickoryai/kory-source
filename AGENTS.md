# Public image contract

Kory Source images must be organization-neutral and reusable by other organizations.
Never include customer or company data, secrets, private decisions or specifications,
internal identifiers, organization-specific defaults, or bundled deployment
configuration. Inject organization configuration, credentials, repository selections,
and integrations at runtime. Use only synthetic or generic examples. Preserve
legitimate copyright, license notices, and package naming.

Before publication, audit the effective build context, wheel contents and metadata,
frontend assets, release pipeline, actual image configuration and history, and every
image layer (including files deleted or overwritten later). Automated checks are a
regression guard, not proof that arbitrary content is safe. Review all historical
package versions and untagged digests before changing package visibility: making a
package public can expose old versions. Missing access or evidence blocks publication.

Kory Source is meant to be distributed publicly. Publish an image, push a release
tag, or change a package's visibility only after the audits above pass for that exact
artifact and the package's full history, and a maintainer with publishing authority
has approved that specific action on the record (for example, on the release PR or
tag). Approving an implementation or PR does not by itself authorize publication.
Publication does not depend on any particular agent, session, or reviewer chain.
Until a package's history passes review, keep that package private.

# Update to the latest dependencies

When asked to update this project to the latest dependencies, run
`scripts/update_dependencies.sh` from a clean, current `main`. It works on a
branch and opens a draft pull request. It merges the upstream project's
`main`, upgrades the Python and UI lockfiles, records the new upstream
baseline in `FORK.md` and the refresh in `CHANGELOG.md`, and runs lint, the
full test suite, the UI build, the source audit, and the image build, layer
audit and reviewer smoke test.

Upstream is always merged with a merge commit. Never rebase `main`, never
force-push `main`, and merge the refresh pull request with a merge commit, not
a squash or a rebase.

Stop and report instead of working around any of these:

- the upstream merge conflicts (resolve inside the fork-delta policy in
  `FORK.md`, commit the merge, then run the script with `--continue`);
- a lint, test, or UI build failure;
- an audit finding;
- a bundled program that does not build or does not answer `--version`.

Use `--image-in-ci` only where no container runtime is available; the pull
request's "Reviewer image" check must then pass before merge.

The bundled reviewer programs and agent packages have no version in this
repository. Every release resolves the newest of each when it builds, and the
release notes list them. To hold back a broken upstream for one release, run
the Release workflow from the version tag with `overrides`; do not add a
version to the `Dockerfile`.

After the pull request merges, the next version is released under the
publication rules above. A maintainer with publishing authority who gives the
instruction to update to the latest dependencies and publish a new version has
approved, on the record, the one version tag that refresh produces. That
approval holds only if every audit passes for that exact image; a finding, a
failed check or a failed build withdraws it, and the release waits for a new
approval. It covers no other tag, no later refresh and no change of package
visibility. Record the instruction on the release pull request. The script
never tags and never publishes.
