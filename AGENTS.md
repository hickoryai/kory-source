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
