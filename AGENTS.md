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

Container packages stay private by default. Changing a package's visibility,
publishing a new public image, pushing release tags or images, or deploying requires
explicit authorization from a maintainer with that authority, recorded on the change
and given after the audits above pass. Implementation work, code review, and PR
approval are not that authorization, and one authorization does not carry over to a
later version or package.
