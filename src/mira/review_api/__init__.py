"""Review-without-post API for Kory Source.

Callers submit ``owner/repo``, a PR number, and the head SHA they expect.
The service freezes that snapshot, runs the normal review engine against it
through router-acp seats, and stores the findings. Nothing is written to
GitHub. See ``docs/review-api.md``.
"""
