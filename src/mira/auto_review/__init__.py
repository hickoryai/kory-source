"""Automatic secondary reviewer. Off unless ``KORY_SOURCE_AUTO_REVIEW=1``.

Greptile stays the primary reviewer. This package queues one frozen-commit
``COMMENT`` review for the single trial repository and posts nothing else.
"""
