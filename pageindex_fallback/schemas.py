"""The values the fallback passes around.

Both are plain data with no behavior beyond one derived property, so they live apart from
the classes that produce them — `FallbackTreeBuilder` returns a `FallbackState` to its
caller and a `TocExtraction` to itself.
"""
from dataclasses import dataclass


@dataclass
class FallbackState:
    """Whether a tree is a page-level index, and why.

    `accuracy` is the pre-repair verification score for a TOC that failed verification, and
    None when no verification ever ran — which is how a caller tells a failed verification
    apart from an extraction that raised on every attempt.
    """

    used: bool = False
    accuracy: float | None = None


@dataclass
class TocExtraction:
    """The outcome of one extraction-and-verification round.

    `flat_toc` is None when the result did not pass verification, which is the whole
    signal: either the repair pass could not place every item, or there was nothing to
    repair because verify_toc took its early exit.
    """

    flat_toc: list | None
    accuracy: float
    unfixed_items: list

    @property
    def usable(self):
        return self.flat_toc is not None
