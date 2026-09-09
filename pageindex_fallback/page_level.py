"""The page-level index: one root node plus one leaf per page."""
from pageindex.utils import post_processing


class PageLevelIndex:
    """One root node plus one leaf per page.

    Nothing here calls an LLM — the whole structure is derived from the page count — so it
    cannot fail and costs nothing beyond the summaries a caller may add afterwards.

        tree = PageLevelIndex(page_list, doc_title="report").tree()

    `start_index` is the absolute page number of `page_list[0]`, so a subrange of a larger
    document keeps absolute page numbering.
    """

    def __init__(self, page_list, doc_title="Document", start_index=1):
        self.page_list = page_list
        self.doc_title = doc_title
        self.start_index = start_index

    @property
    def page_count(self):
        return len(self.page_list)

    @property
    def end_index(self):
        """Absolute page number of the last page."""
        return self.start_index + self.page_count - 1

    def flat_toc(self):
        """A flat TOC in the same shape pageindex.page_index.process_no_toc() returns.

        One root item plus one item per page, so a 10-page document becomes 11 nodes once
        post_processing()/list_to_tree() have run: a root spanning the document and ten
        single-page leaves. Returning the same shape as the real extractor is the point —
        the caller's normal post-processing pipeline handles it unchanged.

        appear_start is preset because post_processing() derives item i's end_index from
        item i+1's appear_start:

            item i   end_index = next.physical_index - 1   if next appear_start == 'yes'
                               = next.physical_index       otherwise
            last item          = end_physical_index

        So pages 2..N carry 'yes' to give the page before them a single-page range, and
        the last page picks up end_physical_index. Page 1 is the exception: its flag
        decides the ROOT's end_index and nothing else, so it carries 'no' to leave the
        root at a sane `start_index` rather than `start_index - 1`. tree() then widens the
        root over the whole document.

        Because appear_start is already set on every item, a caller must NOT run
        check_title_appearance_in_start_concurrent() over this list: that would ask the
        model whether the synthetic title 'Page 7' appears at the top of page 7 and
        overwrite these values with the answers. tree() below is the safe entry point.
        """
        if not self.page_list:
            return []

        items = [{
            'structure': '1',
            'title': self.doc_title,
            'physical_index': self.start_index,
        }]

        for offset, page_number in enumerate(range(self.start_index, self.end_index + 1)):
            items.append({
                'structure': f'1.{offset + 1}',
                'title': f'Page {page_number}',
                'physical_index': page_number,
                # 'no' on the first page only — it sets the root's end_index, which is
                # widened afterwards anyway. See above.
                'appear_start': 'no' if offset == 0 else 'yes',
            })
        return items

    def tree(self):
        """The assembled tree, ready for node ids, text and summaries.

        Skips add_preface_if_needed() (a no-op: the first physical_index is already
        start_index) and check_title_appearance_in_start_concurrent() (must be skipped —
        see flat_toc).
        """
        if not self.page_list:
            return []
        tree = post_processing(self.flat_toc(), self.end_index)
        return self.widen_root(tree, self.end_index)

    def root_summary(self):
        """Deterministic summary for this index's root."""
        return self.root_summary_for(self.doc_title, self.page_count)

    @staticmethod
    def root_summary_for(doc_title, page_count):
        """Deterministic root summary, used when the LLM roll-up came back empty.

        The root spans the whole document, so summarizing it from its own text would mean
        sending the entire document in a single prompt. This keeps the `summary` field
        present on every node without that cost or context-length risk. A static method
        because the summarizer needs the text without holding a page list.
        """
        return (
            f"Page-level index of {doc_title} ({page_count} pages). "
            "Document structure could not be detected; each page is indexed as a "
            "separate node."
        )

    @staticmethod
    def widen_root(tree, end_index):
        """Extend the root over the whole document.

        post_processing() can only give a parent a range reaching as far as its first
        child's start page, which leaves the root spanning one page. Everything else in
        the tree is already correct, so this fixes up the root alone.
        """
        if tree:
            tree[0]['end_index'] = end_index
        return tree
