"""Token windows with source offsets, or Markdown heading sections with intact fences."""
from collections import Counter
import hashlib
from pathlib import Path
import re
import tiktoken
from src.models import DocumentChunk, SourceSection

ENCODING = tiktoken.get_encoding("cl100k_base")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def sections(text: str, parent_doc: str) -> list[SourceSection]:
    """Ignore headings inside fenced code; preserve heading hierarchy and offsets."""
    starts, hierarchy = [(0, [])], []
    fence_char, fence_length, offset = None, 0, 0
    for line in text.splitlines(keepends=True):
        fence = FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if fence_char is None:
                fence_char, fence_length = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length and not line[fence.end():].strip():
                fence_char = None
        elif fence_char is None:
            match = HEADING.match(line.rstrip("\r\n"))
            if match:
                level, title = len(match.group(1)), match.group(2)
                hierarchy = [(n, h) for n, h in hierarchy if n < level] + [(level, title)]
                headers = [h for _, h in hierarchy]
                if offset == 0:
                    starts[0] = (0, headers)
                else:
                    starts.append((offset, headers))
        offset += len(line)
    result, counts = [], Counter()
    for index, (start, headers) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(text)
        if not text[start:end].strip():
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", "/".join(headers).lower()).strip("-") or "preamble"
        counts[slug] += 1
        result.append(SourceSection(section_id=f"{parent_doc}#{slug}~{counts[slug]}",
                                    parent_doc=parent_doc, headers=headers, start=start, end=end))
    return result


def make_chunk(text: str, doc: str, start: int, end: int, strategy: str,
               source_sections: list[SourceSection], **extra) -> DocumentChunk:
    covered = [s for s in source_sections if s.start < end and start < s.end]
    digest = hashlib.sha256(text[start:end].encode()).hexdigest()[:12]
    return DocumentChunk(chunk_id=f"{doc}::{strategy}::{start}-{end}::{digest}",
                         content=text[start:end], parent_doc=doc,
                         metadata={"strategy": strategy, "char_start": start, "char_end": end,
                                   "line_start": text.count("\n", 0, start) + 1,
                                   "line_end": text.count("\n", 0, max(start, end - 1)) + 1,
                                   "headers": [s.headers for s in covered],
                                   "section_ids": [s.section_id for s in covered], **extra})


class FixedSizeChunker:
    """Exact original-token overlap; move boundaries when needed to preserve UTF-8.

    These baseline windows deliberately do not preserve Markdown fences. Repairing
    fences would change the fixed-token experiment; StructureAwareChunker preserves them.
    """
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32):
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap

    def split(self, text: str, parent_doc: str) -> list[DocumentChunk]:
        if not text:
            return []
        ids = ENCODING.encode(text, disallowed_special=())
        byte_parts = [ENCODING.decode_single_token_bytes(t) for t in ids]
        raw, byte_offsets = text.encode(), [0]
        for part in byte_parts:
            byte_offsets.append(byte_offsets[-1] + len(part))
        # UTF-8 continuation bytes cannot start a decoded text fragment.
        valid = {i for i, pos in enumerate(byte_offsets)
                 if pos == len(raw) or raw[pos] & 0xC0 != 0x80}
        source_sections = sections(text, parent_doc)
        output, start = [], 0
        while start < len(ids):
            end = min(start + self.chunk_size, len(ids))
            if end != len(ids):
                while end > start + self.chunk_overlap and (end not in valid or end - self.chunk_overlap not in valid):
                    end -= 1
                if end <= start + self.chunk_overlap:
                    raise ValueError("Token window is too small to preserve Unicode and exact overlap")
            a = len(raw[:byte_offsets[start]].decode())
            b = len(raw[:byte_offsets[end]].decode())
            output.append(make_chunk(text, parent_doc, a, b, f"fixed-{self.chunk_size}", source_sections,
                                     token_start=start, token_end=end, token_ids=ids[start:end],
                                     overlap=self.chunk_overlap))
            if end == len(ids):
                break
            start = end - self.chunk_overlap
        return output


class StructureAwareChunker:
    """One heading section per chunk; fenced blocks are never divided."""
    def split(self, text: str, parent_doc: str) -> list[DocumentChunk]:
        source_sections = sections(text, parent_doc)
        return [make_chunk(text, parent_doc, s.start, s.end, "structure", source_sections)
                for s in source_sections]


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    """Only source Markdown is indexed; data/eval_queries.json cannot leak in."""
    documents = {p.relative_to(corpus_dir).as_posix(): p.read_text(encoding="utf-8")
                 for p in sorted(corpus_dir.rglob("*.md"))}
    if not documents:
        raise ValueError(f"No Markdown documents found in {corpus_dir}")
    return documents


def chunk_corpus(documents: dict[str, str], strategy: str) -> list[DocumentChunk]:
    if strategy == "structure":
        chunker = StructureAwareChunker()
    elif strategy in {"fixed-256", "fixed-512", "fixed-1024"}:
        chunker = FixedSizeChunker(int(strategy.split("-")[1]), 32)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return [chunk for doc, text in documents.items() for chunk in chunker.split(text, doc)]
