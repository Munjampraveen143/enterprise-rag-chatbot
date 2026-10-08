from typing import List, Dict, Any


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    text = text.strip()

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        target_end = min(
            start + chunk_size,
            text_length
        )

        end = target_end

        if end < text_length:

            while (
                end < text_length
                and not text[end].isspace()
            ):
                end += 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(
            0,
            end - chunk_overlap
        )

        while (
            next_start > 0
            and not text[next_start].isspace()
        ):
            next_start -= 1

        start = next_start

        if start >= end:
            start = end

    return chunks


def chunk_sections(
    sections: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[List[str], List[Dict[str, Any]]]:

    all_chunks = []
    all_metadata = []

    global_chunk_id = 1

    for section in sections:

        section_text = section.get(
            "text",
            ""
        )

        if not section_text or not section_text.strip():
            continue

        section_chunks = chunk_text(
            text=section_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for chunk in section_chunks:

            metadata = {
                key: value
                for key, value in section.items()
                if key != "text"
            }

            metadata["chunk_id"] = global_chunk_id

            all_chunks.append(chunk)
            all_metadata.append(metadata)

            global_chunk_id += 1

    return all_chunks, all_metadata