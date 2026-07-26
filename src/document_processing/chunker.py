from typing import List, Dict, Any

class DocumentChunker:
    """Recursive text chunker that splits text into overlapping chunks preserving page metadata."""

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunks(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits page text into overlapping chunks while preserving page metadata.
        Each chunk receives doc_id, chunk_id, page_number, file_name, and text.
        """
        chunks = []
        global_chunk_idx = 0

        for page_info in pages_data:
            text = page_info["text"]
            doc_id = page_info["doc_id"]
            file_name = page_info.get("file_name", "Unknown")
            page_number = page_info["page_number"]

            if not text:
                continue

            # If page text is within chunk size, keep it as single chunk
            if len(text) <= self.chunk_size:
                chunks.append({
                    "chunk_id": f"{doc_id}_c{global_chunk_idx}",
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "page_number": page_number,
                    "chunk_index": global_chunk_idx,
                    "text": text
                })
                global_chunk_idx += 1
                continue

            # Recursive overlap splitting
            start = 0
            text_length = len(text)
            
            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                
                # If we're not at end, try to snap to nearest paragraph or sentence end
                if end < text_length:
                    last_period = text.rfind(". ", start, end)
                    last_newline = text.rfind("\n", start, end)
                    snap_point = max(last_period, last_newline)
                    
                    # Only snap if reasonable (more than 50% into chunk)
                    if snap_point != -1 and snap_point > start + (self.chunk_size // 2):
                        end = snap_point + 1

                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append({
                        "chunk_id": f"{doc_id}_c{global_chunk_idx}",
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "page_number": page_number,
                        "chunk_index": global_chunk_idx,
                        "text": chunk_text
                    })
                    global_chunk_idx += 1

                if end >= text_length:
                    break

                # Advance with overlap
                start += (self.chunk_size - self.chunk_overlap)

        return chunks
