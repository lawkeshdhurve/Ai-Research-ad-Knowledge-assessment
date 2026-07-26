import json
from typing import List, Dict, Any, Optional
from config.settings import settings
from src.vector_store.manager import VectorStoreManager

class RAGQuestionAnswering:
    """
    RAG Question Answering chain with strict context grounding,
    citation tracking, session memory support, and fallback handling.
    """

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or VectorStoreManager()
        self.openai_api_key = settings.OPENAI_API_KEY

    def answer_question(
        self, 
        query: str, 
        session_history: Optional[List[Dict[str, str]]] = None,
        selected_doc_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieves relevant context, constructs grounded prompt, generates response,
        and formats explicit source file and page number citations.
        """
        if not query or not query.strip():
            return {
                "answer": "Please provide a valid question.",
                "citations": [],
                "retrieved_context": []
            }

        # 1. Retrieve top relevant chunks using Hybrid Search
        retrieved_chunks = self.vector_store.hybrid_search(
            query=query,
            top_k=4,
            doc_ids=selected_doc_ids
        )

        # 2. Check for empty context or low relevance fallback
        if not retrieved_chunks:
            return {
                "answer": "I cannot determine the answer from the provided documents.",
                "citations": [],
                "retrieved_context": []
            }

        # 3. Format citations and context block
        context_blocks = []
        citations = []
        seen_citations = set()

        for chunk in retrieved_chunks:
            doc_name = chunk.get("file_name", "Unknown")
            page_no = chunk.get("page_number", 1)
            
            cit_key = f"{doc_name}_p{page_no}"
            if cit_key not in seen_citations:
                seen_citations.add(cit_key)
                citations.append({
                    "document": doc_name,
                    "page": page_no,
                    "doc_id": chunk.get("doc_id")
                })

            context_blocks.append(
                f"--- Source Document: {doc_name} (Page {page_no}) ---\n{chunk['text']}\n"
            )

        context_str = "\n".join(context_blocks)

        # Format conversation history string
        history_str = ""
        if session_history:
            history_lines = []
            for msg in session_history[-4:]: # include last 4 turns
                role = "User" if msg.get("role") == "user" else "Assistant"
                history_lines.append(f"{role}: {msg.get('content', '')}")
            history_str = "\n".join(history_lines)

        # 4. Generate Answer using OpenAI if key is present, else Local Grounded Generator
        if self.openai_api_key and self.openai_api_key.strip():
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)

                system_prompt = (
                    "You are an expert AI Research Assistant. Answer the user's question strictly "
                    "using ONLY the provided document context below.\n"
                    "If the context does not contain sufficient information to answer the question, state: "
                    "\"I cannot determine the answer from the provided documents.\"\n"
                    "Do NOT extrapolate, hallucinate, or use external knowledge not present in the context.\n"
                    "Always mention the document file names and page numbers in your response."
                )

                user_prompt = f"Conversation History:\n{history_str}\n\nContext:\n{context_str}\n\nQuestion: {query}"

                response = client.chat.completions.create(
                    model=settings.LLM_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                answer_text = response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API call failed: {e}. Falling back to local grounded generator.")
                answer_text = self._synthesize_grounded_answer(query, retrieved_chunks, history_str)
        else:
            answer_text = self._synthesize_grounded_answer(query, retrieved_chunks, history_str)

        return {
            "answer": answer_text,
            "citations": citations,
            "retrieved_context": [
                {
                    "text": c["text"],
                    "file_name": c.get("file_name"),
                    "page_number": c.get("page_number")
                }
                for c in retrieved_chunks
            ]
        }

    def _synthesize_grounded_answer(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        history_str: str = ""
    ) -> str:
        """
        Local grounded synthesis engine when external API key is omitted or unreachable.
        Extracts relevant sentences directly from retrieved text segments with citations.
        """
        relevant_sentences = []
        used_sources = []

        query_terms = [t.lower() for t in query.split() if len(t) > 2]

        for chunk in chunks:
            text = chunk["text"]
            file_name = chunk.get("file_name", "Document")
            page_no = chunk.get("page_number", 1)
            
            sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 15]
            matching_sents = []

            for sent in sentences:
                sent_lower = sent.lower()
                if any(term in sent_lower for term in query_terms):
                    matching_sents.append(sent)

            if matching_sents:
                excerpt = ". ".join(matching_sents[:2]) + "."
                relevant_sentences.append(f"• According to **{file_name}** (Page {page_no}):\n  \"{excerpt}\"")
                used_sources.append(f"{file_name} (Page {page_no})")

        if not relevant_sentences:
            # Fallback to top chunk snippet if keyword extraction is sparse
            top_c = chunks[0]
            snippet = top_c["text"][:350].replace("\n", " ") + "..."
            fn = top_c.get("file_name", "Document")
            pg = top_c.get("page_number", 1)
            return f"Based on the retrieved context from **{fn}** (Page {pg}):\n\n\"{snippet}\""

        sources_summary = ", ".join(list(dict.fromkeys(used_sources)))
        body = "\n\n".join(relevant_sentences)

        return (
            f"Here are the details found regarding **\"{query}\"**:\n\n"
            f"{body}\n\n"
            f"**Sources Referenced:** {sources_summary}"
        )
