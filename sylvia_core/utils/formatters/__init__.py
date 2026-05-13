from langchain_core.documents import Document


def format_chat_history(chat_history) -> str:
    if not chat_history:
        return ""

    lines = []

    for entry in chat_history:
        entry_type = (
            entry.get("type")
            if isinstance(entry, dict)
            else getattr(entry, "type", None)
        )

        if entry_type != "message":
            continue

        sender = (
            entry.get("sender")
            if isinstance(entry, dict)
            else entry.sender
        )

        content = (
            entry.get("content", "")
            if isinstance(entry, dict)
            else entry.content
        ).strip()

        if not content:
            continue

        image_refs = (
            entry.get("image_refs")
            if isinstance(entry, dict)
            else entry.image_refs
        )

        suffix = " [has image]" if image_refs else ""
        lines.append(f"{sender}: {content}{suffix}")

    return "\n".join(lines)


def format_retrieved_context(docs: list[Document]) -> str:
    """Format retrieved context for the LLM prompt."""
    if not docs:
        return "No relevant context found."

    context_str = ""
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        retrieved_history = metadata.get("history", [])
        question = metadata.get("question", "")
        answer = metadata.get("answer", "")
        image_description = metadata.get("image_description")

        context_str += f"--- REFERENCE CONTEXT #{i + 1} ---\n"
        context_str += "STYLE EXEMPLAR (match tone and format, do not copy verbatim):\n"
        context_str += f"- User style sample: {question}\n"
        context_str += f"- Sylvia style sample: {answer}\n"

        context_str += "FACTUAL SUPPORT:\n"
        if retrieved_history:
            context_str += "Recent history from this conversation:\n"
            for entry in retrieved_history[-3:]:
                role = "User" if entry.get("role") == "user" else "Sylvia"
                content = entry.get("content", "")
                context_str += f"- {role}: {content}\n"

        if image_description:
            context_str += f"Related image description: {image_description}\n"

        context_str += f"--- END CONTEXT #{i + 1} ---\n\n"

    return context_str.strip()
