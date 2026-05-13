import asyncio
import json
from typing import Any, List, Dict

from langchain_classic.memory import ConversationBufferWindowMemory

from ..config import settings
from sylvia_core.utils.formatters import format_chat_history

class MemoryManager:
    def __init__(self, main_llm):
        self.main_llm = main_llm
        self.enable_interaction_log = settings.ENABLE_INTERACTION_LOG
        self.log_file = settings.LOG_FILE
        self._memories: Dict[str, ConversationBufferWindowMemory] = {}

    def get_memory(self, sender_name: str) -> ConversationBufferWindowMemory:
        if sender_name not in self._memories:
            self._memories[sender_name] = ConversationBufferWindowMemory(
                memory_key="chat_history", 
                return_messages=True, 
                llm=self.main_llm, 
                k=settings.MEMORY_WINDOW_SIZE # Use MEMORY_WINDOW_SIZE from settings
            )
        return self._memories[sender_name]

    async def log_interaction(
    self,
    question: str,
    answer: str,
    sender_name: str,
    chat_history: List | None,
    image_description: Any = None,
    ):
        if not self.enable_interaction_log or not self.log_file:
            return

        parsed_chat_history = []

        if chat_history:
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
                )

                image_refs = (
                    entry.get("image_refs")
                    if isinstance(entry, dict)
                    else entry.image_refs
                )
                
                parsed_chat_history.append(
                    {
                        "sender": sender,
                        "content": content.strip() if content else "",
                        "image_refs": image_refs,
                    }
                )
                
        log_entry = {
            "sender_name": sender_name,
            "question": question.strip() if question else "",
            "answer": (
                " ".join(answer).strip()
                if isinstance(answer, list)
                else answer.strip()
            ),
            "history": parsed_chat_history,
            # "image_description": image_description,
        }

        await asyncio.to_thread(self._perform_sync_log, log_entry)

    def _perform_sync_log(self, log_entry: Dict):
        """Synchronously writes a log entry to the file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
