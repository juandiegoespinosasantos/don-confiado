class LLMUtils:

    MEMORY_STORE: dict = {}

    @staticmethod
    def _get_history(user_id: str):
       if user_id not in LLMUtils.MEMORY_STORE:
           LLMUtils.MEMORY_STORE[user_id] = []

       return LLMUtils.MEMORY_STORE[user_id]

    @staticmethod
    def _append_message(user_id: str, role: str, content: str) -> None:
       history = LLMUtils._get_history(user_id)
       history.append({"role": role, "content": content})

    @staticmethod
    def _history_as_text(user_id: str) -> str:
       lines = []

       for msg in LLMUtils._get_history(user_id):
           if msg.get("role") == "human":
               lines.append(f"Usuario: {msg.get('content', '')}")
           elif msg.get("role") == "ai":
               lines.append(f"Asistente: {msg.get('content', '')}")

       return "\n".join(lines)