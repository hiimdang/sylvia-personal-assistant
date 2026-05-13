from langchain.agents import create_agent
from sylvia_core.utils.prompts import prompt_manager


class AgentBuilder:

    def __init__(self, main_llm, tools):
        self.main_llm = main_llm
        self.tools = tools
        self.prompt_manager = prompt_manager
        self.rag_chain = self._build_rag_chain()

    def _build_rag_chain(self):

        prompt = self.prompt_manager.get("sylvia_persona_prompt")

        system_role_prompt = prompt.split("---")[0].strip()
        core_rules_prompt = prompt.split("---")[1].strip()

        full_system_message = f"{system_role_prompt}\n\n{core_rules_prompt}"

        return create_agent(
            model=self.main_llm,
            tools=self.tools,
            system_prompt=full_system_message,
        )

    def get_rag_chain(self):
        return self.rag_chain