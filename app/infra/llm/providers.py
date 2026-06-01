from app.infra.config.providers import InfraConfig
from app.shared.model import get_llm_client


class LLMProvider:

    def chat(self, model_name: str = None, json_mode: bool = False):
        return get_llm_client(model_name, json_mode)

    def vision_chat(self, model_name: str = InfraConfig.llm.lv_model):
        return get_llm_client(model_name)
