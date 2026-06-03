from app.infra.config.providers import infra_config
from app.shared.model import get_llm_client, get_bge_m3_ef, generate_embeddings


class LLMProvider:

    def chat(self, model_name: str = None, json_mode: bool = False):
        return get_llm_client(model_name, json_mode)

    def vision_chat(self, model_name: str = infra_config.llm.lv_model):
        return get_llm_client(model_name)

    def embedding_mode(self):
        return get_bge_m3_ef()

    def embed_documents(self, documents: list[str]) -> dict[str, list]:
        """
            {
               dense: [[],[]],
               sparse: [{},{}]
            }
        :param documents:
        :return:
        """
        return generate_embeddings(documents)


llm_provider = LLMProvider()