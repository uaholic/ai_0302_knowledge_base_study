from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm.providers import llm_provider
from app.infra.persistence.history_repository import history_repository
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


def validate(session_id: str = None, original_query: str = None):
    if session_id is None or original_query is None:
        raise ValueError("session_id and original_query must be provided")


def build_history_text(history: list[dict]) -> str:
    text = ""
    for i, item in enumerate(history, start=1):
        text += (f"第:{i}条记录,类型:{"提问" if item['role'] == 'user' else "回答"},"
                 f"内容:{item['text'] if item['role'] == 'user' else item['rewritten_query']},"
                 f"关联主体:{item['item_names']}\n")
    logger.info(f"历史对话记录:{text}")
    return text


def deal_by_llm(history_messages_text, original_query) -> dict[str, Any]:
    llm = llm_provider.chat(json_mode=True)
    prompt_text = load_prompt("rewritten_query_and_itemnames", history_text=history_messages_text, query=original_query)
    parser = JsonOutputParser()
    chain = llm | parser
    msg = [
        HumanMessage(content=prompt_text)
    ]
    result = chain.invoke(msg)
    if "item_name" not in result:
        result["item_name"] = []
    if "rewritten_query" not in result:
        result["rewritten_query"] = original_query

    return result


def query_item_name_in_milvus(item_names: list[str]) -> dict[str, list[dict]]:
    result = {}
    for item_name in item_names:
        llm_result = llm_provider.embed_documents([item_name])
        dense_vector = llm_result["dense"][0]
        sparse_vector = llm_result["sparse"][0]
        ann_request_list = milvus_gateway.create_requests(dense_vector, sparse_vector, limit=10)
        search_result = milvus_gateway.hybrid_search(collection_name=milvus_gateway.item_collection_name,
                                                     reqs=ann_request_list, ranker_weights=(0.4, 0.6), norm_score=True,
                                                     limit=5)
        real_result = search_result[0]
        if not real_result or len(real_result) == 0:
            # 没有查到到数据
            logger.warning(f"模型提供的: {item_name} 没有检索到对应数据库数据! 跳过本次!!")
            continue
        """
                item_name -> llm 

                {item_name: [{item_name:数据库中的name,score:distance}....5]}

                [
                  [ -> real 
                    {
                       id: x,
                       distance: 0.6,
                       entity:{
                          item_name: 数据库中的name
                       }
                    },

                    {
                       id: x,
                       distance: 0.6,
                       entity:{
                          item_name: 数据库中的name
                       }
                    }
                    5个..... 20 ->  权重排名器  -> 5 
                  ]
                ]
                """
        # 变形
        result_items = [{"item_name": r.get("entity", {}).get("item_name"), "score": r.get("distance", 0)} for r in
                        real_result]
        result[item_name] = result_items
    return result


def select_item_list(query_result: dict[str, list[dict]]) -> dict[str, list[str]]:
    confirm_list = []
    option_list = []
    for item_name, result_items in query_result.items():
        result_items.sort(key=lambda x: x['score'], reverse=True)

        confirm_items = [item['item_name'] for item in result_items if item['score'] > 0.7]
        option_items = [item['item_name'] for item in result_items if 0.6 < item['score'] <= 0.7]

        if len(confirm_items) > 0:
            confirm_list.append(confirm_items[0])
            continue

        if len(option_items) > 0:
            option_list.extend(option_items[:2])
            continue

    return {
        "confirmed_item_name_list": confirm_list,
        "options_item_name_list": option_list
    }


def change_state(state, final_result, rewritten_query):
    confirmed_item_name_list = final_result.get('confirmed_item_name_list', [])
    options_item_name_list = final_result.get('options_item_name_list', [])

    if confirmed_item_name_list and len(confirmed_item_name_list) > 0:
        state["item_names"] = confirmed_item_name_list
        state["rewritten_query"] = rewritten_query
        return

    if options_item_name_list and len(options_item_name_list) > 0:
        state["rewritten_query"] = rewritten_query
        answer = f"您是要问{",".join(options_item_name_list)}相关内容吗？"
        state["answer"] = answer
        return

    state["rewritten_query"] = rewritten_query
    answer = f"没有找到与您的问题相关的内容，请重新提问"
    state["answer"] = answer


def save_state(state: QueryGraphState):
    history_repository.save_message(
        session_id=state['session_id'],
        role="user",
        text=state["original_query"],
        rewritten_query=state['rewritten_query'],
        item_names=state.get("item_names", [])
    )


def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    """
    意图确认服务：
    1. 结合历史对话提取商品名
    2. 将模糊问题改写为完整独立的精准问题
    3. 在 Milvus 向量库中进行混合搜索
    4. 根据评分高低自动对齐标准型号，或生成反问让用户手动确认
    5. 同步历史记录到 MongoDB
    """
    session_id = state["session_id"]
    original_query = state["original_query"]

    validate(session_id, original_query)

    histories = history_repository.list_recent(session_id=session_id)
    history_messages = [item for item in histories if item.get("item_names") and len(item.get('item_names')) > 0]

    # 假装存储数据
    history_repository.save_message(session_id=state['session_id'], role="user",
                                    text=state["original_query"], rewritten_query=state['rewritten_query'],
                                    item_names=state["item_names"], image_urls=state['image_urls'])

    history_messages_text = build_history_text(history_messages)

    item_name_result = deal_by_llm(history_messages_text, original_query)

    final_result = {}
    if len(item_name_result["item_names"]) > 0:
        query_result = query_item_name_in_milvus(item_name_result['item_names'])
        final_result = select_item_list(query_result)

    change_state(state, final_result, item_name_result['rewritten_query'])

    save_state(state)

    return state
