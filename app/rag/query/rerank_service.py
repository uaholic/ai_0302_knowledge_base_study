from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import RERANK_SUMMARY_CHAR_RATIO, RERANK_MAX_INPUT_TOKENS, RERANK_MIN_SUMMARY_CHARS, \
    RERANK_MAX_TOPK, RERANK_MIN_TOPK, RERANK_GAP_RATIO, RERANK_GAP_ABS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger


def validate_state(state) -> tuple[list[dict], list[dict], str]:
    rewritten_query = state.get("rewritten_query")
    rrf_chunks = state.get("rrf_chunks")
    web_search_docs = state.get("web_search_docs")
    if not rrf_chunks or len(rewritten_query) == 0:
        logger.error(f"{rewritten_query}问题对应的数据为空,无法进行下一步处理!")
        raise ValueError(f"{rewritten_query}问题对应数据为空,无法进行下一步处理!")
    if not web_search_docs or len(web_search_docs) == 0:
        logger.error(f"{rewritten_query}问题对应数据为空,无法进行下一步处理!")
        raise ValueError(f"{rewritten_query}问题对应数据为空,无法进行下一步处理!")
    if not rewritten_query:
        logger.error(f"{rewritten_query}问题对应数据为空,无法进行下一步处理!")
        raise ValueError(f"{rewritten_query}问题对应数据为空,无法进行下一步处理!")
    return rrf_chunks, web_search_docs, rewritten_query


def compression_answer(rewritten_query, answer, limit):
    chat_mode = llm_provider.chat()
    human_prompt = load_prompt("rerank_text_refine", question=rewritten_query, answer=answer, limit=limit)
    messages = [
        HumanMessage(content=human_prompt)
    ]
    chain = chat_mode | StrOutputParser()
    resp = chain.invoke(input=messages)
    return resp


def compute_score(rewritten_query: str, data_list: list[dict]) -> list[dict]:
    model = llm_provider.reranker_model()
    tokenizer = model.tokenizer
    query_tokens = tokenizer.encode(rewritten_query, add_special_tokens=False)
    query_tokens_number = len(query_tokens)
    query_list = []
    for data in data_list:
        answer = data.get("text")
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        answer_tokens_number = len(answer_tokens)

        if query_tokens_number + answer_tokens_number + 4 > RERANK_MAX_INPUT_TOKENS:
            limit = max(RERANK_MIN_SUMMARY_CHARS,
                        int((RERANK_MAX_INPUT_TOKENS - 4 - query_tokens_number) / RERANK_SUMMARY_CHAR_RATIO))
            answer = compression_answer(rewritten_query, answer, limit=limit)
        query_list.append([rewritten_query, answer])

    scores = llm_provider.reranker_model().compute_score(query_list, normalize=True)
    logger.info(f"{rewritten_query}问题对应的数据得分为:{scores}")
    for score, item in zip(scores, data_list):
        item['score'] = score
    data_list.sort(key=lambda x: x['score'], reverse=True)
    return data_list


def merge_answer(rrf_chunks, web_search_docs) -> list[dict]:
    rrf_list = [{
        "title": chunk['title'],
        "text": chunk['content'],
        "url": "",
        "type": "milvus",
        "score": chunk.get("score", 0)
    } for chunk in rrf_chunks]
    web_list = [{
        "title": doc['title'],
        "text": doc['snippet'],
        "url": doc['url'],
        "type": "web",
        "score": 0
    } for doc in web_search_docs]
    return rrf_list + web_list

def dynamic_cut_chunk_list(data_list) -> list[dict]:
    max_count =RERANK_MAX_TOPK
    min_count = RERANK_MIN_TOPK
    gap_rate = RERANK_GAP_RATIO
    gap_abs = RERANK_GAP_ABS
    max_count=min(max_count,len(data_list))
    top_k=max_count
    if max_count> min_count:
        for i in range(min_count-1,max_count-1):
            cur_score = data_list[i].get("score",0.0)
            next_score = data_list[i+1].get("score",0.0)
            abs_score = cur_score-next_score
            rate=abs_score/(cur_score+1e-7) # 防止当前score为0
            if abs_score>gap_abs or rate>gap_rate:
                top_k=i+1
                break
    logger.info(f"数据动态截取数量为:{top_k}")
    return data_list[:top_k]


def rerank_documents(state: QueryGraphState) -> QueryGraphState:
    """
    重排序服务：
    1. 合并 RRF 和 Web Search 的文档
    2. 使用 BGE Reranker 模型计算相关性得分
    3. 根据得分动态截断，智能截取 TopK
    4. 回写 reranked_docs
    """
    rrf_chunks, web_search_docs, rewritten_query = validate_state(state)
    data_list = merge_answer(rrf_chunks, web_search_docs)
    data_list = compute_score(rewritten_query, data_list)

    data_list = dynamic_cut_chunk_list(data_list)
    state["reranked_docs"]=data_list
    return state
