from fastapi import Body, Request
from sse_starlette.sse import EventSourceResponse
from fastapi.concurrency import run_in_threadpool
from configs import (LLM_MODELS, 
                     VECTOR_SEARCH_TOP_K, 
                     SCORE_THRESHOLD, 
                     TEMPERATURE,
                     USE_RERANKER,
                     RERANKER_MODEL,
                     RERANKER_MAX_LENGTH,
                     MODEL_PATH,
                     # 添加混合检索配置导入
                     DEFAULT_SEARCH_MODE, DEFAULT_DENSE_WEIGHT, 
                     DEFAULT_SPARSE_WEIGHT, DEFAULT_RRF_K)
from server.utils import wrap_done, get_ChatOpenAI
from server.utils import BaseResponse, get_prompt_template
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory
import json
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
from server.reranker.reranker import LangchainReranker
from server.utils import embedding_device

# 该函数设计用来通过与知识库的交互来回答用户的查询
async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                              score_threshold: float = Body(
                                  SCORE_THRESHOLD,
                                  description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                  ge=0,
                                  le=2
                              ),
                              history: List[History] = Body(
                                  [],
                                  description="历史对话",
                                  examples=[[
                                      {"role": "user",
                                       "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                      {"role": "assistant",
                                       "content": "虎头虎脑"}]]
                              ),
                              stream: bool = Body(False, description="流式输出"),
                              model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                              temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                              max_tokens: Optional[int] = Body(
                                  None,
                                  description="限制LLM生成Token数量，默认None代表模型最大值"
                              ),
                              prompt_name: str = Body(
                                  "default",
                                  description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
                              ),
                              request: Request = None,
                              
                              # 新增混合检索参数
                              search_mode: str = Body(DEFAULT_SEARCH_MODE, 
                                                     description="检索模式: 'vector'(向量检索), 'bm25'(关键词检索), 'hybrid'(混合检索)",
                                                     examples=["hybrid"]),
                              dense_weight: float = Body(DEFAULT_DENSE_WEIGHT, 
                                                        description="稠密检索权重（仅混合模式使用）", 
                                                        ge=0.0, le=1.0),
                              sparse_weight: float = Body(DEFAULT_SPARSE_WEIGHT, 
                                                         description="稀疏检索权重（仅混合模式使用）", 
                                                         ge=0.0, le=1.0),
                              rrf_k: int = Body(DEFAULT_RRF_K, 
                                               description="RRF算法参数（仅混合模式使用）", 
                                               ge=1),
                              ):
    # 获取指定名称的知识库服务。如果不存在这样的服务，函数将返回404错误代码和相应的错误消息。
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
    # 函数将历史对话记录history中的每一项转换为History对象
    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator(
            query: str, # 用户的查询
            top_k: int, # 返回的最相关文档数量上限
            history: Optional[List[History]], # 历史对话列表
            model_name: str = model_name, # 模型名称。
            prompt_name: str = prompt_name, # 提示模板名称
            # 新增混合检索参数
            search_mode: str = search_mode,
            dense_weight: float = dense_weight,
            sparse_weight: float = sparse_weight,
            rrf_k: int = rrf_k,
    ) -> AsyncIterable[str]:
        # 通过nonlocal关键字引用，允许在函数内部修改外层函数(knowledge_base_chat)的变量。
        nonlocal max_tokens
        callback = AsyncIteratorCallbackHandler()
        # 检查max_tokens是否为整数且小于等于0
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None
        # 创建聊天模型实例
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )
        
        # 验证和归一化混合检索参数
        from server.knowledge_base.kb_service.base import SearchMode
        if search_mode not in [SearchMode.VECTOR_ONLY, SearchMode.BM25_ONLY, SearchMode.HYBRID]:
            search_mode = SearchMode.VECTOR_ONLY
        
        if search_mode == SearchMode.HYBRID:
            total_weight = dense_weight + sparse_weight
            if abs(total_weight - 1.0) > 0.01:  # 允许小的浮点误差
                # 重新归一化权重
                dense_weight = dense_weight / total_weight
                sparse_weight = sparse_weight / total_weight
        
        # 异步执行run_in_threadpool，在后台线程池中运行search_docs函数，以避免阻塞事件循环。
        # 传递混合检索参数
        docs = await run_in_threadpool(search_docs,
                                       query=query,
                                       knowledge_base_name=knowledge_base_name,
                                       top_k=top_k,
                                       score_threshold=score_threshold,
                                       # 新增混合检索参数
                                       search_mode=search_mode,
                                       dense_weight=dense_weight,
                                       sparse_weight=sparse_weight,
                                       rrf_k=rrf_k)

        # 重新排序（Reranking）
        if USE_RERANKER:
            # 如果启用，则从配置中获取重新排序模型的路径
            reranker_model_path = MODEL_PATH["reranker"].get(RERANKER_MODEL,"BAAI/bge-reranker-large")
            print("-----------------model path------------------")
            print("### bge-reranker-large path : ", reranker_model_path)
            # 创建一个LangchainReranker实例，用于对检索到的文档进行重新排序以提高相关性。
            reranker_model = LangchainReranker(top_n=top_k,
                                            device=embedding_device(),
                                            max_length=RERANKER_MAX_LENGTH,
                                            model_name_or_path=reranker_model_path
                                            )
            print("### 原始的docs排序：", docs)
            # 重新排序后的文档通过reranker_model.compress_documents方法得到，
            # 此方法考虑了查询（query）的内容来优化文档的顺序。
            docs = reranker_model.compress_documents(documents=docs,
                                                     query=query)
            print("---------after rerank------------------")
            print(docs)
        # 根据检索到的文档，构建一个上下文（context），它是所有相关文档内容的拼接，将用作模型生成答案的背景信息。
        context = "\n".join([doc.page_content for doc in docs])

        if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
            prompt_template = get_prompt_template("knowledge_base_chat", "empty")
        else:
            # 根据prompt_name选择相应的提示模板
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        # 将选定的提示模板转换成History对象，并与历史对话记录一起构建成完整的聊天提示（chat_prompt）。
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        # 通过创建一个LLMChain实例
        chain = LLMChain(prompt=chat_prompt, llm=model)
        # 启动一个异步任务，该任务调用LLMChain的acall方法，传递用户的查询(query)和文档内容(context)给语言模型，以生成回答。
        # 任务完成时调用callback.done方法。
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )
        # 存储与用户查询相关的文档及其出处信息
        source_documents = []
        # 文档来源处理，遍历每个文档(doc)，构建包含文档出处链接和内容的字符串
        for inum, doc in enumerate(docs):
            filename = doc.metadata.get("source")
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
            base_url = request.base_url
            url = f"{base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)
        # 如果没有找到相关文档，则添加一条消息表示未找到相关文档。
        if len(source_documents) == 0:  # 没有找到相关文档
            source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")
        # 判断是否启用流式输出
        if stream:
            # 如果启用，则通过服务器发送事件实时返回生成的答案。
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token,
                                  "search_mode": search_mode,  # 添加检索模式信息
                                  "search_params": {  # 添加检索参数信息
                                      "dense_weight": dense_weight,
                                      "sparse_weight": sparse_weight,
                                      "rrf_k": rrf_k
                                  } if search_mode == SearchMode.HYBRID else {}
                                  }, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            # 如果未启用流式输出，则等待异步任务完成，收集全部生成的答案，然后一次性返回整个答案和文档出处信息。
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents,
                              "search_mode": search_mode,
                              "search_params": {
                                  "dense_weight": dense_weight,
                                  "sparse_weight": sparse_weight,
                                  "rrf_k": rrf_k
                              } if search_mode == SearchMode.HYBRID else {}
                              },
                             ensure_ascii=False)
        # 确保异步任务完成，即等待语言模型生成完整的答案。
        await task
    # 返回一个事件源响应，该响应通过knowledge_base_chat_iterator生成器函数实现了上述逻辑。
    return EventSourceResponse(knowledge_base_chat_iterator(query, top_k, history, model_name, prompt_name,
                                                            search_mode, dense_weight, sparse_weight, rrf_k))