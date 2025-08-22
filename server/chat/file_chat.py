from fastapi import Body, File, Form, UploadFile
from sse_starlette.sse import EventSourceResponse
from configs import (LLM_MODELS, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE,
                     CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE,
                     # 添加混合检索配置导入
                     DEFAULT_SEARCH_MODE, DEFAULT_DENSE_WEIGHT, 
                     DEFAULT_SPARSE_WEIGHT, DEFAULT_RRF_K)
from server.utils import (wrap_done, get_ChatOpenAI,
                        BaseResponse, get_prompt_template, get_temp_dir, run_in_thread_pool)
from server.knowledge_base.kb_cache.faiss_cache import memo_faiss_pool
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter
from server.knowledge_base.utils import KnowledgeFile
import json
import os
from pathlib import Path


def _parse_files_in_thread(
    files: List[UploadFile], # 上传的文件列表，每个文件是 UploadFile 类型。
    dir: str, # 指定的保存目录
    zh_title_enhance: bool, # 是否开启中文标题加强功能，布尔值。
    chunk_size: int, # 知识库中单段文本的最大长度
    chunk_overlap: int, # 知识库中相邻文本的重叠长度
):
    """
    通过多线程将上传的文件保存至指定目录，并对文件内容进行预处理，最后返回文件处理结果。
    生成器返回保存结果：[success or error, filename, msg, docs]
    """
    def parse_file(file: UploadFile) -> dict:
        '''
        接收一个 UploadFile 类型的参数，尝试读取文件内容并保存到指定目录。
        '''
        try:
            # 生成文件路径，检查目录是否存在，不存在则创建。
            filename = file.filename
            file_path = os.path.join(dir, filename)
            file_content = file.file.read()  # 读取上传文件的内容

            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            # 以二进制写模式打开文件，写入内容。
            with open(file_path, "wb") as f:
                f.write(file_content)
            # 使用 KnowledgeFile 类对文件进行进一步处理，包括文本提取和分块。
            kb_file = KnowledgeFile(filename=filename, knowledge_base_name="temp")
            kb_file.filepath = file_path
            docs = kb_file.file2text(zh_title_enhance=zh_title_enhance,
                                     chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap)
            return True, filename, f"成功上传文件 {filename}", docs
        except Exception as e:
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            return False, filename, msg, []

    params = [{"file": file} for file in files]
    # 为每个上传的文件生成一个参数字典，然后使用 run_in_thread_pool 函数并发执行文件处理，提高效率。
    for result in run_in_thread_pool(parse_file, params=params):
        # 生成器，逐一返回文件处理结果，包括是否成功、文件名、消息和文档列表。
        yield result


def upload_temp_docs(
    files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
    prev_id: str = Form(None, description="前知识库ID"),
    chunk_size: int = Form(CHUNK_SIZE, description="知识库中单段文本最大长度"),
    chunk_overlap: int = Form(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
    zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
) -> BaseResponse:
    '''
    将文件保存到临时目录，并进行向量化。
    返回临时目录名称作为ID，同时也是临时向量库的ID。
    '''
    # 如果提供了前知识库ID，从 memo_faiss_pool 中移除相关信息。
    if prev_id is not None:
        memo_faiss_pool.pop(prev_id)

    failed_files = []
    documents = []
    path, id = get_temp_dir(prev_id)
    # 调用 _parse_files_in_thread 函数处理上传的文件，收集处理结果。
    for success, file, msg, docs in _parse_files_in_thread(files=files,
                                                        dir=path,
                                                        zh_title_enhance=zh_title_enhance,
                                                        chunk_size=chunk_size,
                                                        chunk_overlap=chunk_overlap):
        if success:
            documents += docs
        else:
            failed_files.append({file: msg})
    # 对于处理成功的文件，将其文档内容添加到向量库。
    with memo_faiss_pool.load_vector_store(id).acquire() as vs:
        vs.add_documents(documents)
    # 返回包含临时向量库ID和失败文件列表的响应
    return BaseResponse(data={"id": id, "failed_files": failed_files})


async def file_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                    knowledge_id: str = Body(..., description="临时知识库ID"),
                    top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                    score_threshold: float = Body(SCORE_THRESHOLD, description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右", ge=0, le=2),
                    history: List[History] = Body([],
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
                    max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
                    prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                    
                    # 新增混合检索参数
                    search_mode: str = Body(DEFAULT_SEARCH_MODE, 
                                           description="检索模式: 'vector'(向量检索), 'bm25'(关键词检索), 'hybrid'(混合检索)",
                                           examples=["vector"]),  # 文件对话默认使用向量检索
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
    # 检查 knowledge_id 是否存在于 memo_faiss_pool 中
    if knowledge_id not in memo_faiss_pool.keys():
        return BaseResponse(code=404, msg=f"未找到临时知识库 {knowledge_id}，请先上传文件")
    # 将传入的历史对话列表转换成 History 类型的对象列表
    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        '''用于处理基于知识库的聊天对话'''
        # max_tokens 变量不是局部变量，它引用了外部作用域中的 max_tokens 变量。
        nonlocal max_tokens
        # 创建一个异步迭代器回调处理器的实例，这个实例将用于处理异步任务的回调。
        callback = AsyncIteratorCallbackHandler()
        # 检查 max_tokens 是否为整数且小于等于0
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None
        # 创建一个聊天模型的实例
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )
        
        # 验证搜索模式（文件对话的限制）
        from server.knowledge_base.kb_service.base import SearchMode
        if search_mode not in [SearchMode.VECTOR_ONLY, SearchMode.BM25_ONLY, SearchMode.HYBRID]:
            search_mode = SearchMode.VECTOR_ONLY
        
        # 对于临时文件，暂时只支持向量检索
        # 如果需要完整的混合检索，需要为临时知识库也实现BM25索引
        original_search_mode = search_mode
        if search_mode in [SearchMode.BM25_ONLY, SearchMode.HYBRID]:
            # 暂时降级为向量检索，并给出提示
            search_mode = SearchMode.VECTOR_ONLY
            yield json.dumps({
                "answer": f"注意：文件对话当前只支持向量检索模式。您选择的 {original_search_mode} 模式已自动切换为向量检索。\n\n",
                "search_mode": search_mode,
                "note": f"原选择模式 {original_search_mode} 已切换为 vector"
            }, ensure_ascii=False)

        # 创建一个嵌入函数适配器实例，这个实例用于将查询文本转换为嵌入向量。
        embed_func = EmbeddingsFunAdapter()
        # 异步获取查询文本的嵌入向量
        embeddings = await embed_func.aembed_query(query)
        # 使用上下文管理器获取 knowledge_id 对应的向量搜索实例
        with memo_faiss_pool.acquire(knowledge_id) as vs:
            # 进行相似度搜索，获取最相似的文档列表。
            docs = vs.similarity_search_with_score_by_vector(embeddings, k=top_k, score_threshold=score_threshold)
            # 从搜索结果中提取文档信息
            docs = [x[0] for x in docs]
        # 将所有找到的文档内容连接成一个长字符串
        context = "\n".join([doc.page_content for doc in docs])
        # 根据是否找到相关文档，选择使用空模板或指定的提示模板。
        if len(docs) == 0: ## 如果没有找到相关文档，使用Empty模板
            prompt_template = get_prompt_template("knowledge_base_chat", "empty")
        else:
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
        # 构造输入消息，将其格式化为模板格式。
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        # 创建聊天提示模板，包括历史消息和输入消息。
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        # 创建一个LLM链实例，用于处理聊天流程。
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # 开始一个在后台运行的异步任务，调用 chain.acall 方法处理聊天，并在完成时调用 callback.done 方法。
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )
        # 存储处理后的文档信息
        source_documents = []
        # 遍历 docs（之前由相似度搜索得到的文档列表）
        for inum, doc in enumerate(docs):
            # 获取文档的来源信息
            filename = doc.metadata.get("source")
            # 构造一个包含出处信息的文本块
            text = f"""出处 [{inum + 1}] [{filename}] \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        # 没有找到相关文档
        if len(source_documents) == 0:
            source_documents.append(f"""<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>""")

        if stream:
            # 异步迭代回调处理器中的数据，并逐个发送。
            async for token in callback.aiter():
                # 每次迭代生成的数据（token）被封装成JSON格式，并使用 yield 关键字返回，实现流式输出。
                yield json.dumps({"answer": token,
                                  "search_mode": search_mode,  # 添加实际使用的检索模式
                                  }, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            # 如果 stream 为 False，则同样通过异步迭代获取所有数据拼接成完整的回答，
            # 最后一次性返回整个回答和文档信息的JSON字符串。
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents,
                              "search_mode": search_mode,
                              },
                             ensure_ascii=False)
        # await task 等待之前启动的异步任务完成
        await task
    # 把 knowledge_base_chat_iterator() 生成的数据作为服务器发送事件（SSE）的响应返回给客户端
    return EventSourceResponse(knowledge_base_chat_iterator())