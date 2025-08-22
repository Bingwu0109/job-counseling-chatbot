from fastapi import Body
from configs import (DEFAULT_VS_TYPE, EMBEDDING_MODEL,
                     OVERLAP_SIZE,
                     logger, log_verbose, )
from server.knowledge_base.utils import (list_files_from_folder)
from sse_starlette import EventSourceResponse
import json
from server.knowledge_base.kb_service.base import KBServiceFactory
from typing import List, Optional
from server.knowledge_base.kb_summary.base import KBSummaryService
from server.knowledge_base.kb_summary.summary_chunk import SummaryAdapter
from server.utils import wrap_done, get_ChatOpenAI, BaseResponse
from configs import LLM_MODELS, TEMPERATURE
from server.knowledge_base.model.kb_document_model import DocumentWithVSId

def recreate_summary_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]), # 知识库名称
        allow_empty_kb: bool = Body(True), # 是否允许空知识库
        vs_type: str = Body(DEFAULT_VS_TYPE), # 向量存储类型
        embed_model: str = Body(EMBEDDING_MODEL), # 嵌入模型
        file_description: str = Body(''), # 文件描述
        model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"), # 模型名称
        temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
        max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
):
    """
    实现了知识库摘要的自动化创建和更新
    """

    def output():
        # 获取知识库服务对象kb
        kb = KBServiceFactory.get_service(knowledge_base_name, vs_type, embed_model)
        # 检查知识库是否存在，如果知识库不存在且不允许空知识库，那么就返回一个包含404错误码和错误信息的生成器对象。
        if not kb.exists() and not allow_empty_kb:
            yield {"code": 404, "msg": f"未找到知识库 ‘{knowledge_base_name}’"}
        else:
            # 重新创建知识库
            # 对象kb_summary，用于管理知识库摘要
            kb_summary = KBSummaryService(knowledge_base_name, embed_model)
            # 删除旧的摘要信息
            kb_summary.drop_kb_summary()
            # 创建新的摘要
            kb_summary.create_kb_summary()
            # 创建llm实例
            llm = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # 创建reduce_llm实例
            reduce_llm = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # 基于上一步创建的LLM实例，生成一个文本摘要适配器summary。
            summary = SummaryAdapter.form_summary(llm=llm,
                                                  reduce_llm=reduce_llm,
                                                  overlap_size=OVERLAP_SIZE)
            # 获取知识库中的文件列表
            files = list_files_from_folder(knowledge_base_name)

            i = 0
            # 遍历文件列表，对每个文件进行处理。
            for i, file_name in enumerate(files):
                # 获取文件中的文档信息
                doc_infos = kb.list_docs(file_name=file_name)
                # 调用文本摘要适配器summary的summarize方法，为文件生成摘要。
                docs = summary.summarize(file_description=file_description,
                                         docs=doc_infos)
                # 将生成的摘要信息添加到知识库摘要服务kb_summary中
                status_kb_summary = kb_summary.add_kb_summary(summary_combine_docs=docs)
                if status_kb_summary:
                    # 如果添加成功，则记录日志并通过生成器yield返回成功的消息。
                    logger.info(f"({i + 1} / {len(files)}): {file_name} 总结完成")
                    yield json.dumps({
                        "code": 200,
                        "msg": f"({i + 1} / {len(files)}): {file_name}",
                        "total": len(files),
                        "finished": i + 1,
                        "doc": file_name,
                    }, ensure_ascii=False)
                else:
                    # 如果添加失败，则记录错误日志并通过生成器返回错误消息。
                    msg = f"知识库'{knowledge_base_name}'总结文件‘{file_name}’时出错。已跳过。"
                    logger.error(msg)
                    yield json.dumps({
                        "code": 500,
                        "msg": msg,
                    })
                i += 1
    # 通过EventSourceResponse(output())返回一个事件源响应，实际上是以流的形式发送output函数生成的每一条消息。
    return EventSourceResponse(output())


def summary_file_to_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]), # 知识库名称
        file_name: str = Body(..., examples=["test.pdf"]), # 文件名
        allow_empty_kb: bool = Body(True), # 是否允许空知识库
        vs_type: str = Body(DEFAULT_VS_TYPE), # 向量存储类型
        embed_model: str = Body(EMBEDDING_MODEL), # 嵌入模型
        file_description: str = Body(''), # 文件描述
        model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
        temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
        max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
):
    """
    实现了将指定文件的摘要信息添加到知识库中的功能，通过大语言模型(LLM)生成文本摘要，并处理知识库摘要信息的更新。
    """
    def output():
        '''一个生成器函数，用于按需产生处理结果，这种设计允许函数以流的方式输出结果，适合长时间运行或逐步产生结果的任务。'''
        # 获取指定的知识库服务对象kb
        kb = KBServiceFactory.get_service(knowledge_base_name, vs_type, embed_model)
        # 检查知识库是否存在，如果知识库不存在并且不允许空知识库，则通过yield生成一个包含错误码404和错误信息的对象。
        if not kb.exists() and not allow_empty_kb:
            yield {"code": 404, "msg": f"未找到知识库 ‘{knowledge_base_name}’"}
        else:
            # 如果知识库存在，那么首先创建KBSummaryService对象kb_summary，这个对象用于管理知识库摘要的创建和更新。
            kb_summary = KBSummaryService(knowledge_base_name, embed_model)
            # 初始化或重置知识库摘要信息
            kb_summary.create_kb_summary()

            llm = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            reduce_llm = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # 创建一个文本摘要适配器
            summary = SummaryAdapter.form_summary(llm=llm,
                                                  reduce_llm=reduce_llm,
                                                  overlap_size=OVERLAP_SIZE)
            # 获取指定文件的文档信息
            doc_infos = kb.list_docs(file_name=file_name)
            # 基于文件描述和文档信息生成摘要
            docs = summary.summarize(file_description=file_description,
                                     docs=doc_infos)
            # 将生成的摘要信息添加到知识库中
            status_kb_summary = kb_summary.add_kb_summary(summary_combine_docs=docs)
            if status_kb_summary:
                # 如果添加成功，则记录日志并通过yield生成一个包含成功信息的JSON对象；
                logger.info(f" {file_name} 总结完成")
                yield json.dumps({
                    "code": 200,
                    "msg": f"{file_name} 总结完成",
                    "doc": file_name,
                }, ensure_ascii=False)
            else:
                # 如果失败，则记录错误日志并生成一个包含错误信息的JSON对象。
                msg = f"知识库'{knowledge_base_name}'总结文件‘{file_name}’时出错。已跳过。"
                logger.error(msg)
                yield json.dumps({
                    "code": 500,
                    "msg": msg,
                })
    # 返回一个EventSourceResponse对象，包含output函数生成的所有结果。
    return EventSourceResponse(output())


def summary_doc_ids_to_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]), # 知识库名称
        doc_ids: List = Body([], examples=[["uuid"]]), # 文档ID列表
        vs_type: str = Body(DEFAULT_VS_TYPE), # 向量存储类型
        embed_model: str = Body(EMBEDDING_MODEL), # 嵌入模型
        file_description: str = Body(''), # 文件描述
        model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
        temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
        max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
) -> BaseResponse:
    """
    实现了根据一组文档ID在指定知识库中生成这些文档的摘要信息的功能
    """
    # 获取知识库服务对象kb
    kb = KBServiceFactory.get_service(knowledge_base_name, vs_type, embed_model)
    if not kb.exists():
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}", data={})
    else:
        # 如果知识库存在，创建llm和reduce_llm实例
        llm = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        reduce_llm = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # 创建一个文本摘要适配器
        summary = SummaryAdapter.form_summary(llm=llm,
                                              reduce_llm=reduce_llm,
                                              overlap_size=OVERLAP_SIZE)
        # 根据提供的文档ID列表doc_ids获取对应的文档信息doc_infos。
        doc_infos = kb.get_doc_by_ids(ids=doc_ids)
        # 将获取到的文档信息doc_infos转换成DocumentWithVSId对象的列表doc_info_with_ids，以便包装每个文档的信息和ID。
        doc_info_with_ids = [DocumentWithVSId(**doc.dict(), id=with_id) for with_id, doc in zip(doc_ids, doc_infos)]
        # 基于文件描述和文档信息生成摘要
        docs = summary.summarize(file_description=file_description,
                                 docs=doc_info_with_ids)

        # 将生成的摘要信息docs转换成字典列表resp_summarize，
        # 这样做是为了将摘要信息格式化为JSON兼容的格式，便于作为HTTP响应的一部分返回。
        resp_summarize = [{**doc.dict()} for doc in docs]
        # 返回一个BaseResponse对象，其中包含200状态码、成功消息和包含摘要信息的数据部分。
        return BaseResponse(code=200, msg="总结完成", data={"summarize": resp_summarize})
