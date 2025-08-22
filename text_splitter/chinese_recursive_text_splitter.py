import re
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


def _split_text_with_regex_from_end(
        text: str, # 表示要分割的文本
        separator: str, # 分割文本的分隔符
        keep_separator: bool # 否在返回的列表中保留分隔符
) -> List[str]:
    # 如果分隔符非空，就执行分割逻辑；
    if separator:
        # 在保留分隔符的情况下
        if keep_separator:
            # 使用正则表达式 re.split(f"({separator})", text) 来分割文本，
            # 其中 {separator} 被插入到一个捕获组中（由圆括号定义）。这样做的目的是在结果列表中保留分隔符。
            _splits = re.split(f"({separator})", text)
            # 使用 zip(_splits[0::2], _splits[1::2]) 将文本片段和相应的分隔符组合起来。
            # _splits[0::2] 是偶数索引项（不包含分隔符的文本部分），而 _splits[1::2] 是奇数索引项（分隔符）。
            # 这两者交替组合，得到一个元组列表，每个元组包含文本和其后的分隔符。
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            # 如果 _splits 的长度是奇数，意味着最后一部分文本后面没有分隔符。
            if len(_splits) % 2 == 1:
                # 将最后一个元素（_splits[-1:]）直接添加到 splits 列表中
                splits += _splits[-1:]
            # splits = [_splits[0]] + splits
        else:
            # 直接分割文本，这将不保留分隔符。
            splits = re.split(separator, text)
    else:
        # 否则，将文本分割为单个字符的列表
        splits = list(text)
    # 移除任何空字符串，并返回结果列表。
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    '''该类专门用于处理中文文本的递归分割，可以处理多种分隔符，并支持正则表达式。'''
    def __init__(
            self,
            # 允许指定一个分隔符列表。如果没有提供（默认为 None），则会使用预定义的分隔符列表。
            separators: Optional[List[str]] = None,
            # 用于指定在分割后的文本中是否保留分隔符。默认为 True，意味着保留分隔符。
            keep_separator: bool = True,
            # 指明提供的分隔符是否应当被视为正则表达式。默认为 True，表明分隔符是正则表达式。
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        # 如果调用时提供了 separators 参数，就使用这个参数值；否则，使用默认的分隔符列表。
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s"
        ]
        # 设置是否将分隔符视为正则表达式
        self._is_separator_regex = is_separator_regex

    def _split_text(self,
                    text: str, # 表示要分割的文本字符串
                    separators: List[str] # 分隔符的列表，这些分隔符将用于递归分割文本。
                    ) -> List[str]:
        """Split incoming text and return chunks.
           递归地分割给定的文本并返回文本块的列表"""
        # 储最终的文本块
        final_chunks = []
        # 选择 separators 列表中的最后一个分隔符作为起始点
        separator = separators[-1]
        new_separators = []
        # 遍历 separators 列表来找到第一个可以匹配文本的分隔符
        for i, _s in enumerate(separators):
            # 根据 self._is_separator_regex 确定分隔符是否应当作为正则表达式处理。
            # 如果不是，使用 re.escape 来转义分隔符。
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break
        #  调用 split_text_with_regex_from_end 函数来分割文本，使用找到的分隔符。这里考虑了是否保留分隔符的选项。
        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        # 递归分割较长的文本块，对于分割后的每一个文本块 s
        for s in splits:
            # 如果 s 的长度小于设定的分块大小 (self._chunk_size)，则将其添加到 _good_splits 列表中。
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                # 否则，将 _good_splits 中的文本块合并并添加到 final_chunks 中，然后清空 _good_splits。
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                # 如果没有更多的分隔符可用，直接将当前的 s 添加到 final_chunks。
                if not new_separators:
                    final_chunks.append(s)
                else:
                    # 如果还有分隔符，递归地对 s 应用 _split_text 方法。
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        # 在处理完所有分割后，如果 _good_splits 中还有未处理的文本块，将它们合并并添加到 final_chunks。
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        # 对 final_chunks 中的每个文本块进行清理，移除多余的换行符，并确保文本块不为空。
        # 最后，返回经过清理和处理的文本块列表。
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip()!=""]


if __name__ == "__main__":
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=50,
        chunk_overlap=0
    )
    ls = [
        """中国对外贸易形势报告（75页）。前 10 个月，一般贸易进出口 19.5 万亿元，增长 25.1%， 比整体进出口增速高出 2.9 个百分点，占进出口总额的 61.7%，较去年同期提升 1.6 个百分点。其中，一般贸易出口 10.6 万亿元，增长 25.3%，占出口总额的 60.9%，提升 1.5 个百分点；进口8.9万亿元，增长24.9%，占进口总额的62.7%， 提升 1.8 个百分点。加工贸易进出口 6.8 万亿元，增长 11.8%， 占进出口总额的 21.5%，减少 2.0 个百分点。其中，出口增 长 10.4%，占出口总额的 24.3%，减少 2.6 个百分点；进口增 长 14.2%，占进口总额的 18.0%，减少 1.2 个百分点。此外， 以保税物流方式进出口 3.96 万亿元，增长 27.9%。其中，出 口 1.47 万亿元，增长 38.9%；进口 2.49 万亿元，增长 22.2%。前三季度，中国服务贸易继续保持快速增长态势。服务 进出口总额 37834.3 亿元，增长 11.6%；其中服务出口 17820.9 亿元，增长 27.3%；进口 20013.4 亿元，增长 0.5%，进口增 速实现了疫情以来的首次转正。服务出口增幅大于进口 26.8 个百分点，带动服务贸易逆差下降 62.9%至 2192.5 亿元。服 务贸易结构持续优化，知识密集型服务进出口 16917.7 亿元， 增长 13.3%，占服务进出口总额的比重达到 44.7%，提升 0.7 个百分点。 二、中国对外贸易发展环境分析和展望 全球疫情起伏反复，经济复苏分化加剧，大宗商品价格 上涨、能源紧缺、运力紧张及发达经济体政策调整外溢等风 险交织叠加。同时也要看到，我国经济长期向好的趋势没有 改变，外贸企业韧性和活力不断增强，新业态新模式加快发 展，创新转型步伐提速。产业链供应链面临挑战。美欧等加快出台制造业回迁计 划，加速产业链供应链本土布局，跨国公司调整产业链供应 链，全球双链面临新一轮重构，区域化、近岸化、本土化、 短链化趋势凸显。疫苗供应不足，制造业“缺芯”、物流受限、 运价高企，全球产业链供应链面临压力。 全球通胀持续高位运行。能源价格上涨加大主要经济体 的通胀压力，增加全球经济复苏的不确定性。世界银行今年 10 月发布《大宗商品市场展望》指出，能源价格在 2021 年 大涨逾 80%，并且仍将在 2022 年小幅上涨。IMF 指出，全 球通胀上行风险加剧，通胀前景存在巨大不确定性。""",
        ]
    # text = """"""
    for inum, text in enumerate(ls):
        print(inum)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            print(chunk)
