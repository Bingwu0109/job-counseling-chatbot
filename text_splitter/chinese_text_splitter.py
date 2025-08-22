from langchain.text_splitter import CharacterTextSplitter
import re
from typing import List


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, sentence_size: int = 250, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf # 是否处理的是PDF文本
        self.sentence_size = sentence_size # 句子的最大长度，默认值为250个字符。

    def split_text1(self, text: str) -> List[str]:
        if self.pdf:
            # 合并连续的换行符
            text = re.sub(r"\n{3,}", "\n", text)
            # 将所有空白字符替换为单个空格
            text = re.sub('\s', ' ', text)
            # 删除连续的两个换行符
            text = text.replace("\n\n", "")
        # 义了一个分句模式，用于匹配中文文本中的句子结束符号。
        sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # del ：；
        sent_list = []
        # 遍历通过sent_sep_pattern.split(text)拆分得到的元素
        for ele in sent_sep_pattern.split(text):
            # 根据是否匹配结束符号或是否为空来决定是否将当前元素添加到结果列表sent_list中或者与上一个元素合并
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        # 返回拆分后的句子列表
        return sent_list

    def split_text(self, text: str) -> List[str]:   ##此处需要进一步优化逻辑
        if self.pdf:
            # 通过正则表达式替换来减少连续换行符、将所有空白字符替换为单个空格，并删除连续的双换行符，以便后续处理。
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)

        text = re.sub(r'([;；.!?。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符
        text = re.sub(r'(\.{6})([^"’”」』])', r"\1\n\2", text)  # 英文省略号
        text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
        # 移除文本末尾多余的换行符
        text = text.rstrip()
        # 将文本分割为列表，并进一步拆分长句子
        ls = [i for i in text.split("\n") if i]
        for ele in ls:
            # 检查其长度是否超过了设定的句子最大长度
            if len(ele) > self.sentence_size:
                ele1 = re.sub(r'([,，.]["’”」』]{0,2})([^,，.])', r'\1\n\2', ele)
                ele1_ls = ele1.split("\n")
                for ele_ele1 in ele1_ls:
                    if len(ele_ele1) > self.sentence_size:
                        ele_ele2 = re.sub(r'([\n]{1,}| {2,}["’”」』]{0,2})([^\s])', r'\1\n\2', ele_ele1)
                        ele2_ls = ele_ele2.split("\n")
                        for ele_ele2 in ele2_ls:
                            if len(ele_ele2) > self.sentence_size:
                                ele_ele3 = re.sub('( ["’”」』]{0,2})([^ ])', r'\1\n\2', ele_ele2)
                                ele2_id = ele2_ls.index(ele_ele2)
                                ele2_ls = ele2_ls[:ele2_id] + [i for i in ele_ele3.split("\n") if i] + ele2_ls[
                                                                                                       ele2_id + 1:]
                        ele_id = ele1_ls.index(ele_ele1)
                        ele1_ls = ele1_ls[:ele_id] + [i for i in ele2_ls if i] + ele1_ls[ele_id + 1:]

                id = ls.index(ele)
                ls = ls[:id] + [i for i in ele1_ls if i] + ls[id + 1:]
        return ls
