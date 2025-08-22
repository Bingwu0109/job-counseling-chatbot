"""
英文句子分片器
基于句子边界进行精确分片，保持句子的完整性
"""

import re
import nltk
from typing import List, Optional, Any
from langchain.text_splitter import TextSplitter
from langchain.docstore.document import Document


class EnglishSentenceSplitter(TextSplitter):
    """英文句子分片器，基于句子边界进行分片"""
    
    def __init__(
        self,
        chunk_size: int = 250,
        chunk_overlap: int = 50,
        keep_separator: bool = True,
        min_sentence_length: int = 15,
        max_sentence_length: int = 1000,
        preserve_quotes: bool = True,
        handle_abbreviations: bool = True,
        **kwargs: Any,
    ):
        """
        初始化英文句子分片器
        
        Args:
            chunk_size: 目标chunk大小
            chunk_overlap: chunk重叠大小  
            keep_separator: 是否保留分隔符
            min_sentence_length: 最小句子长度
            max_sentence_length: 最大句子长度
            preserve_quotes: 是否保持引用完整性
            handle_abbreviations: 是否处理缩写
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.keep_separator = keep_separator
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.preserve_quotes = preserve_quotes
        self.handle_abbreviations = handle_abbreviations
        
        # 初始化NLTK句子分词器
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        # 常见英文缩写列表
        self.abbreviations = {
            'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'vs.', 'etc.', 'i.e.', 'e.g.',
            'inc.', 'ltd.', 'corp.', 'co.', 'dept.', 'govt.', 'univ.', 'fig.',
            'no.', 'vol.', 'pp.', 'cf.', 'al.', 'st.', 'ave.', 'blvd.', 'rd.',
        }
    
    def split_text(self, text: str) -> List[str]:
        """
        将文本按句子分割
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的文本块列表
        """
        if not text.strip():
            return []
        
        # 预处理文本
        text = self._preprocess_text(text)
        
        # 使用NLTK进行句子分割
        sentences = self._split_into_sentences(text)
        
        # 过滤和清理句子
        sentences = self._filter_sentences(sentences)
        
        # 将句子组合成chunks
        chunks = self._combine_sentences_into_chunks(sentences)
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 标准化空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 处理引用
        if self.preserve_quotes:
            # 保护引用内容不被错误分割
            text = self._protect_quotes(text)
        
        # 处理缩写
        if self.handle_abbreviations:
            text = self._handle_abbreviations(text)
        
        return text.strip()
    
    def _protect_quotes(self, text: str) -> str:
        """保护引用内容"""
        # 简单的引用保护：将引用内的句号替换为特殊标记
        def replace_periods_in_quotes(match):
            quoted_text = match.group(0)
            return quoted_text.replace('.', '<!PERIOD!>')
        
        # 处理双引号
        text = re.sub(r'"[^"]*"', replace_periods_in_quotes, text)
        # 处理单引号
        text = re.sub(r"'[^']*'", replace_periods_in_quotes, text)
        
        return text
    
    def _handle_abbreviations(self, text: str) -> str:
        """处理缩写，避免在缩写处错误断句"""
        for abbr in self.abbreviations:
            # 将缩写中的句号替换为特殊标记
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            replacement = abbr.replace('.', '<!ABBREV!>')
            text = pattern.sub(replacement, text)
        
        return text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """使用NLTK将文本分割成句子"""
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
        except Exception:
            # 如果NLTK失败，使用正则表达式备用方案
            sentences = self._regex_sentence_split(text)
        
        # 恢复被保护的字符
        sentences = [self._restore_protected_chars(sent) for sent in sentences]
        
        return sentences
    
    def _regex_sentence_split(self, text: str) -> List[str]:
        """正则表达式备用句子分割方案"""
        # 基本的句子结束模式
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        return [sent.strip() for sent in sentences if sent.strip()]
    
    def _restore_protected_chars(self, text: str) -> str:
        """恢复被保护的字符"""
        text = text.replace('<!PERIOD!>', '.')
        text = text.replace('<!ABBREV!>', '.')
        return text
    
    def _filter_sentences(self, sentences: List[str]) -> List[str]:
        """过滤和清理句子"""
        filtered = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # 跳过过短或过长的句子
            if (len(sentence) < self.min_sentence_length or 
                len(sentence) > self.max_sentence_length):
                continue
            
            # 跳过只包含标点符号的句子
            if re.match(r'^[^\w]*$', sentence):
                continue
            
            filtered.append(sentence)
        
        return filtered
    
    def _combine_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        """将句子组合成合适大小的chunks"""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # 如果单个句子就超过chunk_size，则单独成块
            if sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_length = 0
                chunks.append(sentence)
                continue
            
            # 如果添加当前句子会超过chunk_size
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_length = sentence_length
            else:
                # 添加句子到当前chunk
                if current_chunk:
                    separator = " " if self.keep_separator else ""
                    current_chunk += separator + sentence
                    current_length += len(separator) + sentence_length
                else:
                    current_chunk = sentence
                    current_length = sentence_length
        
        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 处理重叠
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """为chunks添加重叠内容"""
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue
            
            # 从前一个chunk获取重叠内容
            prev_chunk = chunks[i-1]
            overlap_text = self._get_overlap_text(prev_chunk, self.chunk_overlap)
            
            if overlap_text:
                overlapped_chunk = overlap_text + " " + chunk
                overlapped_chunks.append(overlapped_chunk)
            else:
                overlapped_chunks.append(chunk)
        
        return overlapped_chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """从文本末尾获取指定大小的重叠内容"""
        if len(text) <= overlap_size:
            return text
        
        # 尝试在句子边界处截取重叠内容
        overlap_start = len(text) - overlap_size
        
        # 寻找最近的句子开始位置
        for i in range(overlap_start, len(text)):
            if i == 0 or text[i-1] in '.!?':
                return text[i:].strip()
        
        # 如果找不到句子边界，直接截取
        return text[-overlap_size:]
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.create_documents(texts, metadatas)


# 便捷函数
def create_english_sentence_splitter(
    chunk_size: int = 250,
    chunk_overlap: int = 50,
    **kwargs
) -> EnglishSentenceSplitter:
    """创建英文句子分片器的便捷函数"""
    return EnglishSentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )
