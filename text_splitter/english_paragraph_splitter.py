"""
英文段落分片器
基于段落边界进行分片，保持段落的逻辑结构完整性
"""

import re
from typing import List, Optional, Any
from langchain.text_splitter import TextSplitter
from langchain.docstore.document import Document


class EnglishParagraphSplitter(TextSplitter):
    """英文段落分片器，基于段落边界进行分片"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        paragraph_separator: str = "\n\n",
        preserve_structure: bool = True,
        min_paragraph_length: int = 50,
        max_paragraph_length: int = 2000,
        merge_short_paragraphs: bool = True,
        split_long_paragraphs: bool = True,
        keep_headings: bool = True,
        **kwargs: Any,
    ):
        """
        初始化英文段落分片器
        
        Args:
            chunk_size: 目标chunk大小
            chunk_overlap: chunk重叠大小
            paragraph_separator: 段落分隔符
            preserve_structure: 是否保持结构
            min_paragraph_length: 最小段落长度
            max_paragraph_length: 最大段落长度
            merge_short_paragraphs: 是否合并短段落
            split_long_paragraphs: 是否拆分长段落
            keep_headings: 是否保持标题
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.paragraph_separator = paragraph_separator
        self.preserve_structure = preserve_structure
        self.min_paragraph_length = min_paragraph_length
        self.max_paragraph_length = max_paragraph_length
        self.merge_short_paragraphs = merge_short_paragraphs
        self.split_long_paragraphs = split_long_paragraphs
        self.keep_headings = keep_headings
        
        # 标题模式（Markdown格式）
        self.heading_patterns = [
            r'^#{1,6}\s+.+$',  # Markdown标题
            r'^.+\n[=]{3,}$',  # 下划线标题
            r'^.+\n[-]{3,}$',  # 下划线标题
        ]
    
    def split_text(self, text: str) -> List[str]:
        """
        将文本按段落分割
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的文本块列表
        """
        if not text.strip():
            return []
        
        # 预处理文本
        text = self._preprocess_text(text)
        
        # 按段落分割
        paragraphs = self._split_into_paragraphs(text)
        
        # 识别和处理标题
        if self.keep_headings:
            paragraphs = self._process_headings(paragraphs)
        
        # 过滤和清理段落
        paragraphs = self._filter_paragraphs(paragraphs)
        
        # 处理段落长度
        if self.merge_short_paragraphs:
            paragraphs = self._merge_short_paragraphs(paragraphs)
        
        if self.split_long_paragraphs:
            paragraphs = self._split_long_paragraphs(paragraphs)
        
        # 将段落组合成chunks
        chunks = self._combine_paragraphs_into_chunks(paragraphs)
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 标准化行结束符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 清理多余的空白行，但保留段落分隔
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 清理行内多余空白
        lines = []
        for line in text.split('\n'):
            cleaned_line = re.sub(r'\s+', ' ', line).strip()
            lines.append(cleaned_line)
        
        return '\n'.join(lines)
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """将文本分割成段落"""
        # 使用指定的段落分隔符分割
        paragraphs = text.split(self.paragraph_separator)
        
        # 清理空段落
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def _process_headings(self, paragraphs: List[str]) -> List[str]:
        """处理标题，保持标题与内容的关联"""
        processed = []
        current_heading = ""
        
        for paragraph in paragraphs:
            if self._is_heading(paragraph):
                current_heading = paragraph
                processed.append(paragraph)
            else:
                # 如果有当前标题且段落不是标题，可以选择将标题信息加入段落
                if current_heading and self.preserve_structure:
                    # 在元数据中记录标题信息，而不是直接合并到内容中
                    # 这里简单地保持原样，实际应用中可能需要更复杂的处理
                    processed.append(paragraph)
                else:
                    processed.append(paragraph)
        
        return processed
    
    def _is_heading(self, text: str) -> bool:
        """判断文本是否为标题"""
        for pattern in self.heading_patterns:
            if re.match(pattern, text, re.MULTILINE):
                return True
        
        # 额外的启发式规则
        lines = text.split('\n')
        if len(lines) == 1:
            line = lines[0].strip()
            # 短且以大写字母开头，可能是标题
            if (len(line) < 100 and 
                line and line[0].isupper() and 
                not line.endswith('.') and
                not line.endswith(',') and
                not line.endswith(';')):
                return True
        
        return False
    
    def _filter_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """过滤和清理段落"""
        filtered = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            
            # 跳过过短的段落（除非是标题）
            if (len(paragraph) < self.min_paragraph_length and 
                not self._is_heading(paragraph)):
                continue
            
            # 跳过只包含特殊字符的段落
            if re.match(r'^[^\w\s]*$', paragraph):
                continue
            
            filtered.append(paragraph)
        
        return filtered
    
    def _merge_short_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """合并过短的段落"""
        if not paragraphs:
            return []
        
        merged = []
        current_group = []
        current_length = 0
        
        for paragraph in paragraphs:
            # 标题不参与合并
            if self._is_heading(paragraph):
                if current_group:
                    merged.append('\n\n'.join(current_group))
                    current_group = []
                    current_length = 0
                merged.append(paragraph)
                continue
            
            paragraph_length = len(paragraph)
            
            # 如果段落已经足够长，单独成组
            if paragraph_length >= self.min_paragraph_length * 2:
                if current_group:
                    merged.append('\n\n'.join(current_group))
                    current_group = []
                    current_length = 0
                merged.append(paragraph)
                continue
            
            # 检查是否可以加入当前组
            if (current_length + paragraph_length + 2 <= self.min_paragraph_length * 3):
                current_group.append(paragraph)
                current_length += paragraph_length + 2  # +2 for separator
            else:
                if current_group:
                    merged.append('\n\n'.join(current_group))
                current_group = [paragraph]
                current_length = paragraph_length
        
        # 处理最后一组
        if current_group:
            merged.append('\n\n'.join(current_group))
        
        return merged
    
    def _split_long_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """拆分过长的段落"""
        split_paragraphs = []
        
        for paragraph in paragraphs:
            if len(paragraph) <= self.max_paragraph_length:
                split_paragraphs.append(paragraph)
                continue
            
            # 标题不拆分
            if self._is_heading(paragraph):
                split_paragraphs.append(paragraph)
                continue
            
            # 尝试按句子拆分长段落
            splits = self._split_long_paragraph_by_sentences(paragraph)
            split_paragraphs.extend(splits)
        
        return split_paragraphs
    
    def _split_long_paragraph_by_sentences(self, paragraph: str) -> List[str]:
        """按句子拆分长段落"""
        # 简单的句子分割
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        splits = []
        current_split = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 如果添加当前句子会超过最大长度
            if (current_split and 
                len(current_split) + len(sentence) + 1 > self.max_paragraph_length):
                splits.append(current_split.strip())
                current_split = sentence
            else:
                if current_split:
                    current_split += " " + sentence
                else:
                    current_split = sentence
        
        if current_split:
            splits.append(current_split.strip())
        
        return splits if splits else [paragraph]
    
    def _combine_paragraphs_into_chunks(self, paragraphs: List[str]) -> List[str]:
        """将段落组合成合适大小的chunks"""
        if not paragraphs:
            return []
        
        chunks = []
        current_chunk_paragraphs = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            # 如果单个段落就超过chunk_size，则单独成块
            if paragraph_length > self.chunk_size:
                if current_chunk_paragraphs:
                    chunks.append('\n\n'.join(current_chunk_paragraphs))
                    current_chunk_paragraphs = []
                    current_length = 0
                chunks.append(paragraph)
                continue
            
            # 计算添加当前段落后的总长度
            separator_length = 2 if current_chunk_paragraphs else 0  # \n\n
            total_length = current_length + separator_length + paragraph_length
            
            # 如果添加当前段落会超过chunk_size
            if total_length > self.chunk_size and current_chunk_paragraphs:
                chunks.append('\n\n'.join(current_chunk_paragraphs))
                current_chunk_paragraphs = [paragraph]
                current_length = paragraph_length
            else:
                current_chunk_paragraphs.append(paragraph)
                current_length += separator_length + paragraph_length
        
        # 添加最后一个chunk
        if current_chunk_paragraphs:
            chunks.append('\n\n'.join(current_chunk_paragraphs))
        
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
                overlapped_chunk = overlap_text + "\n\n" + chunk
                overlapped_chunks.append(overlapped_chunk)
            else:
                overlapped_chunks.append(chunk)
        
        return overlapped_chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """从文本末尾获取指定大小的重叠内容"""
        if len(text) <= overlap_size:
            return text
        
        # 尝试在段落边界处截取重叠内容
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            # 从最后几个段落中获取重叠内容
            overlap_text = ""
            for i in range(len(paragraphs)-1, -1, -1):
                candidate = paragraphs[i] + ("\n\n" + overlap_text if overlap_text else "")
                if len(candidate) <= overlap_size:
                    overlap_text = candidate
                else:
                    break
            if overlap_text:
                return overlap_text
        
        # 如果找不到合适的段落边界，直接截取
        return text[-overlap_size:]
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.create_documents(texts, metadatas)


# 便捷函数
def create_english_paragraph_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    **kwargs
) -> EnglishParagraphSplitter:
    """创建英文段落分片器的便捷函数"""
    return EnglishParagraphSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )
