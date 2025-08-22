"""
滑动窗口分片器
使用滑动窗口策略进行分片，确保最大的检索覆盖率
"""

import re
from typing import List, Optional, Any
from langchain.text_splitter import TextSplitter
from langchain.docstore.document import Document


class SlidingWindowSplitter(TextSplitter):
    """滑动窗口分片器，使用重叠窗口策略确保最大覆盖率"""
    
    def __init__(
        self,
        chunk_size: int = 250,
        chunk_overlap: int = 50,
        window_size: int = 200,
        step_size: int = 100,
        min_window_size: int = 50,
        boundary_strategy: str = "sentence",
        preserve_sentences: bool = True,
        overlap_strategy: str = "symmetric",
        dynamic_sizing: bool = True,
        **kwargs: Any,
    ):
        """
        初始化滑动窗口分片器
        
        Args:
            chunk_size: 目标chunk大小（字符数）
            chunk_overlap: chunk重叠大小
            window_size: 窗口大小（字符数）
            step_size: 步长（重叠部分）
            min_window_size: 最小窗口大小
            boundary_strategy: 边界策略（sentence, word, character）
            preserve_sentences: 是否保持句子完整性
            overlap_strategy: 重叠策略（symmetric, forward, backward）
            dynamic_sizing: 是否动态调整窗口大小
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.window_size = window_size
        self.step_size = step_size
        self.min_window_size = min_window_size
        self.boundary_strategy = boundary_strategy
        self.preserve_sentences = preserve_sentences
        self.overlap_strategy = overlap_strategy
        self.dynamic_sizing = dynamic_sizing
        
        # 验证参数
        if self.step_size >= self.window_size:
            raise ValueError("step_size must be less than window_size for overlap")
        
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
    
    def split_text(self, text: str) -> List[str]:
        """
        使用滑动窗口策略分割文本
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的文本块列表
        """
        if not text.strip():
            return []
        
        # 预处理文本
        text = self._preprocess_text(text)
        
        # 根据边界策略准备分割单位
        if self.boundary_strategy == "character":
            chunks = self._split_by_character(text)
        elif self.boundary_strategy == "word":
            chunks = self._split_by_word(text)
        elif self.boundary_strategy == "sentence":
            chunks = self._split_by_sentence(text)
        else:
            raise ValueError(f"Unknown boundary strategy: {self.boundary_strategy}")
        
        # 去重和清理
        chunks = self._deduplicate_and_clean(chunks)
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 标准化空白字符
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _split_by_character(self, text: str) -> List[str]:
        """按字符级别的滑动窗口分割"""
        chunks = []
        text_length = len(text)
        
        if text_length <= self.window_size:
            return [text]
        
        position = 0
        while position < text_length:
            # 确定当前窗口的结束位置
            end_position = min(position + self.window_size, text_length)
            
            # 获取当前窗口的文本
            window_text = text[position:end_position]
            
            # 如果窗口太小，跳过
            if len(window_text.strip()) < self.min_window_size:
                break
            
            chunks.append(window_text.strip())
            
            # 移动到下一个位置
            position += self.step_size
            
            # 如果剩余文本很少，直接处理完
            if text_length - position < self.min_window_size:
                if position < text_length:
                    remaining_text = text[position:].strip()
                    if remaining_text and len(remaining_text) >= self.min_window_size:
                        chunks.append(remaining_text)
                break
        
        return chunks
    
    def _split_by_word(self, text: str) -> List[str]:
        """按单词级别的滑动窗口分割"""
        words = text.split()
        if not words:
            return []
        
        chunks = []
        total_words = len(words)
        
        # 估算每个窗口的单词数（基于平均单词长度）
        avg_word_length = sum(len(word) for word in words) / len(words)
        words_per_window = max(1, int(self.window_size / (avg_word_length + 1)))
        words_per_step = max(1, int(self.step_size / (avg_word_length + 1)))
        
        position = 0
        while position < total_words:
            # 确定当前窗口的单词范围
            end_position = min(position + words_per_window, total_words)
            
            # 获取当前窗口的单词
            window_words = words[position:end_position]
            window_text = ' '.join(window_words)
            
            # 检查窗口大小
            if len(window_text.strip()) < self.min_window_size:
                break
            
            # 如果启用动态调整，根据实际字符数调整
            if self.dynamic_sizing:
                window_text = self._adjust_window_size(window_text, words, position)
            
            chunks.append(window_text.strip())
            
            # 移动到下一个位置
            position += words_per_step
            
            # 处理剩余文本
            if total_words - position < words_per_step:
                if position < total_words:
                    remaining_words = words[position:]
                    remaining_text = ' '.join(remaining_words).strip()
                    if remaining_text and len(remaining_text) >= self.min_window_size:
                        chunks.append(remaining_text)
                break
        
        return chunks
    
    def _split_by_sentence(self, text: str) -> List[str]:
        """按句子级别的滑动窗口分割"""
        # 分割成句子
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        chunks = []
        total_sentences = len(sentences)
        
        position = 0
        while position < total_sentences:
            # 构建当前窗口
            current_window = []
            current_length = 0
            
            # 添加句子直到达到窗口大小
            for i in range(position, total_sentences):
                sentence = sentences[i]
                sentence_length = len(sentence)
                
                # 检查是否超过窗口大小
                if current_length + sentence_length > self.window_size and current_window:
                    break
                
                current_window.append(sentence)
                current_length += sentence_length + 1  # +1 for space
            
            # 如果窗口为空或太小，跳过
            if not current_window:
                break
            
            window_text = ' '.join(current_window).strip()
            if len(window_text) >= self.min_window_size:
                chunks.append(window_text)
            
            # 计算下一个位置
            if self.preserve_sentences:
                # 按句子数量步进
                sentences_per_step = max(1, len(current_window) // 2)
                position += sentences_per_step
            else:
                # 尝试按字符步长步进
                step_sentences = self._calculate_sentence_step(sentences, position, self.step_size)
                position += max(1, step_sentences)
            
            # 处理剩余句子
            if total_sentences - position <= 2:  # 剩余句子很少
                if position < total_sentences:
                    remaining_sentences = sentences[position:]
                    remaining_text = ' '.join(remaining_sentences).strip()
                    if remaining_text and len(remaining_text) >= self.min_window_size:
                        chunks.append(remaining_text)
                break
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 简单的句子分割
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # 清理和过滤句子
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:  # 忽略过短的片段
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _adjust_window_size(self, window_text: str, words: List[str], position: int) -> str:
        """动态调整窗口大小"""
        current_length = len(window_text)
        
        # 如果当前窗口太小，尝试添加更多单词
        if current_length < self.window_size * 0.8:
            additional_words = []
            word_pos = position + len(window_text.split())
            
            while (word_pos < len(words) and 
                   current_length < self.window_size):
                word = words[word_pos]
                if current_length + len(word) + 1 > self.window_size:
                    break
                additional_words.append(word)
                current_length += len(word) + 1
                word_pos += 1
            
            if additional_words:
                window_text += ' ' + ' '.join(additional_words)
        
        # 如果窗口太大，尝试删除一些单词
        elif current_length > self.window_size:
            words_in_window = window_text.split()
            while len(' '.join(words_in_window)) > self.window_size and words_in_window:
                words_in_window.pop()
            window_text = ' '.join(words_in_window)
        
        return window_text
    
    def _calculate_sentence_step(self, sentences: List[str], position: int, step_size: int) -> int:
        """计算按字符步长对应的句子数"""
        accumulated_length = 0
        step_sentences = 0
        
        for i in range(position, len(sentences)):
            sentence_length = len(sentences[i]) + 1  # +1 for space
            if accumulated_length + sentence_length > step_size:
                break
            accumulated_length += sentence_length
            step_sentences += 1
        
        return max(1, step_sentences)
    
    def _deduplicate_and_clean(self, chunks: List[str]) -> List[str]:
        """去重和清理chunks"""
        seen = set()
        cleaned_chunks = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            
            # 跳过空或过短的chunk
            if not chunk or len(chunk) < self.min_window_size:
                continue
            
            # 简单的去重（基于内容哈希）
            chunk_hash = hash(chunk)
            if chunk_hash in seen:
                continue
            
            seen.add(chunk_hash)
            cleaned_chunks.append(chunk)
        
        # 根据重叠策略调整
        if self.overlap_strategy == "symmetric":
            # 默认行为，已经包含前向和后向重叠
            pass
        elif self.overlap_strategy == "forward":
            # 只保留前向重叠
            cleaned_chunks = self._apply_forward_overlap(cleaned_chunks)
        elif self.overlap_strategy == "backward":
            # 只保留后向重叠
            cleaned_chunks = self._apply_backward_overlap(cleaned_chunks)
        
        return cleaned_chunks
    
    def _apply_forward_overlap(self, chunks: List[str]) -> List[str]:
        """应用前向重叠策略"""
        if len(chunks) <= 1:
            return chunks
        
        # 为每个chunk添加来自下一个chunk的内容
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                overlap_text = self._get_overlap_text(next_chunk, self.chunk_overlap, from_start=True)
                if overlap_text:
                    enhanced_chunk = chunk + " " + overlap_text
                    overlapped_chunks.append(enhanced_chunk)
                else:
                    overlapped_chunks.append(chunk)
            else:
                overlapped_chunks.append(chunk)
        
        return overlapped_chunks
    
    def _apply_backward_overlap(self, chunks: List[str]) -> List[str]:
        """应用后向重叠策略"""
        if len(chunks) <= 1:
            return chunks
        
        # 为每个chunk添加来自前一个chunk的内容
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_chunk = chunks[i - 1]
                overlap_text = self._get_overlap_text(prev_chunk, self.chunk_overlap, from_start=False)
                if overlap_text:
                    enhanced_chunk = overlap_text + " " + chunk
                    overlapped_chunks.append(enhanced_chunk)
                else:
                    overlapped_chunks.append(chunk)
            else:
                overlapped_chunks.append(chunk)
        
        return overlapped_chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int, from_start: bool = False) -> str:
        """从文本获取重叠内容"""
        if len(text) <= overlap_size:
            return text
        
        if from_start:
            # 从开始获取
            end_pos = overlap_size
            # 尝试在单词边界处截取
            space_pos = text.find(' ', end_pos)
            if space_pos != -1 and space_pos - end_pos < 20:  # 不要偏差太远
                return text[:space_pos]
            else:
                return text[:overlap_size]
        else:
            # 从结尾获取
            start_pos = len(text) - overlap_size
            # 尝试在单词边界处截取
            space_pos = text.find(' ', start_pos)
            if space_pos != -1:
                return text[space_pos + 1:]
            else:
                return text[-overlap_size:]
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.create_documents(texts, metadatas)


# 便捷函数
def create_sliding_window_splitter(
    chunk_size: int = 250,
    chunk_overlap: int = 50,
    **kwargs
) -> SlidingWindowSplitter:
    """创建滑动窗口分片器的便捷函数"""
    return SlidingWindowSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )
