"""
语义分块器
基于语义相似度进行智能分片，确保语义相关的内容在同一个chunk中
"""

import re
import numpy as np
from typing import List, Optional, Any, Tuple
from langchain.text_splitter import TextSplitter
from langchain.docstore.document import Document
from sklearn.metrics.pairwise import cosine_similarity


class SemanticChunkSplitter(TextSplitter):
    """语义分块器，基于语义相似度进行智能分片"""
    
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 75,
        embed_model: str = "bge-large-en-v1.5",
        similarity_threshold: float = 0.75,
        min_chunk_size: int = 100,
        max_chunk_size: int = 800,
        window_size: int = 3,
        stride: int = 1,
        boundary_strategy: str = "sentence",
        merge_threshold: float = 0.85,
        split_threshold: float = 0.6,
        **kwargs: Any,
    ):
        """
        初始化语义分块器
        
        Args:
            chunk_size: 目标chunk大小
            chunk_overlap: chunk重叠大小
            embed_model: 使用的embedding模型
            similarity_threshold: 语义相似度阈值
            min_chunk_size: 最小chunk大小
            max_chunk_size: 最大chunk大小
            window_size: 滑动窗口大小
            stride: 窗口步长
            boundary_strategy: 边界策略（sentence, paragraph, both）
            merge_threshold: 合并阈值
            split_threshold: 分割阈值
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.embed_model = embed_model
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.window_size = window_size
        self.stride = stride
        self.boundary_strategy = boundary_strategy
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold
        
        # 初始化embedding函数
        self.embedding_func = None
        self._init_embedding_func()
    
    def _init_embedding_func(self):
        """初始化embedding函数"""
        try:
            # 尝试使用项目中的embedding API
            from server.embeddings_api import embed_texts
            
            def embedding_func(texts: List[str]) -> np.ndarray:
                if isinstance(texts, str):
                    texts = [texts]
                try:
                    result = embed_texts(texts=texts, embed_model=self.embed_model, to_query=False)
                    return np.array(result.data)
                except Exception as e:
                    print(f"Warning: Failed to get embeddings: {e}")
                    # 返回随机向量作为备选方案
                    return np.random.random((len(texts), 768))
            
            self.embedding_func = embedding_func
        except ImportError:
            print("Warning: Could not import embedding API, using fallback")
            self.embedding_func = self._fallback_embedding
    
    def _fallback_embedding(self, texts: List[str]) -> np.ndarray:
        """备选embedding方案（基于简单的TF-IDF）"""
        from collections import Counter
        import math
        
        # 简单的TF-IDF实现作为备选方案
        def tokenize(text):
            return re.findall(r'\b\w+\b', text.lower())
        
        # 构建词汇表
        all_words = set()
        tokenized_texts = []
        for text in texts:
            tokens = tokenize(text)
            tokenized_texts.append(tokens)
            all_words.update(tokens)
        
        vocab = list(all_words)
        vocab_size = len(vocab)
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        
        # 计算TF-IDF向量
        vectors = []
        for tokens in tokenized_texts:
            tf = Counter(tokens)
            vector = np.zeros(vocab_size)
            
            for word, count in tf.items():
                if word in word_to_idx:
                    tf_score = count / len(tokens)
                    # 简化的IDF计算
                    idf_score = math.log(len(texts) / (1 + sum(1 for t in tokenized_texts if word in t)))
                    vector[word_to_idx[word]] = tf_score * idf_score
            
            vectors.append(vector)
        
        return np.array(vectors)
    
    def split_text(self, text: str) -> List[str]:
        """
        基于语义相似度进行文本分片
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的文本块列表
        """
        if not text.strip():
            return []
        
        # 预处理文本
        text = self._preprocess_text(text)
        
        # 根据边界策略分割文本单元
        units = self._split_into_units(text)
        
        if len(units) <= 1:
            return [text]
        
        # 计算语义相似度
        similarities = self._calculate_semantic_similarities(units)
        
        # 基于相似度进行分组
        groups = self._group_by_similarity(units, similarities)
        
        # 将分组转换为chunks
        chunks = self._groups_to_chunks(groups)
        
        # 后处理：合并过小的chunks，分割过大的chunks
        chunks = self._post_process_chunks(chunks)
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 标准化空白字符
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def _split_into_units(self, text: str) -> List[str]:
        """根据边界策略分割文本单元"""
        if self.boundary_strategy == "sentence":
            return self._split_into_sentences(text)
        elif self.boundary_strategy == "paragraph":
            return self._split_into_paragraphs(text)
        elif self.boundary_strategy == "both":
            # 先按段落分割，然后对长段落按句子分割
            paragraphs = self._split_into_paragraphs(text)
            units = []
            for para in paragraphs:
                if len(para) > self.chunk_size:
                    sentences = self._split_into_sentences(para)
                    units.extend(sentences)
                else:
                    units.append(para)
            return units
        else:
            raise ValueError(f"Unknown boundary strategy: {self.boundary_strategy}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """分割成句子"""
        # 简单的句子分割
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """分割成段落"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _calculate_semantic_similarities(self, units: List[str]) -> np.ndarray:
        """计算单元之间的语义相似度"""
        if len(units) <= 1:
            return np.array([[1.0]])
        
        # 获取embeddings
        embeddings = self.embedding_func(units)
        
        # 计算cosine相似度
        similarities = cosine_similarity(embeddings)
        
        return similarities
    
    def _group_by_similarity(self, units: List[str], similarities: np.ndarray) -> List[List[int]]:
        """基于相似度对单元进行分组"""
        n_units = len(units)
        if n_units <= 1:
            return [[0]] if n_units == 1 else []
        
        # 使用滑动窗口方法分组
        groups = []
        visited = set()
        
        for i in range(n_units):
            if i in visited:
                continue
            
            # 开始新的组
            current_group = [i]
            visited.add(i)
            current_size = len(units[i])
            
            # 向前扩展组
            for j in range(i + 1, n_units):
                if j in visited:
                    continue
                
                # 检查与当前组的相似度
                group_similarity = self._calculate_group_similarity(
                    current_group, j, similarities
                )
                
                # 检查大小限制
                new_size = current_size + len(units[j])
                
                if (group_similarity >= self.similarity_threshold and 
                    new_size <= self.max_chunk_size):
                    current_group.append(j)
                    visited.add(j)
                    current_size = new_size
                elif new_size > self.max_chunk_size:
                    # 如果大小超限，停止扩展
                    break
            
            groups.append(current_group)
        
        return groups
    
    def _calculate_group_similarity(self, group: List[int], candidate: int, 
                                  similarities: np.ndarray) -> float:
        """计算候选单元与组的相似度"""
        if not group:
            return 0.0
        
        # 计算与组内所有单元的平均相似度
        similarities_with_group = [similarities[candidate][idx] for idx in group]
        return np.mean(similarities_with_group)
    
    def _groups_to_chunks(self, groups: List[List[int]]) -> List[str]:
        """将分组转换为文本chunks"""
        chunks = []
        
        for group in groups:
            if not group:
                continue
            
            # 合并组内的单元
            group_texts = [units[i] for i in group]
            
            # 根据边界策略选择合并方式
            if self.boundary_strategy == "sentence":
                chunk_text = " ".join(group_texts)
            else:  # paragraph or both
                chunk_text = "\n\n".join(group_texts)
            
            chunks.append(chunk_text.strip())
        
        return chunks
    
    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """后处理chunks：合并过小的，分割过大的"""
        if not chunks:
            return []
        
        # 合并过小的chunks
        merged_chunks = self._merge_small_chunks(chunks)
        
        # 分割过大的chunks
        final_chunks = self._split_large_chunks(merged_chunks)
        
        # 添加重叠
        if self.chunk_overlap > 0 and len(final_chunks) > 1:
            final_chunks = self._add_overlap(final_chunks)
        
        return final_chunks
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """合并过小的chunks"""
        merged = []
        current_merged = ""
        
        for chunk in chunks:
            if len(chunk) < self.min_chunk_size:
                if current_merged:
                    separator = " " if self.boundary_strategy == "sentence" else "\n\n"
                    candidate = current_merged + separator + chunk
                    if len(candidate) <= self.max_chunk_size:
                        current_merged = candidate
                    else:
                        merged.append(current_merged)
                        current_merged = chunk
                else:
                    current_merged = chunk
            else:
                if current_merged:
                    merged.append(current_merged)
                    current_merged = ""
                merged.append(chunk)
        
        if current_merged:
            merged.append(current_merged)
        
        return merged
    
    def _split_large_chunks(self, chunks: List[str]) -> List[str]:
        """分割过大的chunks"""
        result = []
        
        for chunk in chunks:
            if len(chunk) <= self.max_chunk_size:
                result.append(chunk)
            else:
                # 递归分割大chunks
                sub_chunks = self._split_large_chunk(chunk)
                result.extend(sub_chunks)
        
        return result
    
    def _split_large_chunk(self, chunk: str) -> List[str]:
        """分割单个大chunk"""
        # 尝试按更小的单位分割
        if self.boundary_strategy == "paragraph":
            units = self._split_into_sentences(chunk)
        else:
            # 如果已经是句子级别，按固定大小分割
            return self._split_by_size(chunk)
        
        if len(units) <= 1:
            return self._split_by_size(chunk)
        
        # 重新应用语义分组，但使用更严格的大小限制
        similarities = self._calculate_semantic_similarities(units)
        groups = self._group_by_similarity_strict(units, similarities, self.max_chunk_size)
        
        result = []
        for group in groups:
            group_texts = [units[i] for i in group]
            if self.boundary_strategy == "sentence" or len(group) == 1:
                chunk_text = " ".join(group_texts)
            else:
                chunk_text = "\n\n".join(group_texts)
            result.append(chunk_text.strip())
        
        return result
    
    def _group_by_similarity_strict(self, units: List[str], similarities: np.ndarray, 
                                  max_size: int) -> List[List[int]]:
        """严格大小限制的相似度分组"""
        groups = []
        visited = set()
        
        for i in range(len(units)):
            if i in visited:
                continue
            
            current_group = [i]
            visited.add(i)
            current_size = len(units[i])
            
            for j in range(i + 1, len(units)):
                if j in visited:
                    continue
                
                new_size = current_size + len(units[j])
                if new_size > max_size:
                    break
                
                group_similarity = self._calculate_group_similarity(
                    current_group, j, similarities
                )
                
                if group_similarity >= self.similarity_threshold:
                    current_group.append(j)
                    visited.add(j)
                    current_size = new_size
            
            groups.append(current_group)
        
        return groups
    
    def _split_by_size(self, text: str) -> List[str]:
        """按固定大小分割文本"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # 尝试在单词边界处分割
            split_pos = text.rfind(' ', start, end)
            if split_pos > start:
                chunks.append(text[start:split_pos])
                start = split_pos + 1
            else:
                chunks.append(text[start:end])
                start = end
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """为chunks添加重叠内容"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # 从前一个chunk获取重叠内容
            overlap_text = self._get_overlap_text(prev_chunk, self.chunk_overlap)
            
            if overlap_text:
                separator = " " if self.boundary_strategy == "sentence" else "\n\n"
                overlapped_chunk = overlap_text + separator + current_chunk
                overlapped_chunks.append(overlapped_chunk)
            else:
                overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """从文本末尾获取指定大小的重叠内容"""
        if len(text) <= overlap_size:
            return text
        
        # 尝试在单词边界处截取
        start_pos = len(text) - overlap_size
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
def create_semantic_chunk_splitter(
    chunk_size: int = 300,
    chunk_overlap: int = 75,
    **kwargs
) -> SemanticChunkSplitter:
    """创建语义分块器的便捷函数"""
    return SemanticChunkSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )
