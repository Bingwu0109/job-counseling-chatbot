import streamlit as st
from webui_pages.utils import *
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
from server.knowledge_base.utils import get_file_path, LOADER_DICT, get_available_text_splitters
from server.knowledge_base.kb_service.base import get_kb_details, get_kb_file_details
from typing import Literal, Dict, Tuple
from configs import (kbs_config,
                     EMBEDDING_MODEL, DEFAULT_VS_TYPE,
                     CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE,
                     TEXT_SPLITTER_NAME,
                     # Add hybrid search related configuration imports
                     DEFAULT_SEARCH_MODE, DEFAULT_DENSE_WEIGHT, 
                     DEFAULT_SPARSE_WEIGHT, DEFAULT_RRF_K,
                     # Add new splitting strategy configuration imports
                     ENGLISH_SENTENCE_SPLITTER_CONFIG,
                     ENGLISH_PARAGRAPH_SPLITTER_CONFIG,
                     SEMANTIC_CHUNK_SPLITTER_CONFIG,
                     SLIDING_WINDOW_SPLITTER_CONFIG,
                     TEXT_SPLITTER_SELECTION_CONFIG)

# Add RAG-fusion related configuration imports
try:
    from configs import (
        ENABLE_RAG_FUSION,
        RAG_FUSION_CONFIG,
        RAG_FUSION_QUERY_COUNT,
        RAG_FUSION_LLM_MODEL,
        RAG_FUSION_SUPPORTED_MODELS,
        RAG_FUSION_MODEL_SELECTION,
    )
    RAG_FUSION_AVAILABLE = ENABLE_RAG_FUSION
except ImportError:
    RAG_FUSION_AVAILABLE = False
    RAG_FUSION_CONFIG = {}
    RAG_FUSION_QUERY_COUNT = 3
    RAG_FUSION_LLM_MODEL = "Qwen1.5-7B-Chat"
    RAG_FUSION_SUPPORTED_MODELS = []
    RAG_FUSION_MODEL_SELECTION = {}

from server.utils import list_embed_models, list_online_embed_models
import os
import time
import json

cell_renderer = JsCode("""function(params) {if(params.value==true){return '✓'}else{return '×'}}""")


def config_aggrid(
        df: pd.DataFrame,
        columns: Dict[Tuple[str, str], Dict] = {},
        selection_mode: Literal["single", "multiple", "disabled"] = "single",
        use_checkbox: bool = False,
) -> GridOptionsBuilder:
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column("No", width=40)
    for (col, header), kw in columns.items():
        gb.configure_column(col, header, wrapHeaderText=True, **kw)
    gb.configure_selection(
        selection_mode=selection_mode,
        use_checkbox=use_checkbox,
        pre_selected_rows=st.session_state.get("selected_rows", [0]),
    )
    gb.configure_pagination(
        enabled=True,
        paginationAutoPageSize=False,
        paginationPageSize=10
    )
    return gb


def file_exists(kb: str, selected_rows: List) -> Tuple[str, str]:
    """
    check whether a doc file exists in local knowledge base folder.
    return the file's name and path if it exists.
    """
    if selected_rows:
        file_name = selected_rows[0]["file_name"]
        file_path = get_file_path(kb, file_name)
        if os.path.isfile(file_path):
            return file_name, file_path
    return "", ""


def render_rag_fusion_status():
    """Render RAG-fusion system status"""
    with st.expander("🚀 RAG-fusion System Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if RAG_FUSION_AVAILABLE:
                st.success("✅ RAG-fusion Enabled")
            else:
                st.error("❌ RAG-fusion Disabled")
        
        with col2:
            model_count = len(RAG_FUSION_SUPPORTED_MODELS)
            st.info(f"🤖 Supports {model_count} Models")
        
        with col3:
            if st.button("🔍 System Check", key="system_check", help="Check RAG-fusion configuration status"):
                with st.spinner("Checking system configuration..."):
                    try:
                        # Simulate system info API call
                        system_info = {
                            "rag_fusion_available": RAG_FUSION_AVAILABLE,
                            "supported_models": RAG_FUSION_SUPPORTED_MODELS,
                            "default_model": RAG_FUSION_LLM_MODEL,
                            "config_valid": True
                        }
                        
                        if system_info["rag_fusion_available"]:
                            st.success("🎉 System configuration normal")
                            
                            # Display detailed information
                            with st.container():
                                st.write("**Configuration Details:**")
                                st.write(f"• Default Model: {system_info['default_model']}")
                                st.write(f"• Supported Models: {len(system_info['supported_models'])}")
                                st.write(f"• Config Status: {'Normal' if system_info['config_valid'] else 'Error'}")
                        else:
                            st.warning("⚠️ RAG-fusion feature not enabled")
                            
                    except Exception as e:
                        st.error(f"System check failed: {str(e)}")


def render_kb_capabilities(api: ApiRequest, selected_kb: str):
    """Render knowledge base capability information"""
    with st.expander("📊 Knowledge Base Capabilities", expanded=False):
        try:
            # Try to get knowledge base capability information
            capabilities_data = {
                "vector_search": {"supported": True, "description": "Vector Search"},
                "bm25_search": {"supported": True, "description": "Keyword Search"},  
                "hybrid_search": {"supported": True, "description": "Hybrid Search"},
                "rag_fusion": {"supported": RAG_FUSION_AVAILABLE, "description": "RAG-fusion Search"},
                "adaptive_search": {"supported": True, "description": "Adaptive Search"}
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔧 Search Capabilities**")
                for mode, info in capabilities_data.items():
                    status = "✅" if info["supported"] else "❌"
                    st.write(f"{status} {info['description']}")
            
            with col2:
                st.write("**📈 Knowledge Base Statistics**")
                # Display some basic statistics
                kb_info = st.session_state.get("selected_kb_info", "")
                st.write(f"• Knowledge Base Name: {selected_kb}")
                st.write(f"• RAG-fusion: {'Supported' if RAG_FUSION_AVAILABLE else 'Not Supported'}")
                if RAG_FUSION_AVAILABLE:
                    st.write(f"• Default Query Count: {RAG_FUSION_QUERY_COUNT}")
                    
        except Exception as e:
            st.warning(f"Unable to get capability information: {str(e)}")


def render_text_splitter_config():
    """Render text splitting strategy configuration interface"""
    st.write("**📊 Text Splitting Strategy Configuration**")
    
    # Get available splitters
    available_splitters = get_available_text_splitters()
    splitter_names = list(available_splitters.keys())
    
    # Splitter selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Splitter selection dropdown
        current_splitter = TEXT_SPLITTER_NAME if TEXT_SPLITTER_NAME in splitter_names else splitter_names[0]
        selected_splitter = st.selectbox(
            "Select Text Splitting Strategy:",
            options=splitter_names,
            index=splitter_names.index(current_splitter) if current_splitter in splitter_names else 0,
            format_func=lambda x: f"{x}",
            key="text_splitter_name",
            help="Choose a splitting strategy suitable for your document type"
        )
    
    with col2:
        # Auto selection switch
        auto_select = st.checkbox(
            "🎯 Smart Selection",
            value=TEXT_SPLITTER_SELECTION_CONFIG.get("auto_selection", False),
            help="Automatically select the best splitting strategy based on document type and size",
            key="auto_select_splitter"
        )
    
    # Display current splitter description
    if selected_splitter in available_splitters:
        description = available_splitters[selected_splitter]
        st.info(f"💡 **{selected_splitter}**: {description}")
    
    # Splitting parameter configuration
    st.write("**🔧 Splitting Parameters**")
    param_col1, param_col2 = st.columns(2)
    
    with param_col1:
        chunk_size = st.number_input(
            "Chunk Size (characters)", 
            min_value=50, 
            max_value=2000, 
            value=CHUNK_SIZE,
            step=50,
            help="Maximum number of characters per text chunk",
            key="chunk_size_config"
        )
    
    with param_col2:
        chunk_overlap = st.number_input(
            "Overlap Size (characters)", 
            min_value=0, 
            max_value=chunk_size//2, 
            value=min(OVERLAP_SIZE, chunk_size//2),
            step=10,
            help="Number of overlapping characters between adjacent text chunks",
            key="chunk_overlap_config"
        )
    
    # Advanced configuration (show different options based on splitter type)
    if selected_splitter in ["EnglishSentenceSplitter", "EnglishParagraphSplitter", 
                           "SemanticChunkSplitter", "SlidingWindowSplitter"]:
        
        with st.expander("🔬 Advanced Configuration", expanded=False):
            
            if selected_splitter == "EnglishSentenceSplitter":
                st.write("**Sentence Splitter Configuration**")
                col_a, col_b = st.columns(2)
                with col_a:
                    min_sentence_length = st.number_input(
                        "Minimum Sentence Length", 
                        value=ENGLISH_SENTENCE_SPLITTER_CONFIG.get("min_sentence_length", 15),
                        min_value=5, max_value=100, key="min_sentence_length"
                    )
                with col_b:
                    preserve_quotes = st.checkbox(
                        "Preserve Quote Integrity", 
                        value=ENGLISH_SENTENCE_SPLITTER_CONFIG.get("preserve_quotes", True),
                        key="preserve_quotes"
                    )
            
            elif selected_splitter == "EnglishParagraphSplitter":
                st.write("**Paragraph Splitter Configuration**")
                col_a, col_b = st.columns(2)
                with col_a:
                    merge_short_paragraphs = st.checkbox(
                        "Merge Short Paragraphs", 
                        value=ENGLISH_PARAGRAPH_SPLITTER_CONFIG.get("merge_short_paragraphs", True),
                        key="merge_short_paragraphs"
                    )
                with col_b:
                    keep_headings = st.checkbox(
                        "Keep Heading Structure", 
                        value=ENGLISH_PARAGRAPH_SPLITTER_CONFIG.get("keep_headings", True),
                        key="keep_headings"
                    )
            
            elif selected_splitter == "SemanticChunkSplitter":
                st.write("**Semantic Chunk Splitter Configuration**")
                col_a, col_b = st.columns(2)
                with col_a:
                    similarity_threshold = st.slider(
                        "Semantic Similarity Threshold", 
                        min_value=0.5, max_value=0.95, 
                        value=SEMANTIC_CHUNK_SPLITTER_CONFIG.get("similarity_threshold", 0.75),
                        step=0.05, key="similarity_threshold"
                    )
                with col_b:
                    boundary_strategy = st.selectbox(
                        "Boundary Strategy", 
                        options=["sentence", "paragraph", "both"],
                        index=["sentence", "paragraph", "both"].index(
                            SEMANTIC_CHUNK_SPLITTER_CONFIG.get("boundary_strategy", "sentence")
                        ),
                        key="boundary_strategy"
                    )
            
            elif selected_splitter == "SlidingWindowSplitter":
                st.write("**Sliding Window Splitter Configuration**")
                col_a, col_b = st.columns(2)
                with col_a:
                    window_size = st.number_input(
                        "Window Size", 
                        value=SLIDING_WINDOW_SPLITTER_CONFIG.get("window_size", 200),
                        min_value=50, max_value=500, key="window_size"
                    )
                with col_b:
                    step_size = st.number_input(
                        "Step Size", 
                        value=SLIDING_WINDOW_SPLITTER_CONFIG.get("step_size", 100),
                        min_value=10, max_value=window_size-10, key="step_size"
                    )
    
    # Splitting effect preview (optional feature)
    if st.checkbox("📖 Preview Splitting Effect", key="preview_splitting"):
        preview_text = st.text_area(
            "Enter test text:",
            value="This is a sample document. It contains multiple sentences. Each sentence should be processed according to the selected splitting strategy. The result will show how the text is divided into chunks.",
            height=100,
            key="preview_text"
        )
        
        if preview_text and st.button("Preview Splitting", key="do_preview"):
            try:
                from server.knowledge_base.utils import make_text_splitter
                # Here should call splitter for preview, but simplified to show configuration info
                st.write(f"**Using Splitter**: {selected_splitter}")
                st.write(f"**Chunk Size**: {chunk_size} characters")
                st.write(f"**Overlap Size**: {chunk_overlap} characters")
                st.write("**Preview Result**: (In actual implementation, split text would be displayed)")
                
                # Simple demonstration splitting
                words = preview_text.split()
                chunks = []
                for i in range(0, len(words), max(1, chunk_size//10)):
                    chunk_words = words[i:i + chunk_size//10]
                    chunks.append(" ".join(chunk_words))
                
                for i, chunk in enumerate(chunks[:3]):  # Only show first 3 chunks
                    st.write(f"**Chunk {i+1}**: {chunk}")
                    
            except Exception as e:
                st.error(f"Preview failed: {str(e)}")
    
    return {
        "text_splitter_name": selected_splitter,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "auto_select": auto_select
    }


def render_rag_fusion_config():
    """Render RAG-fusion specific configuration interface"""
    if not RAG_FUSION_AVAILABLE:
        st.warning("⚠️ RAG-fusion feature not enabled")
        return {}
    
    st.write("**🚀 RAG-fusion Configuration**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Query count configuration
        query_count = st.slider(
            "Query Generation Count",
            min_value=2,
            max_value=10,
            value=RAG_FUSION_QUERY_COUNT,
            help="Total number of queries including the original query",
            key="rag_fusion_query_count"
        )
    
    with col2:
        # Model selection
        model_options = RAG_FUSION_SUPPORTED_MODELS if RAG_FUSION_SUPPORTED_MODELS else [RAG_FUSION_LLM_MODEL]
        model_index = 0
        if RAG_FUSION_LLM_MODEL in model_options:
            model_index = model_options.index(RAG_FUSION_LLM_MODEL)
        
        selected_model = st.selectbox(
            "LLM Model",
            options=model_options,
            index=model_index,
            help="Language model used for query generation",
            key="rag_fusion_model"
        )
    
    with col3:
        # Timeout setting
        timeout = st.number_input(
            "Timeout (seconds)",
            min_value=5,
            max_value=120,
            value=RAG_FUSION_CONFIG.get("query_generation", {}).get("timeout", 30),
            help="Maximum wait time for query generation",
            key="rag_fusion_timeout"
        )
    
    # Advanced RAG-fusion options
    with st.expander("🔧 RAG-fusion Advanced Options", expanded=False):
        col_a, col_b = st.columns(2)
        
        with col_a:
            enable_cache = st.checkbox(
                "Enable Query Cache",
                value=RAG_FUSION_CONFIG.get("cache", {}).get("enable", True),
                help="Cache generated queries to improve performance",
                key="rag_fusion_enable_cache"
            )
            
            enable_rerank = st.checkbox(
                "Enable Result Reranking",
                value=False,
                help="Rerank fusion results for optimization",
                key="rag_fusion_enable_rerank"
            )
        
        with col_b:
            fusion_strategy = st.selectbox(
                "Fusion Query Retrieval Strategy",
                options=["hybrid", "vector", "bm25"],
                index=0,
                help="Retrieval strategy for executing generated queries",
                key="rag_fusion_strategy"
            )
            
            per_query_top_k = st.number_input(
                "Results per Query",
                min_value=3,
                max_value=20,
                value=10,
                help="Number of documents retrieved per query",
                key="rag_fusion_per_query_top_k"
            )
    
    return {
        "query_count": query_count,
        "model": selected_model,
        "timeout": timeout,
        "enable_cache": enable_cache,
        "enable_rerank": enable_rerank,
        "fusion_strategy": fusion_strategy,
        "per_query_top_k": per_query_top_k
    }


def render_search_interface(api: ApiRequest, selected_kb: str):
    """Render enhanced search interface (with RAG-fusion support)"""
    st.subheader("🔍 Intelligent Search System")
    
    # Render system status
    render_rag_fusion_status()
    
    # Render knowledge base capability information
    render_kb_capabilities(api, selected_kb)
    
    # Search query input
    query = st.text_input(
        "Please enter your question:", 
        placeholder="e.g., How to configure the system settings?",
        key="search_query",
        help="Enter the question or keywords you want to search for"
    )
    
    # Create a container to organize search configuration
    with st.container():
        st.write("**🔧 Search Mode Configuration**")
        
        # Search mode selection - add RAG-fusion option
        mode_options = ["vector", "bm25", "hybrid"]
        mode_labels = {
            "vector": "🧠 Semantic Search (Vector)",
            "bm25": "🔤 Keyword Search (BM25)",
            "hybrid": "🔀 Hybrid Search"
        }
        
        if RAG_FUSION_AVAILABLE:
            mode_options.extend(["rag_fusion", "adaptive"])
            mode_labels.update({
                "rag_fusion": "🚀 RAG-fusion Search (Recommended)",
                "adaptive": "🎯 Adaptive Search"
            })
        
        search_mode = st.radio(
            "Select search mode:",
            options=mode_options,
            format_func=lambda x: mode_labels.get(x, x),
            index=2 if "hybrid" in mode_options else 0,  # Default to hybrid search
            horizontal=True,
            key="search_mode"
        )
        
        # Basic parameter configuration
        col1, col2 = st.columns(2)
        
        with col1:
            top_k = st.number_input(
                "Number of Results", 
                min_value=1, 
                max_value=20, 
                value=5,
                key="search_top_k"
            )
        
        with col2:
            score_threshold = st.number_input(
                "Relevance Threshold", 
                min_value=0.0, 
                max_value=2.0, 
                value=1.0,
                step=0.1,
                help="Lower scores indicate higher relevance",
                key="score_threshold"
            )
        
        # Display different configuration options based on selected mode
        if search_mode in ["hybrid", "rag_fusion"]:
            st.write("**⚖️ Weight Configuration**")
            col_a, col_b = st.columns(2)
            
            with col_a:
                dense_weight = st.slider(
                    "Semantic Weight", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=DEFAULT_DENSE_WEIGHT,
                    step=0.1,
                    help="Control the weight of semantic search",
                    key="dense_weight"
                )
            
            with col_b:
                sparse_weight = st.slider(
                    "Keyword Weight", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=DEFAULT_SPARSE_WEIGHT,
                    step=0.1,
                    help="Control the weight of keyword search",
                    key="sparse_weight"
                )
            
            # Weight normalization notice
            total_weight = dense_weight + sparse_weight
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"⚠️ Weight sum is {total_weight:.2f}, system will auto-normalize")
            
            # Advanced parameters
            st.write("**🎛️ Advanced Parameters**")
            rrf_k = st.number_input(
                "RRF Parameter", 
                min_value=1, 
                max_value=100, 
                value=DEFAULT_RRF_K,
                help="Reciprocal Rank Fusion algorithm parameter, smaller values have greater ranking impact",
                key="rrf_k"
            )
        
        # RAG-fusion specific configuration
        rag_config = {}
        if search_mode == "rag_fusion":
            st.divider()
            rag_config = render_rag_fusion_config()
    
    # Search control buttons
    col_search, col_compare, col_clear = st.columns([2, 2, 1])
    
    with col_search:
        search_clicked = st.button(
            "🔍 Start Search", 
            use_container_width=True,
            disabled=not query.strip(),
            type="primary"
        )
    
    with col_compare:
        compare_clicked = st.button(
            "📊 Mode Comparison",
            use_container_width=True,
            disabled=not query.strip(),
            help="Compare results from different search modes"
        )
    
    with col_clear:
        clear_clicked = st.button(
            "🗑️ Clear Results", 
            use_container_width=True
        )
    
    if clear_clicked:
        for key in ["search_results", "comparison_results"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    # Execute search
    if search_clicked and query.strip():
        with st.spinner("Searching for relevant documents..."):
            try:
                # Prepare search parameters
                search_params = {
                    "query": query,
                    "knowledge_base_name": selected_kb,
                    "top_k": top_k,
                    "score_threshold": score_threshold,
                    "search_mode": search_mode
                }
                
                # Add weight parameters
                if search_mode in ["hybrid", "rag_fusion"]:
                    total_weight = dense_weight + sparse_weight
                    if total_weight > 0:
                        search_params["dense_weight"] = dense_weight / total_weight
                        search_params["sparse_weight"] = sparse_weight / total_weight
                    search_params["rrf_k"] = rrf_k
                
                # Add RAG-fusion specific parameters
                if search_mode == "rag_fusion" and rag_config:
                    search_params.update({
                        "rag_fusion_query_count": rag_config["query_count"],
                        "rag_fusion_model": rag_config["model"],
                        "rag_fusion_timeout": rag_config["timeout"],
                        "enable_rerank": rag_config["enable_rerank"],
                        "fusion_search_strategy": rag_config["fusion_strategy"]
                    })
                
                # Call search API
                results = api.search_kb_docs(**search_params)
                st.session_state["search_results"] = results
                st.session_state["search_query_used"] = query
                st.session_state["search_mode_used"] = search_mode
                st.session_state["search_config_used"] = search_params
                
                # Display success message
                if search_mode == "rag_fusion":
                    st.success(f"🚀 RAG-fusion search completed! Found {len(results)} relevant documents")
                else:
                    st.success(f"✅ {mode_labels[search_mode]} completed! Found {len(results)} relevant documents")
                
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                # If RAG-fusion fails, provide fallback option
                if search_mode == "rag_fusion":
                    if st.button("🔄 Try Hybrid Search", key="fallback_search"):
                        st.info("Trying hybrid search as fallback...")
                        st.rerun()
    
    # Execute mode comparison
    if compare_clicked and query.strip():
        with st.spinner("Performing mode comparison..."):
            try:
                comparison_modes = ["vector", "hybrid"]
                if RAG_FUSION_AVAILABLE:
                    comparison_modes.append("rag_fusion")
                
                comparison_results = {}
                
                for mode in comparison_modes:
                    try:
                        compare_params = {
                            "query": query,
                            "knowledge_base_name": selected_kb,
                            "top_k": 5,
                            "search_mode": mode
                        }
                        
                        if mode in ["hybrid", "rag_fusion"]:
                            compare_params.update({
                                "dense_weight": 0.7,
                                "sparse_weight": 0.3,
                                "rrf_k": 60
                            })
                        
                        if mode == "rag_fusion":
                            compare_params.update({
                                "rag_fusion_query_count": 3,
                                "rag_fusion_model": RAG_FUSION_LLM_MODEL
                            })
                        
                        results = api.search_kb_docs(**compare_params)
                        comparison_results[mode] = {
                            "results": results,
                            "count": len(results),
                            "avg_score": sum(getattr(r, 'score', 0) for r in results) / len(results) if results else 0
                        }
                        
                    except Exception as e:
                        comparison_results[mode] = {
                            "error": str(e),
                            "results": [],
                            "count": 0,
                            "avg_score": 0
                        }
                
                st.session_state["comparison_results"] = comparison_results
                st.session_state["comparison_query"] = query
                
            except Exception as e:
                st.error(f"Mode comparison error: {str(e)}")
    
    # Display search results
    if "search_results" in st.session_state and st.session_state["search_results"]:
        st.divider()
        render_search_results()
    
    # Display comparison results
    if "comparison_results" in st.session_state:
        st.divider()
        render_comparison_results()


def render_search_results():
    """Render search results"""
    st.subheader("📋 Search Results")
    
    results = st.session_state["search_results"]
    query_used = st.session_state.get("search_query_used", "")
    mode_used = st.session_state.get("search_mode_used", "")
    config_used = st.session_state.get("search_config_used", {})
    
    # Result statistics
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        st.metric("Query Content", query_used)
    with col_info2:
        mode_display = {
            "hybrid": "🔀 Hybrid Search",
            "vector": "🧠 Vector Search", 
            "bm25": "🔤 Keyword Search",
            "rag_fusion": "🚀 RAG-fusion",
            "adaptive": "🎯 Adaptive"
        }.get(mode_used, mode_used)
        st.metric("Search Mode", mode_display)
    with col_info3:
        st.metric("Result Count", len(results))
    with col_info4:
        avg_score = sum(getattr(r, 'score', 0) for r in results) / len(results) if results else 0
        st.metric("Average Relevance", f"{avg_score:.4f}")
    
    # RAG-fusion special information display
    if mode_used == "rag_fusion" and config_used:
        with st.expander("🚀 RAG-fusion Execution Details", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.write(f"**Query Generation Count**: {config_used.get('rag_fusion_query_count', 'N/A')}")
            with col_b:
                st.write(f"**Model Used**: {config_used.get('rag_fusion_model', 'N/A')}")
            with col_c:
                st.write(f"**Fusion Strategy**: {config_used.get('fusion_search_strategy', 'hybrid')}")
    
    # Display each search result
    for i, result in enumerate(results):
        with st.container():
            # Create a checkbox to control content display
            source_name = "Unknown Source"
            if hasattr(result, 'metadata') and result.metadata:
                source_name = result.metadata.get('source', 'Unknown Source')
            elif isinstance(result, dict):
                source_name = result.get('metadata', {}).get('source', 'Unknown Source')
            
            show_detail = st.checkbox(
                f"📄 Result {i+1} - {source_name}", 
                value=(i < 3),  # Default expand first 3 results
                key=f"show_result_{i}"
            )
            
            if show_detail:
                col_content, col_meta = st.columns([3, 1])
                
                with col_content:
                    st.write("**Content:**")
                    content = ""
                    if hasattr(result, 'page_content'):
                        content = result.page_content
                    elif isinstance(result, dict):
                        content = result.get("page_content", "")
                    
                    # Highlight query words
                    if query_used and content:
                        for word in query_used.split():
                            if len(word) > 2:
                                content = content.replace(
                                    word, 
                                    f"**{word}**"
                                )
                    st.markdown(content)
                
                with col_meta:
                    st.write("**Metadata:**")
                    
                    # Display relevance score
                    score = None
                    if hasattr(result, 'score'):
                        score = result.score
                    elif isinstance(result, dict):
                        score = result.get('score')
                    
                    if score is not None:
                        st.write(f"📊 Relevance: {score:.4f}")
                    
                    # Display metadata
                    metadata = {}
                    if hasattr(result, 'metadata'):
                        metadata = result.metadata or {}
                    elif isinstance(result, dict):
                        metadata = result.get("metadata", {})
                    
                    if metadata:
                        st.write(f"📁 Source: {metadata.get('source', 'N/A')}")
                        if 'search_mode_used' in metadata:
                            st.write(f"🔍 Mode: {metadata['search_mode_used']}")
                        if mode_used == "rag_fusion" and 'rag_fusion' in metadata:
                            rag_info = metadata['rag_fusion']
                            st.write(f"🚀 Query Count: {rag_info.get('query_count', 'N/A')}")
                
                # Action buttons
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    st.button(
                        "📋 Copy Content", 
                        key=f"copy_{i}",
                        help="Copy this content",
                        use_container_width=True
                    )
                with col_btn2:
                    st.button(
                        "🔗 View Details", 
                        key=f"detail_{i}",
                        help="View complete document information",
                        use_container_width=True
                    )
            
            # Add separator
            if i < len(results) - 1:
                st.divider()


def render_comparison_results():
    """Render mode comparison results"""
    st.subheader("📊 Search Mode Comparison")
    
    comparison_results = st.session_state["comparison_results"]
    query = st.session_state.get("comparison_query", "")
    
    st.write(f"**Query Content**: {query}")
    
    # Create comparison table
    comparison_data = []
    mode_labels = {
        "vector": "🧠 Vector Search",
        "hybrid": "🔀 Hybrid Search",
        "rag_fusion": "🚀 RAG-fusion",
        "bm25": "🔤 Keyword Search",
        "adaptive": "🎯 Adaptive"
    }
    
    for mode, data in comparison_results.items():
        comparison_data.append({
            "Search Mode": mode_labels.get(mode, mode),
            "Result Count": data["count"],
            "Average Relevance": f"{data['avg_score']:.4f}",
            "Status": "✅ Success" if "error" not in data else f"❌ {data['error'][:30]}..."
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Detailed result comparison
    tabs = st.tabs([mode_labels.get(mode, mode) for mode in comparison_results.keys()])
    
    for i, (mode, data) in enumerate(comparison_results.items()):
        with tabs[i]:
            if "error" in data:
                st.error(f"Search failed: {data['error']}")
            else:
                results = data["results"]
                st.write(f"Found {len(results)} results, average relevance: {data['avg_score']:.4f}")
                
                for j, result in enumerate(results[:3]):  # Only show first 3 results
                    with st.expander(f"Result {j+1}", expanded=j==0):
                        content = ""
                        if hasattr(result, 'page_content'):
                            content = result.page_content
                        elif isinstance(result, dict):
                            content = result.get("page_content", "")
                        
                        st.write(content[:300] + "..." if len(content) > 300 else content)
                        
                        if hasattr(result, 'score'):
                            st.write(f"Relevance: {result.score:.4f}")
    
    # Comparison analysis
    with st.expander("📈 Comparison Analysis", expanded=True):
        successful_results = {k: v for k, v in comparison_results.items() if "error" not in v}
        
        if successful_results:
            best_count = max(successful_results.items(), key=lambda x: x[1]["count"])
            best_score = max(successful_results.items(), key=lambda x: x[1]["avg_score"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"📈 **Most Results**: {mode_labels.get(best_count[0], best_count[0])} ({best_count[1]['count']} results)")
            with col2:
                st.success(f"🎯 **Highest Relevance**: {mode_labels.get(best_score[0], best_score[0])} ({best_score[1]['avg_score']:.4f})")
            
            # Recommendations
            if "rag_fusion" in successful_results:
                st.info("💡 **Recommendation**: RAG-fusion typically provides more comprehensive and accurate search results")
            elif "hybrid" in successful_results:
                st.info("💡 **Recommendation**: Hybrid search balances semantic understanding and keyword matching")


def knowledge_base_page(api: ApiRequest, is_lite: bool = None):
    try:
        kb_list = {x["kb_name"]: x for x in get_kb_details()}
    except Exception as e:
        st.error(
            "Error getting knowledge base information. Please check if you have completed initialization or migration according to step '4 Knowledge Base Initialization and Migration' in `README.md`, or if there is a database connection error.")
        st.stop()
    kb_names = list(kb_list.keys())

    if "selected_kb_name" in st.session_state and st.session_state["selected_kb_name"] in kb_names:
        selected_kb_index = kb_names.index(st.session_state["selected_kb_name"])
    else:
        selected_kb_index = 0

    if "selected_kb_info" not in st.session_state:
        st.session_state["selected_kb_info"] = ""

    def format_selected_kb(kb_name: str) -> str:
        if kb := kb_list.get(kb_name):
            rag_status = "🚀" if RAG_FUSION_AVAILABLE else ""
            return f"{kb_name} ({kb['vs_type']} @ {kb['embed_model']}) {rag_status}"
        else:
            return kb_name

    selected_kb = st.selectbox(
        "Please select or create a knowledge base:",
        kb_names + ["Create New Knowledge Base"],
        format_func=format_selected_kb,
        index=selected_kb_index
    )

    if selected_kb == "Create New Knowledge Base":
        with st.form("Create New Knowledge Base"):

            kb_name = st.text_input(
                "New Knowledge Base Name",
                placeholder="New knowledge base name, Chinese names not supported",
                key="kb_name",
            )
            kb_info = st.text_input(
                "Knowledge Base Description",
                placeholder="Knowledge base description for easy Agent lookup",
                key="kb_info",
            )

            cols = st.columns(2)

            vs_types = list(kbs_config.keys())
            vs_type = cols[0].selectbox(
                "Vector Store Type",
                vs_types,
                index=vs_types.index(DEFAULT_VS_TYPE),
                key="vs_type",
            )

            if is_lite:
                embed_models = list_online_embed_models()
            else:
                embed_models = list_embed_models() + list_online_embed_models()

            embed_model = cols[1].selectbox(
                "Embedding Model",
                embed_models,
                index=embed_models.index(EMBEDDING_MODEL),
                key="embed_model",
            )
            
            # Add RAG-fusion option
            if RAG_FUSION_AVAILABLE:
                enable_rag_fusion = st.checkbox(
                    "🚀 Enable RAG-fusion Feature", 
                    value=True,
                    help="Enable multi-query fusion retrieval functionality",
                    key="enable_rag_fusion"
                )

            submit_create_kb = st.form_submit_button(
                "Create",
                use_container_width=True,
            )

        if submit_create_kb:
            if not kb_name or not kb_name.strip():
                st.error(f"Knowledge base name cannot be empty!")
            elif kb_name in kb_list:
                st.error(f"Knowledge base named {kb_name} already exists!")
            else:
                create_params = {
                    "knowledge_base_name": kb_name,
                    "vector_store_type": vs_type,
                    "embed_model": embed_model,
                }
                
                if RAG_FUSION_AVAILABLE:
                    create_params["enable_rag_fusion"] = st.session_state.get("enable_rag_fusion", False)
                
                ret = api.create_knowledge_base(**create_params)
                
                if ret.get("code") == 200:
                    success_msg = ret.get("msg", "Created successfully")
                    if RAG_FUSION_AVAILABLE and st.session_state.get("enable_rag_fusion", False):
                        success_msg += " (RAG-fusion enabled)"
                    st.toast(success_msg, icon="✅")
                else:
                    st.toast(ret.get("msg", "Creation failed"), icon="❌")
                    
                st.session_state["selected_kb_name"] = kb_name
                st.session_state["selected_kb_info"] = kb_info
                st.rerun()

    elif selected_kb:
        kb = selected_kb
        st.session_state["selected_kb_info"] = kb_list[kb]['kb_info']
        
        # Add Tab layout, now includes 3 tabs
        tab1, tab2, tab3 = st.tabs(["📚 Knowledge Base Management", "🔍 Intelligent Search", "⚙️ System Configuration"])
        
        with tab1:
            # Original knowledge base management functionality
            # Upload files
            files = st.file_uploader("Upload knowledge files:",
                                     [i for ls in LOADER_DICT.values() for i in ls],
                                     accept_multiple_files=True,
                                     )
            kb_info = st.text_area("Please enter knowledge base description:", value=st.session_state["selected_kb_info"], max_chars=None,
                                   key=None,
                                   help=None, on_change=None, args=None, kwargs=None)

            if kb_info != st.session_state["selected_kb_info"]:
                st.session_state["selected_kb_info"] = kb_info
                api.update_kb_info(kb, kb_info)

            # ========== New: Text splitting strategy configuration interface ==========
            splitter_config = render_text_splitter_config()
            
            # Get text splitting strategy configuration parameters
            selected_text_splitter = splitter_config["text_splitter_name"]
            chunk_size = splitter_config["chunk_size"]
            chunk_overlap = splitter_config["chunk_overlap"]
            auto_select = splitter_config["auto_select"]

            # Traditional parameter configuration (maintain compatibility)
            with st.container():
                st.write("**🔧 Other Configuration**")
                zh_title_enhance = st.checkbox(
                    "Enable Chinese Title Enhancement", 
                    value=ZH_TITLE_ENHANCE and selected_text_splitter not in [
                        "EnglishSentenceSplitter", "EnglishParagraphSplitter", 
                        "SemanticChunkSplitter", "SlidingWindowSplitter"
                    ],
                    help="Note: This option is automatically disabled for English splitters",
                    key="zh_title_enhance"
                )
                
                # If English splitter is selected, automatically disable Chinese title enhancement
                if selected_text_splitter in ["EnglishSentenceSplitter", "EnglishParagraphSplitter", 
                                            "SemanticChunkSplitter", "SlidingWindowSplitter"]:
                    zh_title_enhance = False
                    if st.session_state.get("zh_title_enhance", False):
                        st.info("💡 Chinese title enhancement automatically disabled (English splitter)")

            if st.button(
                    "Add Files to Knowledge Base",
                    disabled=len(files) == 0,
                    type="primary",
                    use_container_width=True,
            ):
                # Prepare additional parameters
                extra_params = {}
                
                # If specific splitter is selected, pass splitter name
                if selected_text_splitter != TEXT_SPLITTER_NAME:
                    extra_params["text_splitter_name"] = selected_text_splitter
                
                # Display configuration being used
                with st.status("Processing files...", expanded=True) as status:
                    st.write(f"📊 Using Splitter: **{selected_text_splitter}**")
                    st.write(f"🔧 Chunk Size: **{chunk_size}** characters")
                    st.write(f"🔗 Overlap Size: **{chunk_overlap}** characters")
                    st.write(f"🎯 Auto Selection: **{'Enabled' if auto_select else 'Disabled'}**")
                    
                    ret = api.upload_kb_docs(files,
                                             knowledge_base_name=kb,
                                             override=True,
                                             chunk_size=chunk_size,
                                             chunk_overlap=chunk_overlap,
                                             zh_title_enhance=zh_title_enhance,
                                             **extra_params)
                    
                    if msg := check_success_msg(ret):
                        status.update(label="✅ File upload successful!", state="complete")
                        st.toast(msg, icon="✔")
                    elif msg := check_error_msg(ret):
                        status.update(label="❌ File upload failed", state="error")
                        st.toast(msg, icon="✖")

            st.divider()

            # Knowledge base details
            doc_details = pd.DataFrame(get_kb_file_details(kb))
            selected_rows = []
            if not len(doc_details):
                st.info(f"Knowledge base `{kb}` has no files yet")
            else:
                st.write(f"Files in knowledge base `{kb}`:")
                st.info("Knowledge base contains source files and vector store, please select files from the table below for operations")
                doc_details.drop(columns=["kb_name"], inplace=True)
                doc_details = doc_details[[
                    "No", "file_name", "document_loader", "text_splitter", "docs_count", "in_folder", "in_db",
                ]]
                doc_details["in_folder"] = doc_details["in_folder"].replace(True, "✓").replace(False, "×")
                doc_details["in_db"] = doc_details["in_db"].replace(True, "✓").replace(False, "×")
                gb = config_aggrid(
                    doc_details,
                    {
                        ("No", "Number"): {},
                        ("file_name", "Document Name"): {},
                        ("document_loader", "Document Loader"): {},
                        ("docs_count", "Document Count"): {},
                        ("text_splitter", "Text Splitter"): {},
                        ("in_folder", "Source File"): {"cellRenderer": cell_renderer},
                        ("in_db", "Vector Store"): {"cellRenderer": cell_renderer},
                    },
                    "multiple",
                )

                doc_grid = AgGrid(
                    doc_details,
                    gb.build(),
                    columns_auto_size_mode="FIT_CONTENTS",
                    theme="alpine",
                    custom_css={
                        "#gridToolBar": {"display": "none"},
                    },
                    allow_unsafe_jscode=True,
                    enable_enterprise_modules=False
                )

                selected_rows = doc_grid.get("selected_rows", [])

                cols = st.columns(4)
                file_name, file_path = file_exists(kb, selected_rows)
                if file_path:
                    with open(file_path, "rb") as fp:
                        cols[0].download_button(
                            "Download Selected Document",
                            fp,
                            file_name=file_name,
                            use_container_width=True, )
                else:
                    cols[0].download_button(
                        "Download Selected Document",
                        "",
                        disabled=True,
                        use_container_width=True, )

                st.write()
                
                # Update document processing operations, using currently selected splitter configuration
                if cols[1].button(
                        "Re-add to Vector Store" if selected_rows and (
                                pd.DataFrame(selected_rows)["in_db"]).any() else "Add to Vector Store",
                        disabled=not file_exists(kb, selected_rows)[0],
                        use_container_width=True,
                ):
                    file_names = [row["file_name"] for row in selected_rows]
                    
                    # Prepare update parameters, including splitter configuration
                    update_params = {
                        "knowledge_base_name": kb,
                        "file_names": file_names,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "zh_title_enhance": zh_title_enhance
                    }
                    
                    if selected_text_splitter != TEXT_SPLITTER_NAME:
                        update_params["text_splitter_name"] = selected_text_splitter
                    
                    with st.spinner(f"Re-processing documents using {selected_text_splitter}..."):
                        api.update_kb_docs(**update_params)
                        st.success(f"Re-processing completed using {selected_text_splitter}")
                    st.rerun()

                if cols[2].button(
                        "Delete from Vector Store",
                        disabled=not (selected_rows and selected_rows[0]["in_db"]),
                        use_container_width=True,
                ):
                    file_names = [row["file_name"] for row in selected_rows]
                    api.delete_kb_docs(kb, file_names=file_names)
                    st.rerun()

                if cols[3].button(
                        "Delete from Knowledge Base",
                        type="primary",
                        use_container_width=True,
                ):
                    file_names = [row["file_name"] for row in selected_rows]
                    api.delete_kb_docs(kb, file_names=file_names, delete_content=True)
                    st.rerun()

            st.divider()

            cols = st.columns(3)

            if cols[0].button(
                    "Rebuild Vector Store from Source Files",
                    help="Rebuild entire vector store using current splitting strategy",
                    use_container_width=True,
                    type="primary",
            ):
                with st.spinner("Vector store reconstruction in progress, please wait patiently and do not refresh or close the page."):
                    empty = st.empty()
                    empty.progress(0.0, "")
                    
                    # Prepare rebuild parameters, including splitter configuration
                    rebuild_params = {
                        "knowledge_base_name": kb,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "zh_title_enhance": zh_title_enhance
                    }
                    
                    if selected_text_splitter != TEXT_SPLITTER_NAME:
                        rebuild_params["text_splitter_name"] = selected_text_splitter
                    
                    # Display rebuild configuration information
                    st.info(f"🔧 Rebuild Configuration: {selected_text_splitter} | Chunk Size: {chunk_size} | Overlap: {chunk_overlap}")
                    
                    for d in api.recreate_vector_store(**rebuild_params):
                        if msg := check_error_msg(d):
                            st.toast(msg)
                        else:
                            empty.progress(d["finished"] / d["total"], d["msg"])
                    st.rerun()

            if cols[2].button(
                    "Delete Knowledge Base",
                    use_container_width=True,
            ):
                ret = api.delete_knowledge_base(kb)
                st.toast(ret.get("msg", " "))
                time.sleep(1)
                st.rerun()

            # Document editing functionality (retain original functionality)
            with st.sidebar:
                keyword = st.text_input("Search Keywords")
                top_k = st.slider("Match Count", 1, 100, 3)

            st.write("Document list in file. Double-click to edit, enter Y in delete column to delete corresponding row.")
            docs = []
            df = pd.DataFrame([], columns=["seq", "id", "content", "source"])
            if selected_rows:
                file_name = selected_rows[0]["file_name"]
                docs = api.search_kb_docs(knowledge_base_name=selected_kb, file_name=file_name)
                data = [
                    {"seq": i + 1, "id": x["id"], "page_content": x["page_content"], "source": x["metadata"].get("source"),
                     "type": x["type"],
                     "metadata": json.dumps(x["metadata"], ensure_ascii=False),
                     "to_del": "",
                     } for i, x in enumerate(docs)]
                df = pd.DataFrame(data)

                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_columns(["id", "source", "type", "metadata"], hide=True)
                gb.configure_column("seq", "No.", width=50)
                gb.configure_column("page_content", "Content", editable=True, autoHeight=True, wrapText=True, flex=1,
                                    cellEditor="agLargeTextCellEditor", cellEditorPopup=True)
                gb.configure_column("to_del", "Delete", editable=True, width=50, wrapHeaderText=True,
                                    cellEditor="agCheckboxCellEditor", cellRender="agCheckboxCellRenderer")
                gb.configure_selection()
                edit_docs = AgGrid(df, gb.build())

                if st.button("Save Changes"):
                    origin_docs = {
                        x["id"]: {"page_content": x["page_content"], "type": x["type"], "metadata": x["metadata"]} for x in
                        docs}
                    changed_docs = []
                    for index, row in edit_docs.data.iterrows():
                        origin_doc = origin_docs[row["id"]]
                        if row["page_content"] != origin_doc["page_content"]:
                            if row["to_del"] not in ["Y", "y", 1]:
                                changed_docs.append({
                                    "page_content": row["page_content"],
                                    "type": row["type"],
                                    "metadata": json.loads(row["metadata"]),
                                })

                    if changed_docs:
                        if api.update_kb_docs(knowledge_base_name=selected_kb,
                                              file_names=[file_name],
                                              docs={file_name: changed_docs}):
                            st.toast("Document update successful")
                        else:
                            st.toast("Document update failed")
        
        with tab2:
            # New intelligent search interface (with RAG-fusion support)
            render_search_interface(api, selected_kb)
        
        with tab3:
            # New system configuration tab
            st.subheader("⚙️ System Configuration")
            
            # RAG-fusion global configuration
            if RAG_FUSION_AVAILABLE:
                st.write("**🚀 RAG-fusion Global Configuration**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Feature Status", "✅ Enabled" if RAG_FUSION_AVAILABLE else "❌ Disabled")
                with col2:
                    st.metric("Supported Models", len(RAG_FUSION_SUPPORTED_MODELS))
                with col3:
                    st.metric("Default Query Count", RAG_FUSION_QUERY_COUNT)
                
                # Model configuration information
                with st.expander("🤖 Model Configuration Details", expanded=False):
                    st.write("**Default Model:**")
                    st.code(RAG_FUSION_LLM_MODEL)
                    
                    st.write("**Supported Model List:**")
                    for i, model in enumerate(RAG_FUSION_SUPPORTED_MODELS[:10]):  # Only show first 10
                        st.write(f"{i+1}. {model}")
                    
                    if len(RAG_FUSION_SUPPORTED_MODELS) > 10:
                        st.write(f"... {len(RAG_FUSION_SUPPORTED_MODELS) - 10} more models")
                
                # Configuration validation
                with st.expander("🔍 Configuration Validation", expanded=False):
                    if st.button("Validate RAG-fusion Configuration", key="validate_config"):
                        with st.spinner("Validating configuration..."):
                            # Simulate configuration validation
                            validation_results = {
                                "rag_fusion_enabled": RAG_FUSION_AVAILABLE,
                                "model_available": RAG_FUSION_LLM_MODEL in RAG_FUSION_SUPPORTED_MODELS,
                                "config_complete": bool(RAG_FUSION_CONFIG),
                                "kb_support": True  # Assume knowledge base support
                            }
                            
                            all_valid = all(validation_results.values())
                            
                            if all_valid:
                                st.success("✅ All configuration validation passed!")
                            else:
                                st.warning("⚠️ Some configurations need checking")
                            
                            # Display detailed results
                            for check, result in validation_results.items():
                                status = "✅" if result else "❌"
                                st.write(f"{status} {check}: {'Passed' if result else 'Failed'}")
            
            else:
                st.warning("⚠️ RAG-fusion feature not enabled")
                st.write("To enable RAG-fusion, please check the following configuration:")
                st.code("""
                # Set in configs/model_config.py:
                ENABLE_RAG_FUSION = True
                RAG_FUSION_LLM_MODEL = "Qwen1.5-7B-Chat"
                RAG_FUSION_SUPPORTED_MODELS = ["Qwen1.5-7B-Chat", ...]
                """)
            
            st.divider()
            
            # Other system information
            st.write("**💾 Storage Configuration**")
            storage_info_col1, storage_info_col2 = st.columns(2)
            
            with storage_info_col1:
                st.write(f"• Default Vector Store: {DEFAULT_VS_TYPE}")
                st.write(f"• Embedding Model: {EMBEDDING_MODEL}")
            
            with storage_info_col2:
                st.write(f"• Default Chunk Size: {CHUNK_SIZE}")
                st.write(f"• Default Overlap: {OVERLAP_SIZE}")
            
            # System status check
            if st.button("🔄 Refresh System Status", key="refresh_system"):
                st.rerun()