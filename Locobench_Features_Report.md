# Locobench Features Impact Report

## Project Overview
LoCoBench is a comprehensive benchmark designed to evaluate long-context Large Language Models (LLMs) in complex software development scenarios. The project provides 8,000 evaluation scenarios across 10 programming languages with context lengths spanning 10K to 1M tokens.

## Executive Summary: Evolution & Score Impact

| Version | Score | Key Features | Main Improvements |
|:---|:---|:---|:---|
| **Baseline** | 1.789 | No retrieval | Baseline measurement |
| **v1** | ~1.90 | Basic Semantic Retrieval | Initial implementation (5% files) |
| **v2** | ~2.00-2.05 | Hybrid + Dependencies | BM25/Semantic hybrid, dependency graph, smart chunking |
| **v3** | ~2.05-2.10 | Query Expansion | Multi-query (8 queries), enhanced synonyms |
| **v4** | ~2.07 | Task Specialization | Category-specific parameters |
| **v5** | 2.115 (max) | Iterative Optimization | Adaptive strategy by task type |
| **v6** | 2.197 | MCP Integration | Model Context Protocol, tool-based selection |
| **v7** | 2.11 | Original Reranker + Multi-Query | Integrated reranker with custom retriever |
| **v8** | 2.266 | Response Processing | Improved model response processing |

---

## Detailed Feature Analysis

### 1. Basic Semantic Retrieval (Version 1)
**Feature:**
- **Selection:** Selected top 5% of files based purely on semantic similarity (cosine similarity) to the task prompt.
- **Mechanism:** Simple embedding-based search using `sentence-transformers`.
- **Constraint:** Context limited to ~25K characters (~6K tokens).

**Impact & Results:**
- **Score:** ~1.90 (Minimal improvement over baseline 1.789).
- **Analysis:**
    - **Failure Point:** Missed critical dependency files. Semantic similarity alone ignored structural relationships.
    - **Comparison:** Performance was roughly equivalent to simple truncation.

### 2. Hybrid Search & Dependency Analysis (Version 2)
**Feature:**
- **Hybrid Search:** Combined BM25 (keyword-based) and semantic search using an adaptive `hybrid_alpha` parameter.
- **Dependency Graph:** Included files linked by direct/inverse dependencies (2-3 levels deep).
    - *Architectural tasks:* 20-25 files/level.
- **Smart Chunking:**
    - Files split into 2000-char chunks.
    - Top-5 relevant chunks selected per file.
    - **Boosting:** 0.10-0.12 priority boost for file beginnings (class definitions).
- **Multi-Level Strategy:**
    1.  **Level 1:** Semantically relevant files.
    2.  **Level 2:** Files from dependency graph.
    3.  **Level 3:** Beginnings of other important files.

**Impact & Results:**
- **Score:** ~2.00 - 2.05.
- **Analysis:**
    - **Coverage:** +30-40% improved coverage for architectural tasks.
    - **Accuracy:** +15-20% better file selection accuracy.

### 3. Query Expansion & Multi-Query Retrieval (Version 3)
**Feature:**
- **Query Expansion:**
    - Dictionary increased to 4 synonyms per keyword.
    - **Domain-specific terms:**
        - *Arch:* hierarchy, composition, coupling.
        - *Security:* encryption, sanitization.
        - *Comprehension:* control flow, call stack.
- **Multi-Query Strategy:**
    - Increased queries from 5 to 8.
    - Specialized queries per task type (e.g., "Architectural focus", "Security focus").
    - **Weighting:** Original query (1.0) > Specialized (0.9) > Others (0.7).
- **Adaptive Hybrid Alpha:**
    - *Security:* 0.80 (High semantic).
    - *Comprehension:* 0.70.
    - *Architectural:* 0.65 (High keyword/BM25).

**Impact & Results:**
- **Score:** ~2.05 - 2.10.
- **Analysis:**
    - **Coverage:** +25-30% query coverage improvement.
    - **Accuracy:** +10-15% search accuracy improvement.

### 4. Task Category Specialization (Version 4)
**Feature:**
- **Architectural Understanding:**
    - `file_multiplier`: 1.60-1.85.
    - `level1_ratio` (Semantic): 0.50-0.65.
    - `level2_ratio` (Dependencies): 0.25-0.40.
- **Code Comprehension:**
    - `level1_ratio`: 0.60-0.72.
    - `level2_ratio` (Tracing): 0.23-0.35.
- **Security Analysis:**
    - `level1_ratio`: 0.75-0.80 (Max semantics).
    - `level2_ratio`: 0.15-0.20.
- **Enhanced Boosting:**
    - Keyword/Pattern boost increased to 0.28-0.35.
    - Test files boost: 0.25.
    - Task prompt match boost: 0.20.

**Impact & Results:**
- **Score:** ~2.07.
- **Analysis:** +15-25% accuracy improvement per category.

### 5. Iterative Parameter Optimization (Version 5)
**Feature:**
- **Cycle 1 (Arch focus):** Increased semantics (+16%) and hybrid alpha (0.72).
- **Cycle 2 (Comprehension focus):** Optimized hybrid search (0.75) and increased file multiplier (1.40).
- **Cycle 3 (Max Semantics):** `hybrid_alpha` increased to 0.78.

**Impact & Results:**
- **Score:** 2.115 (Max achieved for v5).
- **Analysis:** Proven that adaptive strategy by task type yields best results.

### 6. Model Context Protocol (MCP) Integration (Version 6)
**Feature:**
- **Architecture:** Task → MCP Server → LLM Client → Tool Calls → File Selection.
- **Tools:**
    - `find_security_sensitive_files`
    - `analyze_dependency_graph_for_security`
    - `find_architectural_components`
    - `trace_code_execution_flow`
- **Logic:** Interactive file selection with LLM (up to 5 iterations).

**Impact & Results:**
- **Score:** 2.197.
- **Analysis:** Significant jump in score due to intelligent, agentic file selection.

### 7. Original Reranker + Multi-Query Retrieval (Version 7)
**Feature:**
- **Integration:** Integrated the original LocoBench reranker with custom Retriever logic from Version 3.
- **Mechanism:** Combined specialized query strategies and adaptive hybrid alpha with the reranker.

**Impact & Results:**
- **Score:** 2.11.
- **Analysis:** Validated the combination of reranking with specialized queries.

### 8. Response Processing & Environment Adaptation (Version 8)
**Feature:**
- **Response Processing:** Improved model response processing logic.
- **Adaptation:** Aligned benchmark with current environment.
- **Support:** Proprietary model support, language/complexity filters.

**Impact & Results:**
- **Score:** 2.266.
- **Context:** ~20k prompt length.
- **Analysis:** Highest achieved score, driven by better handling of model outputs and environment alignment.

## Conclusion
For the majority of models, the original implementation remains suitable, as the benchmark demonstrates limited sensitivity to alternative implementations.
