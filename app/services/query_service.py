"""Query handling service."""

import logging
from fastapi import HTTPException
from app.types.schemas import QueryRequest, QueryResponse
from app.core.prompts import format_natural_response as _format_natural_response

logger = logging.getLogger("ai-ca")


async def handle_query(request: QueryRequest) -> QueryResponse:
    """Execute analytical query with unique query ID for audit trail."""
    try:
        from app.agents.router import (
            get_router_agent,
            TRACK_DATA,
            TRACK_DOC,
            TRACK_WEB,
            TRACK_DOC_SUMMARY,
            TRACK_OUT_OF_DOMAIN,
        )
        from app.agents.data_analyst import get_data_analyst_agent
        from app.core.llm_wrapper import get_llm_wrapper
        from app.core.id_generator import generate_query_id, get_iso_timestamp

        # Generate unique query ID for audit trail
        query_id = generate_query_id()
        timestamp = get_iso_timestamp()
        logger.info(
            f"Query {query_id}: '{request.query[:50]}...' from {request.client}"
        )

        llm = get_llm_wrapper()
        router = get_router_agent(llm)
        agent = get_data_analyst_agent(llm)

        # Check if client has data loaded
        datasets = agent.list_datasets_for_client(request.client)
        has_loaded_data = len(datasets) > 0

        # Route query with data context
        route_result = router.route(
            request.query, f"Client: {request.client}", has_loaded_data=has_loaded_data
        )
        track = route_result.get("track", TRACK_DATA)

        logger.info(
            f"Query {query_id}: Routed to {track} (confidence: {route_result.get('confidence', 0):.2f})"
        )

        # ==================================================================
        # HANDLE OUT-OF-DOMAIN QUERIES
        # ==================================================================
        if track == TRACK_OUT_OF_DOMAIN:
            return QueryResponse(
                success=True,
                result=(
                    "I'm an AI Chartered Accountant assistant. I can help you with financial "
                    "data analysis, revenue/expense calculations, and document search. Please "
                    "ask me something related to your financial data!"
                ),
                method="out_of_domain",
                explanation="Query was not related to CA/financial domain",
                query_id=query_id,
                metadata={"route": track},
            )

        # ==================================================================
        # HANDLE DATASET SUMMARY QUERIES
        # ==================================================================
        if track == TRACK_DOC_SUMMARY:
            if not datasets:
                return QueryResponse(
                    success=False,
                    error="No datasets found. Please upload data first.",
                    method="summary",
                    query_id=query_id,
                )

            # Use summarize_dataset for each dataset
            summaries = []
            for ds in datasets[:5]:  # Limit to 5
                ds_id = ds.get("dataset_id", "")
                result = agent.summarize_dataset(ds_id, client_id=request.client, user_query=request.query)
                if result.get("value"):
                    sheet_name = ds_id.split(":")[-1]
                    summaries.append(f"**{sheet_name}:** {result['value']}")

            if summaries:
                combined = "\n\n".join(summaries)
                return QueryResponse(
                    success=True,
                    result=combined,
                    method="summarize_dataset",
                    explanation=f"Summary of {len(summaries)} dataset(s)",
                    query_id=query_id,
                    metadata={"route": track, "datasets": len(summaries)},
                )
            return QueryResponse(
                success=False,
                error="Could not generate dataset summaries",
                method="summary",
                query_id=query_id,
            )

        if track == TRACK_DATA:
            # datasets already fetched earlier (for has_loaded_data check)
            if not datasets:
                return QueryResponse(
                    success=False,
                    error="No datasets found for client. Please upload data first.",
                    query_id=query_id,
                )

            # Try specific dataset if provided, otherwise find matching one
            best_result = None
            best_dataset_id = None

            if request.dataset_id:
                # Use specified dataset
                result = agent.execute_sql_query(
                    query=request.query,
                    df_id=request.dataset_id,
                    client_id=request.client,
                    use_cache=request.use_cache,
                )
                if result.success:
                    best_result = result
                    best_dataset_id = request.dataset_id
            else:
                # Keyword-based matching for JSON tables
                query_lower = request.query.lower()
                keyword_map = {
                    "balance sheet": "balance_sheet",
                    "assets": "balance_sheet",
                    "liabilities": "balance_sheet",
                    "income statement": "income_statement",
                    "revenue": "income_statement",
                    "profit": "income_statement",
                    "company": "company_meta",
                    "ticker": "company_meta",
                    "employees": "company_meta",
                }

                keyword_match = None
                for keyword, table_suffix in keyword_map.items():
                    if keyword in query_lower:
                        for ds in datasets:
                            ds_id = ds.get("dataset_id", "")
                            if ds_id.endswith(table_suffix):
                                keyword_match = ds_id
                                break
                        if keyword_match:
                            break

                # Priority: keyword match > smart match
                matched_id = keyword_match or agent.match_dataset_by_query(
                    request.query, datasets
                )

                if matched_id:
                    result = agent.execute_sql_query(
                        query=request.query,
                        df_id=matched_id,
                        client_id=request.client,
                        use_cache=request.use_cache,
                    )
                    if result.success:
                        best_result = result
                        best_dataset_id = matched_id

                # If matched dataset failed, try all datasets
                if not best_result or not best_result.success:
                    for ds in datasets[:5]:  # Limit to 5 datasets
                        ds_id = ds.get("dataset_id")
                        if ds_id == matched_id:
                            continue
                        result = agent.execute_sql_query(
                            query=request.query,
                            df_id=ds_id,
                            client_id=request.client,
                            use_cache=request.use_cache,
                        )
                        if result.success:
                            if best_result is None or result.value is not None:
                                best_result = result
                                best_dataset_id = ds_id
                                if result.value is not None:
                                    break  # Found a good result

            if best_result and best_result.success:
                # PRODUCTION FIX: Generate unique cache context for result formatting
                # This prevents "project name" and "revenue" queries from getting same
                # cached formatted response even if analysis data is different
                import hashlib
                query_hash = hashlib.md5(request.query.lower().encode()).hexdigest()[:8]
                format_cache_context = f"{request.client}:format:{query_hash}"
                
                # Format as human-like response
                natural_result = _format_natural_response(
                    query=request.query,
                    raw_result=best_result.result,
                    explanation=best_result.explanation,
                    llm_wrapper=llm,
                    cache_context=format_cache_context,
                )

                return QueryResponse(
                    success=True,
                    result=natural_result,
                    method=best_result.method,
                    explanation=best_result.explanation,
                    error=None,
                    query_id=query_id,
                    metadata={
                        "dataset_id": best_dataset_id,
                        "route": track,
                        "raw_value": best_result.value,
                    },
                    provenance=best_result.provenance,
                )

            # FALLBACK: RAG Semantic Search
            try:
                from app.rag.ingest import get_rag_pipeline

                rag = get_rag_pipeline()
                if rag and rag.is_available:
                    rag_results = rag._ingestor.search(
                        query=request.query,
                        client_id=request.client,
                        top_k=3,
                        score_threshold=0.3,
                    )
                    if rag_results:
                        context = "\n\n".join(
                            [
                                f"Source: {r.get('metadata', {}).get('table_name', 'data')}\n{r.get('content', '')[:800]}"
                                for r in rag_results[:3]
                            ]
                        )
                        prompt = f"Based on this data:\n\n{context}\n\nAnswer: {request.query}"
                        # PRODUCTION FIX: Pass client ID as cache context for RAG queries
                        response = llm.invoke(prompt, cache_context=f"rag:{request.client}")
                        return QueryResponse(
                            success=True,
                            result=response,
                            method="rag:semantic",
                            explanation="Answer synthesized from indexed data",
                            query_id=query_id,
                            metadata={"route": track, "fallback": "rag"},
                            provenance={
                                "method": "rag_semantic",
                                "sources_count": len(rag_results),
                                "client_id": request.client
                            }
                        )
            except Exception as e:
                logger.debug(f"RAG fallback failed: {e}")

            # FALLBACK: Comprehensive LLM analysis
            # PRODUCTION FIX: Pass dataset IDs as cache context, add provenance
            try:
                # Collect data from all datasets
                all_data_context = []
                dataset_ids_used = []
                for ds in datasets[:3]:
                    df = agent._get_dataframe(ds.get("dataset_id"))
                    if df is not None:
                        table_name = ds.get("dataset_id", "").split(":")[-1]
                        dataset_ids_used.append(ds.get("dataset_id"))
                        context_parts = [f"\n--- TABLE: {table_name} ---"]
                        context_parts.append(f"Columns: {list(df.columns)}")
                        context_parts.append(f"Data:\n{df.to_string()}")
                        all_data_context.append("\n".join(context_parts))

                if all_data_context:
                    full_context = "\n".join(all_data_context)[:6000]
                    prompt = f"""Analyze this data and answer:\n{full_context}\n\nQuestion: {request.query}"""
                    # PRODUCTION FIX: Pass dataset context to prevent cross-dataset cache pollution
                    cache_context = ":".join(dataset_ids_used[:2])  # Use first 2 datasets in cache key
                    response = llm.invoke(prompt, cache_context=cache_context)
                    return QueryResponse(
                        success=True,
                        result=response,
                        method="llm:comprehensive",
                        explanation="Answer from comprehensive data analysis",
                        query_id=query_id,
                        metadata={"route": track, "fallback": "llm"},
                        provenance={
                            "method": "llm_comprehensive_fallback",
                            "datasets_used": dataset_ids_used,
                            "datasets_count": len(dataset_ids_used),
                            "context_length": len(full_context)
                        }
                    )
            except Exception as e:
                logger.debug(f"LLM fallback failed: {e}")

            return QueryResponse(
                success=False,
                error=best_result.error if best_result else "Query execution failed",
                method="failed",
                query_id=query_id,
            )

        if track == TRACK_DOC:
            # Document RAG query
            from app.rag.ingest import get_document_ingestor
            from app.core.prompts import get_document_rag_prompt

            ingestor = get_document_ingestor()
            docs = ingestor.search(request.query, request.client)

            if not docs:
                return QueryResponse(
                    success=False,
                    error="No relevant documents found",
                    method="rag",
                    query_id=query_id,
                )

            # Synthesize response
            context = "\n\n".join([d.get("content", "") for d in docs[:3]])
            prompt = get_document_rag_prompt(context, request.query)
            # PRODUCTION FIX: Pass document track as cache context
            response = llm.invoke(prompt, cache_context=f"doc:{request.client}")

            return QueryResponse(
                success=True,
                result=response,
                method="rag:document",
                explanation=f"Found {len(docs)} relevant documents",
                query_id=query_id,
                metadata={"route": track, "docs": len(docs)},
                provenance={
                    "method": "rag_document",
                    "documents_found": len(docs),
                    "client_id": request.client
                }
            )

        if track == TRACK_WEB:
            # Web search
            from app.tools.web_search import web_search

            result = web_search.invoke({"query": request.query, "num_results": 3})

            if result.get("result") == "success":
                # Synthesize response
                from app.core.prompts import get_web_search_prompt

                snippets = "\n\n".join(
                    [
                        f"**{r['title']}**\n{r['snippet']}\nURL: {r['url']}"
                        for r in result.get("results", [])
                    ]
                )

                prompt = get_web_search_prompt(snippets, request.query)
                # PRODUCTION FIX: Use web context for cache scoping
                response = llm.invoke(prompt, cache_context="web:search")

                return QueryResponse(
                    success=True,
                    result=response,
                    method="web:search",
                    explanation=f"Synthesized from {len(result.get('results', []))} web results",
                    query_id=query_id,
                    metadata={"route": track, "sources": result.get("results", [])},
                    provenance={
                        "method": "web_search",
                        "sources_count": len(result.get("results", [])),
                        "query": request.query
                    }
                )
            return QueryResponse(
                success=False,
                error="Web search returned no results",
                method="web:search",
                query_id=query_id,
            )

        return QueryResponse(
            success=False, error=f"Unknown track: {track}", query_id=query_id
        )

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
