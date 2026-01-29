"""Streaming query service (SSE)."""

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any

from app.core.id_generator import generate_query_id, normalize_client_id

logger = logging.getLogger("ai-ca")


async def stream_query_events(request) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events for a streaming query.
    """
    query_id = generate_query_id()
    safe_client = normalize_client_id(request.client)

    try:
        yield (
            f"event: start\ndata: {json.dumps({'query_id': query_id, 'client': safe_client, 'status': 'processing'})}\n\n"
        )

        from app.agents.data_analyst import get_data_analyst_agent
        from app.agents.router import get_router_agent
        from app.core.llm_wrapper import get_llm_wrapper
        from app.core.data_registry import get_data_registry

        agent = get_data_analyst_agent()
        router = get_router_agent()
        llm = get_llm_wrapper()
        registry = get_data_registry()

        datasets = registry.list_for_client(safe_client)
        has_data = len(datasets) > 0

        route_result = router.route(request.query, has_loaded_data=has_data)
        track = route_result.get("track", "TRACK_DATA")

        yield f"event: route\ndata: {json.dumps({'track': track, 'confidence': route_result.get('confidence', 0)})}\n\n"

        result_text = ""
        method = "unknown"

        if track == "TRACK_DOC_SUMMARY" and datasets:
            summaries = []
            for i, ds in enumerate(datasets[:3]):
                ds_id = ds.get("dataset_id", "")
                sheet_name = ds_id.split(":")[-1] if ":" in ds_id else ds_id
                yield (
                    f"event: progress\ndata: {json.dumps({'sheet': sheet_name, 'index': i+1, 'total': min(len(datasets), 3)})}\n\n"
                )

                summary_result = agent.summarize_dataset(ds_id, client_id=safe_client)
                if summary_result.get("value"):
                    summary_text = f"**{sheet_name}:** {summary_result['value']}"
                    summaries.append(summary_text)
                    for word in summary_text.split():
                        yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
                        await asyncio.sleep(0.01)
                    newline_token = "\n\n"
                    yield f"event: token\ndata: {json.dumps({'token': newline_token})}\n\n"

            result_text = "\n\n".join(summaries)
            method = "summarize_dataset"

        elif track == "TRACK_DATA" and datasets:
            matched_id = request.dataset_id or agent.match_dataset_by_query(
                request.query, datasets
            )
            if not matched_id and datasets:
                matched_id = datasets[0].get("dataset_id")

            yield f"event: progress\ndata: {json.dumps({'step': 'executing_sql', 'dataset': matched_id})}\n\n"

            sql_result = agent.execute_sql_query(
                query=request.query, df_id=matched_id, client_id=safe_client
            )

            if sql_result and sql_result.success:
                result_text = str(sql_result.result)
                method = sql_result.method
                for word in result_text.split():
                    yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
                    await asyncio.sleep(0.01)
            else:
                result_text = (
                    sql_result.error if sql_result else "Query execution failed"
                )
                method = "error"

        else:
            yield f"event: progress\ndata: {json.dumps({'step': 'llm_generation'})}\n\n"

            try:
                response = llm.invoke(f"Answer this query: {request.query}")
                result_text = response
                method = "llm_direct"
                for word in result_text.split():
                    yield f"event: token\ndata: {json.dumps({'token': word + ' '})}\n\n"
                    await asyncio.sleep(0.01)
            except Exception as e:
                result_text = f"Error: {str(e)}"
                method = "error"

        payload: Dict[str, Any] = {
            "query_id": query_id,
            "result": result_text,
            "method": method,
            "success": method != "error",
        }
        yield f"event: complete\ndata: {json.dumps(payload)}\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"event: error\ndata: {json.dumps({'error': str(e), 'query_id': query_id})}\n\n"
