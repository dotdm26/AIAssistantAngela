from typing import List, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def estimate_token_count(messages: List[Union[HumanMessage, AIMessage, SystemMessage]], extract_text_fn) -> int:
	"""Fast approximation for observability: ~4 chars per token for English-heavy text."""
	total_chars = 0
	for message in messages:
		total_chars += len(extract_text_fn(getattr(message, "content", "")))
	return max(1, total_chars // 4)


def log_token_estimate_input(token_count: int, message_count: int, hybrid_used: bool) -> None:
	print(
		f"[token_estimate] input~{token_count} tokens | "
		f"messages={message_count} | hybrid_used={hybrid_used}"
	)


def log_token_estimate_output(output_text: str) -> None:
	print(f"[token_estimate] output~{max(1, len(output_text) // 4)} tokens")


def log_token_usage(response) -> None:
	if hasattr(response, "usage_metadata"):
		print(f"[token_usage] {response.usage_metadata}")
		return

	if hasattr(response, "response_metadata"):
		usage = getattr(response, "response_metadata", {}).get("usage_metadata")
		if usage:
			print(f"[token_usage] {usage}")


def log_hybrid_filter_skipped(top_score: float, threshold: float) -> None:
	print(
		f"[hybrid_filter] skipped retrieval context | "
		f"top_score={top_score:.3f} < threshold={threshold:.3f}"
	)


def log_tool_filter_selected(selected_tool_names) -> None:
	print(f"[tool_filter] selected_tools={selected_tool_names}")


def log_tool_filter_fallback(exc: Exception) -> None:
	print(f"[tool_filter] fallback to full tool set: {exc}")


def log_tool_router_decision(top_score: float, threshold: float, decision: bool) -> None:
	print(
		f"[tool_router] tool_intent=True top_score={top_score:.3f} "
		f"threshold={threshold:.3f} enable_tools={decision}"
	)


def log_tool_router_fallback(exc: Exception) -> None:
	print(f"[tool_router] score check failed, defaulting to intent-only routing: {exc}")
