"""
CA Agent Prompts - The "Brain of Logic" for the Agent.

Production-grade prompt engineering with:
- Token management and smart chunking
- Progressive summarization to avoid rate limits
- Dynamic prompt optimization
- YAML override capability
"""
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
import yaml
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


# ==============================================================================
# TOKEN MANAGEMENT - Production-grade token counting and chunking
# ==============================================================================
@dataclass
class TokenBudget:
    """Token budget configuration for different LLM providers."""
    max_context: int = 8192  # Default context window
    max_output: int = 2048   # Max output tokens
    reserved_system: int = 500  # Reserved for system prompt
    reserved_response: int = 1000  # Reserved for response
    
    @property
    def available_for_content(self) -> int:
        """Tokens available for user content (data, context, query)."""
        return self.max_context - self.reserved_system - self.reserved_response


# Token budgets for different providers
TOKEN_BUDGETS = {
    "gemini": TokenBudget(max_context=32768, max_output=8192, reserved_system=800),
    "gpt-4": TokenBudget(max_context=128000, max_output=4096, reserved_system=800),
    "gpt-3.5": TokenBudget(max_context=16384, max_output=4096, reserved_system=500),
    "claude": TokenBudget(max_context=200000, max_output=4096, reserved_system=1000),
    "groq": TokenBudget(max_context=8192, max_output=2048, reserved_system=500),
    "ollama": TokenBudget(max_context=4096, max_output=2048, reserved_system=400),
    "default": TokenBudget(max_context=4096, max_output=2048, reserved_system=400),
}


class TokenManager:
    """
    Production-grade token management with:
    - Accurate token estimation (tiktoken-compatible)
    - Smart chunking strategies
    - Progressive summarization
    - Rate limit protection
    """
    
    # Average characters per token (empirical for English text)
    CHARS_PER_TOKEN = 4.0
    
    def __init__(self, provider: str = "default"):
        self.provider = provider
        self.budget = TOKEN_BUDGETS.get(provider, TOKEN_BUDGETS["default"])
        self._tiktoken_available = False
        self._encoder = None
        
        # Try to load tiktoken for accurate counting
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding("cl100k_base")
            self._tiktoken_available = True
        except ImportError:
            logger.debug("tiktoken not available, using character-based estimation")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (accurate with tiktoken, estimated otherwise)."""
        if not text:
            return 0
        if self._tiktoken_available and self._encoder:
            return len(self._encoder.encode(text))
        # Fallback: character-based estimation
        return int(len(text) / self.CHARS_PER_TOKEN)
    
    def fits_in_context(self, text: str, reserved: int = 0) -> bool:
        """Check if text fits in available context window."""
        tokens = self.count_tokens(text)
        available = self.budget.available_for_content - reserved
        return tokens <= available
    
    def truncate_to_fit(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit, preserving structure."""
        current_tokens = self.count_tokens(text)
        if current_tokens <= max_tokens:
            return text
        
        # Estimate characters to keep
        ratio = max_tokens / current_tokens
        target_chars = int(len(text) * ratio * 0.95)  # 5% safety margin
        
        # Try to truncate at a natural boundary
        truncated = text[:target_chars]
        
        # Try to end at sentence or line boundary
        for boundary in ['\n\n', '\n', '. ', ', ']:
            last_boundary = truncated.rfind(boundary)
            if last_boundary > len(truncated) * 0.7:
                truncated = truncated[:last_boundary + len(boundary)]
                break
        
        return truncated + "\n...[truncated]"
    
    def smart_chunk(
        self,
        text: str,
        chunk_size: int,
        overlap: int = 100,
        preserve_structure: bool = True
    ) -> List[str]:
        """
        Smart chunking with structure awareness.
        
        Args:
            text: Text to chunk
            chunk_size: Target tokens per chunk
            overlap: Token overlap between chunks (for context continuity)
            preserve_structure: Try to preserve paragraph/section boundaries
            
        Returns:
            List of text chunks
        """
        if self.count_tokens(text) <= chunk_size:
            return [text]
        
        chunks = []
        
        if preserve_structure:
            # Split on structural boundaries first
            sections = re.split(r'\n\n+', text)
            current_chunk = []
            current_tokens = 0
            
            for section in sections:
                section_tokens = self.count_tokens(section)
                
                if current_tokens + section_tokens <= chunk_size:
                    current_chunk.append(section)
                    current_tokens += section_tokens
                else:
                    # Save current chunk
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                    
                    # Handle large sections
                    if section_tokens > chunk_size:
                        # Further split large sections
                        sub_chunks = self._split_large_section(section, chunk_size)
                        chunks.extend(sub_chunks[:-1])
                        current_chunk = [sub_chunks[-1]] if sub_chunks else []
                        current_tokens = self.count_tokens(current_chunk[0]) if current_chunk else 0
                    else:
                        current_chunk = [section]
                        current_tokens = section_tokens
            
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
        else:
            # Simple sliding window
            char_chunk_size = int(chunk_size * self.CHARS_PER_TOKEN)
            char_overlap = int(overlap * self.CHARS_PER_TOKEN)
            
            start = 0
            while start < len(text):
                end = min(start + char_chunk_size, len(text))
                chunks.append(text[start:end])
                start = end - char_overlap
        
        return chunks
    
    def _split_large_section(self, section: str, max_tokens: int) -> List[str]:
        """Split a large section that exceeds max_tokens."""
        lines = section.split('\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self.count_tokens(line)
            if current_tokens + line_tokens <= max_tokens:
                current_chunk.append(line)
                current_tokens += line_tokens
            else:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_tokens = line_tokens
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def summarize_for_context(
        self,
        text: str,
        target_tokens: int,
        preserve_key_info: bool = True
    ) -> str:
        """
        Compress text while preserving key information.
        
        Uses extractive summarization (no LLM call) for speed.
        """
        current_tokens = self.count_tokens(text)
        if current_tokens <= target_tokens:
            return text
        
        lines = text.split('\n')
        
        if preserve_key_info:
            # Score lines by importance heuristics
            scored_lines = []
            for i, line in enumerate(lines):
                score = 0
                line_lower = line.lower().strip()
                
                # Boost for key patterns
                if any(kw in line_lower for kw in ['total', 'sum', 'result', 'output', 'answer']):
                    score += 10
                if any(kw in line_lower for kw in ['error', 'warning', 'important', 'note']):
                    score += 8
                if re.search(r'\d+\.?\d*', line):  # Contains numbers
                    score += 5
                if line.startswith('#') or line.startswith('**'):  # Headers
                    score += 7
                if i < 3 or i >= len(lines) - 3:  # First/last lines
                    score += 3
                
                scored_lines.append((score, i, line))
            
            # Sort by score, keep highest
            scored_lines.sort(reverse=True)
            
            # Keep lines until we hit target
            kept_lines = []
            kept_tokens = 0
            
            for score, idx, line in scored_lines:
                line_tokens = self.count_tokens(line)
                if kept_tokens + line_tokens <= target_tokens:
                    kept_lines.append((idx, line))
                    kept_tokens += line_tokens
            
            # Sort by original order
            kept_lines.sort()
            result = '\n'.join(line for _, line in kept_lines)
            
            if kept_tokens < current_tokens:
                result += f"\n...[summarized from {current_tokens} to {kept_tokens} tokens]"
            
            return result
        else:
            # Simple truncation
            return self.truncate_to_fit(text, target_tokens)
    
    def prepare_data_context(
        self,
        data_sample: str,
        schema: str,
        query: str,
        max_data_tokens: Optional[int] = None
    ) -> Tuple[str, str]:
        """
        Prepare data context optimized for LLM consumption.
        
        Returns (optimized_sample, optimized_schema)
        """
        max_tokens = max_data_tokens or (self.budget.available_for_content // 2)
        
        # Schema is usually smaller and more important
        schema_tokens = self.count_tokens(schema)
        sample_budget = max_tokens - min(schema_tokens, max_tokens // 3)
        
        optimized_sample = self.summarize_for_context(data_sample, sample_budget)
        optimized_schema = self.truncate_to_fit(schema, max_tokens // 3)
        
        return optimized_sample, optimized_schema


# Global token manager instance
_token_manager: Optional[TokenManager] = None


def get_token_manager(provider: str = "default") -> TokenManager:
    """Get or create token manager singleton."""
    global _token_manager
    if _token_manager is None or _token_manager.provider != provider:
        _token_manager = TokenManager(provider)
    return _token_manager


# --- Global CA System Guardrails (Optimized for token efficiency) ---
CA_SYSTEM_GUARDRAILS = """You are a Senior Chartered Accountant (CA) expert in:
- Financial Analysis (GAAP/IFRS/Schedule III)
- MIS reporting & Tax compliance
- Forensic accounting

RULES:
1. ACCURACY: Never guess numbers - use data tools
2. SOURCE-FIRST: Only answer from provided data
3. COMPLIANCE: Follow accounting standards
4. FORMAT: Markdown, INR Mn/Cr as per source"""

# --- Track Classification ---
TRACK_DATA = "TRACK_DATA"  # SQL/Pandas analytics
TRACK_DOC = "TRACK_DOC"    # Document RAG / policy lookup
TRACK_WEB = "TRACK_WEB"    # Web search for latest info

# --- Prompt Templates Registry ---
_PROMPT_TEMPLATES: Dict[str, str] = {}


def _load_yaml_templates() -> Dict[str, str]:
    """Load prompt templates from YAML if available."""
    yaml_path = Path(__file__).parent.parent / "prompts" / "templates.yaml"
    if yaml_path.exists():
        try:
            with open(yaml_path, encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                return data.get("templates", {})
        except Exception as e:
            logger.warning(f"Failed to load prompt templates YAML: {e}")
    return {}


# Initialize templates from YAML or use defaults
_PROMPT_TEMPLATES = _load_yaml_templates()


# ==============================================================================
# CHART EXTRACTION - Extract and validate chart JSON from LLM responses
# ==============================================================================
class ChartExtractor:
    """
    Production-grade chart extraction from LLM responses.
    
    Features:
    - Extracts chart JSON from markdown code blocks
    - Validates against Recharts-compatible schema
    - Normalizes data for frontend consumption
    - Handles malformed JSON gracefully
    """
    
    # Supported chart types (Recharts compatible)
    VALID_CHART_TYPES = {"line", "area", "bar", "scatter", "pie", "radar", "composed"}
    
    # Chart JSON schema for validation
    CHART_SCHEMA = {
        "required": ["type", "title", "data"],
        "optional": ["xAxis", "yAxis", "composed", "colors", "legend"]
    }
    
    @classmethod
    def extract_charts(cls, response: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract chart JSON from LLM response.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Tuple of (text_without_chart_json, list_of_chart_objects)
        """
        charts = []
        clean_text = response
        
        # Pattern to match JSON code blocks containing charts
        json_pattern = r'```json\s*(\{[\s\S]*?"charts"[\s\S]*?\})\s*```'
        matches = re.findall(json_pattern, response, re.IGNORECASE)
        
        for match in matches:
            try:
                parsed = cls._parse_chart_json(match)
                if parsed:
                    charts.extend(parsed)
                    # Remove the JSON block from text
                    clean_text = re.sub(
                        r'```json\s*' + re.escape(match) + r'\s*```',
                        '',
                        clean_text,
                        flags=re.IGNORECASE
                    )
            except Exception as e:
                logger.debug(f"Failed to parse chart JSON: {e}")
        
        # Also try to find standalone chart objects
        standalone_pattern = r'(\{[^{}]*"charts"\s*:\s*\[[^\]]+\][^{}]*\})'
        for match in re.findall(standalone_pattern, response):
            if match not in str(matches):  # Avoid duplicates
                try:
                    parsed = cls._parse_chart_json(match)
                    if parsed:
                        charts.extend(parsed)
                except Exception:
                    pass
        
        return clean_text.strip(), charts
    
    @classmethod
    def _parse_chart_json(cls, json_str: str) -> Optional[List[Dict[str, Any]]]:
        """Parse and validate chart JSON."""
        import json
        
        try:
            # Clean common JSON issues
            cleaned = json_str.strip()
            cleaned = re.sub(r',\s*}', '}', cleaned)  # Remove trailing commas
            cleaned = re.sub(r',\s*]', ']', cleaned)
            
            data = json.loads(cleaned)
            
            # Extract charts array
            if isinstance(data, dict) and "charts" in data:
                charts = data["charts"]
            elif isinstance(data, list):
                charts = data
            else:
                return None
            
            # Validate and normalize each chart
            valid_charts = []
            for chart in charts:
                validated = cls._validate_chart(chart)
                if validated:
                    valid_charts.append(validated)
            
            return valid_charts if valid_charts else None
            
        except json.JSONDecodeError:
            return None
    
    @classmethod
    def _validate_chart(cls, chart: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate a single chart object against schema."""
        # Check required fields
        for field in cls.CHART_SCHEMA["required"]:
            if field not in chart:
                logger.debug(f"Chart missing required field: {field}")
                return None
        
        # Validate chart type
        chart_type = chart.get("type", "").lower()
        if chart_type not in cls.VALID_CHART_TYPES:
            logger.debug(f"Invalid chart type: {chart_type}")
            return None
        
        # Validate data array
        data = chart.get("data", [])
        if not isinstance(data, list) or len(data) == 0:
            logger.debug("Chart data must be a non-empty array")
            return None
        
        # Normalize the chart object
        normalized = {
            "type": chart_type,
            "title": str(chart.get("title", "Chart")),
            "data": cls._normalize_chart_data(data),
            "xAxis": str(chart.get("xAxis", "")),
            "yAxis": str(chart.get("yAxis", "")),
        }
        
        # Handle composed charts
        if chart_type == "composed" and "composed" in chart:
            normalized["composed"] = chart["composed"]
        
        return normalized
    
    @classmethod
    def _normalize_chart_data(cls, data: List[Dict]) -> List[Dict]:
        """Normalize chart data for Recharts compatibility."""
        normalized = []
        for item in data:
            if not isinstance(item, dict):
                continue
            
            clean_item = {}
            for key, value in item.items():
                # Ensure numeric values are actually numbers
                if isinstance(value, str):
                    try:
                        # Try to parse as number
                        if '.' in value:
                            clean_item[key] = float(value)
                        else:
                            clean_item[key] = int(value)
                    except ValueError:
                        clean_item[key] = value
                else:
                    clean_item[key] = value
            
            normalized.append(clean_item)
        
        return normalized
    
    # Cached semantic matcher instance for chart detection
    _semantic_matcher = None
    
    @classmethod
    def _get_semantic_matcher(cls):
        """Lazy-load SemanticMatcher for financial synonym matching."""
        if cls._semantic_matcher is None:
            try:
                from app.core.semantic_understanding import SemanticMatcher
                cls._semantic_matcher = SemanticMatcher()
                logger.debug("ChartExtractor loaded SemanticMatcher for intelligent intent detection")
            except ImportError:
                logger.debug("SemanticMatcher not available, using basic keyword matching")
        return cls._semantic_matcher
    
    @classmethod
    def detect_chart_intent(cls, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if user wants a chart and what type.
        Uses SemanticMatcher for domain-aware synonym expansion.
        
        Returns:
            Tuple of (wants_chart, suggested_type)
        """
        query_lower = query.lower()
        
        # Explicit chart requests
        chart_keywords = [
            "chart", "graph", "plot", "visualize", "visualization",
            "show me a", "draw", "display chart", "create chart"
        ]
        
        wants_chart = any(kw in query_lower for kw in chart_keywords)
        
        # Detect specific chart type
        chart_type = None
        type_hints = {
            "line": ["line chart", "line graph", "trend", "over time", "timeline"],
            "bar": ["bar chart", "bar graph", "comparison", "compare", "breakdown by"],
            "pie": ["pie chart", "pie graph", "breakdown", "distribution", "share", "proportion"],
            "area": ["area chart", "cumulative", "stacked area", "filled"],
            "scatter": ["scatter", "correlation", "relationship between", "x vs y"],
            "radar": ["radar", "spider", "multi-dimensional", "multi-metric"],
            "composed": ["combined", "dual axis", "bar and line", "mixed chart"],
        }
        
        for ctype, hints in type_hints.items():
            if any(hint in query_lower for hint in hints):
                chart_type = ctype
                break
        
        # ================================================================
        # SEMANTIC CHART INTENT DETECTION
        # Use SemanticMatcher to expand financial synonyms for better matching
        # ================================================================
        
        # Core trend/comparison keywords that suggest visualization
        trend_base_keywords = ["trend", "growth", "yoy", "cagr", "comparison"]
        
        # Get expanded synonyms using SemanticMatcher (if available)
        expanded_trend_keywords = set(trend_base_keywords)
        semantic_matcher = cls._get_semantic_matcher()
        
        if semantic_matcher:
            # Expand each base keyword with financial synonyms
            for keyword in trend_base_keywords:
                try:
                    synonyms = semantic_matcher.get_financial_synonyms(keyword)
                    expanded_trend_keywords.update(synonyms)
                except Exception:
                    pass
            
            # Also add explicit financial patterns that suggest visualization
            expanded_trend_keywords.update({
                # Time-based patterns (line charts)
                "year on year", "year over year", "y-o-y",
                "quarter on quarter", "quarter over quarter", "q-o-q",
                "month on month", "month over month", "m-o-m",
                "compound annual growth rate", "compound growth",
                "over the years", "across periods", "time series",
                "historical", "trajectory", "progression",
                
                # Comparison patterns (bar charts)
                "breakdown", "by category", "by segment", "by product",
                "vs", "versus", "compared to", "relative to",
                
                # Proportion patterns (pie charts)
                "share of", "portion of", "percentage breakdown",
                "distribution of", "split between", "composition",
            })
        
        # Check for any trend/visualization keywords
        if not wants_chart:
            for kw in expanded_trend_keywords:
                if kw in query_lower:
                    wants_chart = True
                    # Infer chart type from keyword if not already set
                    if chart_type is None:
                        if kw in {"breakdown", "share of", "portion of", "percentage breakdown", 
                                  "distribution of", "split between", "composition"}:
                            chart_type = "pie"
                        elif kw in {"vs", "versus", "compared to", "relative to", 
                                    "by category", "by segment", "by product"}:
                            chart_type = "bar"
                        else:
                            chart_type = "line"  # Default for trend keywords
                    break
        
        return wants_chart, chart_type
    
    @classmethod
    def format_response_with_charts(
        cls, 
        text: str, 
        charts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Format response with separated text and charts.
        
        Returns:
            {
                "text": "markdown text without chart JSON",
                "charts": [...],
                "has_charts": bool
            }
        """
        return {
            "text": text,
            "charts": charts,
            "has_charts": len(charts) > 0
        }


def get_router_prompt(query: str, client_context: str) -> str:
    """
    Classifies intent into: TRACK_DATA, TRACK_DOC, or TRACK_WEB.
    Uses semantic understanding, not keyword matching.
    
    Args:
        query: User's question
        client_context: Context about the client's available datasets
        
    Returns:
        Formatted prompt for router classification
    """
    template_str = _PROMPT_TEMPLATES.get("router", """
{guardrails}

CLIENT/DATA CONTEXT:
{context}

USER QUERY: "{query}"

UNDERSTAND THE USER'S INTENT semantically (not just keywords) and classify into ONE track:

TRACK_DATA: Use when the user wants to:
- Query, analyze, or explore LOADED DATA (spreadsheets, CSV, Excel files)
- Get counts, sums, averages, totals, or any calculations FROM data
- Find specific values, records, or metrics in datasets
- Ask about sheet names, columns, rows, or data structure
- Ask "what is [term]" when that term might exist in loaded data
- SEARCH/FIND content in data: "Is there any mention of...", "Find company names", "Does it contain..."
- Entity extraction: "What companies are mentioned?", "List all names in the data"
- Content verification: "Check if there is...", "Verify if exists..."
- Any question that can be answered by looking at tabular data
EXAMPLES: "how many sheets", "what are the sheet names", "total revenue", 
          "what is nifty", "show volume gainers", "list top stocks",
          "is there any mention of a company name", "find company names"

TRACK_DOC: Use when the user wants to:
- Look up definitions, policies, or clauses from DOCUMENTS (PDFs, contracts)
- Find legal terms, compliance text, or regulatory content
- Search through unstructured text documents (not spreadsheets)
- Answer questions about document content that requires reading prose
EXAMPLES: "what does clause 5 say", "find the cancellation policy", 
          "accounting policy for inventory"

TRACK_WEB: Use when the user wants to:
- Get CURRENT, REAL-TIME, or LATEST information from the internet
- Find external data not in loaded files (rates, news, regulations)
- Answer questions requiring up-to-date information
EXAMPLES: "current repo rate", "latest GST rules", "today's market news"

IMPORTANT DISTINCTION - Content Search vs Summary:
- "Is there any mention of X?" = SEARCH in data → TRACK_DATA (entity extraction)
- "Give me a summary" = SUMMARIZE data → handled separately, but also TRACK_DATA context
- "Does it contain X?" = SEARCH → TRACK_DATA
- "What is this about?" = Summary request → handled separately

DECISION PRIORITY:
1. If query is a SEARCH/FIND/MENTION query with loaded data → TRACK_DATA (entity extraction)
2. If data is loaded AND query seems related to analyzing that data → TRACK_DATA
3. If asking about document content/policies → TRACK_DOC  
4. If needing real-time/external info → TRACK_WEB
5. When in doubt with loaded data → TRACK_DATA

OUTPUT: Return ONLY the track code (TRACK_DATA, TRACK_DOC, or TRACK_WEB).
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "query", "context"],
        template=template_str
    )
    return template.format(guardrails=CA_SYSTEM_GUARDRAILS, query=query, context=client_context)


def get_financial_advisor_prompt(
    query: str,
    context: str,
    previous_context: str = "",
    chat_history: str = "",
    force_chart: bool = False,
    chart_type_hint: Optional[str] = None
) -> Tuple[str, str]:
    """
    Get financial advisor prompt with visualization support.
    
    Args:
        query: User's question
        context: Data context from RAG/SQL results
        previous_context: Previous conversation context
        chat_history: Full chat history
        force_chart: If True, explicitly request chart generation
        chart_type_hint: Suggested chart type (line, bar, pie, etc.)
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Get system prompt from YAML or use default
    system_prompt = _PROMPT_TEMPLATES.get("financial_advisor_system", """
You are a financial advisor with expertise in M&A and due diligence.
Use the provided context and conversation to answer questions professionally.

CORE RULES:
1. All figures should be in the appropriate currency based on the data
2. Provide answers in GitHub-preferred markdown format
3. Use structured markdown tables for data and figures
4. If visualization is needed, provide JSON for charts in a code block
5. Do not include unnecessary explanations or greetings
6. If context is insufficient, respond: "I don't have necessary info for your query"
7. Ensure output is concise and directly usable by the frontend
8. Use natural, conversational language that business users understand
9. Explain what numbers mean in practical terms
10. Avoid technical terms or jargon - audience is business users
""")
    
    # Get user prompt template
    user_template = _PROMPT_TEMPLATES.get("financial_advisor_response", """
{guardrails}

CONTEXT: {context}
PREVIOUS CONTEXT: {previous_context}
CONVERSATION HISTORY: {chat_history}

QUERY: {query}

RESPONSE GUIDELINES:
- Lead with the direct answer (no preambles like "Based on the data...")
- Use markdown tables for tabular data
- For trends, mention if increasing, decreasing, or stable
- No greetings or salutations
- Provide context about what the user is looking at

{chart_instruction}

OUTPUT: Markdown response with optional chart JSON block.
""")
    
    # Build chart instruction based on intent
    if force_chart:
        chart_instruction = f"""
CHART REQUIRED: Generate a {chart_type_hint or 'appropriate'} chart for this data.
Use the following JSON format in a code block:
```json
{{
  "charts": [
    {{
      "type": "{chart_type_hint or 'line'}",
      "title": "Descriptive title",
      "data": [{{"name": "label", "value": 100}}],
      "xAxis": "X Label",
      "yAxis": "Y Label"
    }}
  ]
}}
```
"""
    else:
        chart_instruction = """
CHART GENERATION (if applicable):
Include a chart JSON when visualization adds value (trends, comparisons, proportions).
Format: ```json {"charts": [{"type": "line|bar|pie|...", "title": "...", "data": [...], "xAxis": "...", "yAxis": "..."}]} ```
"""
    
    # Format the user prompt
    user_prompt = PromptTemplate(
        input_variables=["guardrails", "context", "previous_context", "chat_history", "query", "chart_instruction"],
        template=user_template
    ).format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        context=context or "No data context available.",
        previous_context=previous_context or "None",
        chat_history=chat_history or "None",
        query=query,
        chart_instruction=chart_instruction
    )
    
    return system_prompt, user_prompt


def get_chart_generation_prompt(
    data: str,
    query: str,
    chart_type_hint: Optional[str] = None
) -> str:
    """
    Get prompt for dedicated chart generation.
    
    Args:
        data: Data to visualize (JSON or tabular)
        query: What kind of chart/analysis user wants
        chart_type_hint: Suggested chart type
        
    Returns:
        Formatted prompt for chart-only generation
    """
    template_str = _PROMPT_TEMPLATES.get("chart_generation", """
Generate a chart/visualization from the provided data.

DATA: {data}
QUERY: {query}
CHART TYPE HINT: {chart_type_hint}

OUTPUT JSON ONLY (no markdown, no explanation):
{{
  "charts": [
    {{
      "type": "line",
      "title": "Chart Title",
      "data": [{{"name": "Label", "value": 100}}],
      "xAxis": "X Label",
      "yAxis": "Y Label"
    }}
  ]
}}
""")
    
    return PromptTemplate(
        input_variables=["data", "query", "chart_type_hint"],
        template=template_str
    ).format(
        data=data,
        query=query,
        chart_type_hint=chart_type_hint or "auto-detect"
    )


def get_data_analyst_sql_prompt(schema_info: str, query: str, available_columns: str = "", data_sample: str = "", semantic_info: str = "") -> str:
    """
    Generates DuckDB SQL for data analysis queries using semantic data understanding.
    
    Args:
        schema_info: Database schema information
        query: User's analytical question
        available_columns: Optional list of available columns
        data_sample: Sample rows from the actual data
        semantic_info: Semantic understanding of the data (period columns, label column, etc.)
        
    Returns:
        Formatted prompt for SQL generation
    """
    template_str = _PROMPT_TEMPLATES.get("data_analyst_sql", """
{guardrails}

DATABASE SCHEMA (DuckDB SQL):
{schema}

{columns_section}

{data_sample_section}

{semantic_section}

CRITICAL DuckDB SQL RULES:
1. Use ONLY the exact column names shown above - they are lowercase with underscores.
2. ALL numeric operations MUST use explicit CAST: CAST(column AS DOUBLE) for calculations.
3. Handle mixed types: Use TRY_CAST(column AS DOUBLE) for columns that might have non-numeric values.
4. String comparisons are case-sensitive - use LOWER() for matching text.
5. For NULL-like values, the data is pre-cleaned but use COALESCE(col, 0) for safety.
6. The table name is the sanitized version shown in schema - use EXACTLY that name.

ANTI-HALLUCINATION RULES (CRITICAL):
- DO NOT invent table names like 'customer_data', 'customers', 'users', 'orders', etc.
- ONLY use the EXACT table name provided in DATABASE SCHEMA above.
- ONLY use column names that EXIST in the schema - DO NOT make up columns.
- If the requested data doesn't exist in the schema, return a query that filters for it.
- NEVER generate CREATE TABLE, DROP, DELETE, INSERT, or UPDATE statements.

QUERY PATTERNS:
- For "growth from X to Y": Calculate (Y_value - X_value) using the appropriate columns
- For "value in period Z": Select the value from column matching period Z
- For aggregations: Use SUM(), AVG(), COUNT() with explicit CAST to DOUBLE
- For row filtering: Use WHERE with the label column to find specific metrics

USER QUERY: {query}

OUTPUT FORMAT:
Return ONLY valid JSON with this structure:
{{
    "sql": "SELECT CAST(... AS DOUBLE) FROM table_name WHERE ...",
    "columns_used": ["col1", "col2"],
    "explanation": "What this query does and how it answers the question"
}}
""")
    
    columns_section = f"AVAILABLE COLUMNS:\n{available_columns}" if available_columns else ""
    data_sample_section = f"DATA SAMPLE (first few rows for context):\n{data_sample}" if data_sample else ""
    semantic_section = f"DATA STRUCTURE UNDERSTANDING:\n{semantic_info}" if semantic_info else ""
    
    template = PromptTemplate(
        input_variables=["guardrails", "schema", "query", "columns_section", "data_sample_section", "semantic_section"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        schema=schema_info,
        query=query,
        columns_section=columns_section,
        data_sample_section=data_sample_section,
        semantic_section=semantic_section
    )


def get_data_analyst_python_prompt(schema_info: str, query: str, sample_data: str = "") -> str:
    """
    Generates Python/Pandas code for complex analysis when SQL is insufficient.
    
    Args:
        schema_info: DataFrame schema
        query: User's question
        sample_data: Optional sample of the data
        
    Returns:
        Formatted prompt for Python code generation
    """
    template_str = _PROMPT_TEMPLATES.get("data_analyst_python", """
{guardrails}

DATAFRAME SCHEMA:
{schema}

SAMPLE DATA:
{sample}

USER QUERY: {query}

Generate Python code that:
1. Works with a pandas DataFrame named 'df' (already loaded).
2. Defines a function 'run(df)' that returns the result.
3. Uses only: pd (pandas), np (numpy), math, decimal, datetime - ALREADY AVAILABLE IN SCOPE.
4. Returns a JSON-serializable result (dict, list, number, or string).
5. Handles missing values gracefully with pd.isna() or pd.notna().

CRITICAL RESTRICTIONS (code will be REJECTED if violated):
- DO NOT use import statements - modules are pre-loaded
- DO NOT use __import__, eval(), exec(), open(), or getattr()
- DO NOT access files, networks, or system resources
- DO NOT use globals(), locals(), or vars()
- Use 'pd' not 'pandas', 'np' not 'numpy'

OUTPUT FORMAT:
Return ONLY valid JSON:
{{
    "code": "def run(df):\\n    # your code here\\n    return result",
    "explanation": "Brief explanation of the approach"
}}
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "schema", "query", "sample"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        schema=schema_info,
        query=query,
        sample=sample_data or "Not provided"
    )


def get_ca_synthesis_prompt(results: str, query: str, method: str = "unknown") -> str:
    """
    Final synthesis - converts raw results into professional CA response.
    
    Args:
        results: Raw tool/query results
        query: Original user question
        method: Method used (sql_duckdb, pandas, llm_estimation)
        
    Returns:
        Formatted prompt for response synthesis
    """
    template_str = _PROMPT_TEMPLATES.get("synthesis", """
{guardrails}

RAW RESULTS FROM {method}:
{results}

ORIGINAL QUERY: {query}

INSTRUCTIONS:
1. Summarize findings professionally and concisely.
2. If numeric, present with appropriate precision and units.
3. If variance/anomaly exists, explain in plain terms.
4. Note the data source and calculation method used.
5. Add a brief 'Note' if any assumptions were made.

OUTPUT FORMAT:
Provide a clear, professional response suitable for presentation to management.
If result is uncertain, add a disclaimer.
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "results", "query", "method"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        results=results,
        query=query,
        method=method.upper()
    )


def get_document_rag_prompt(context: str, query: str) -> str:
    """
    Document RAG prompt for policy/clause lookups.
    
    Args:
        context: Retrieved document chunks
        query: User's question
        
    Returns:
        Formatted prompt for document QA
    """
    template_str = _PROMPT_TEMPLATES.get("document_rag", """
{guardrails}

DOCUMENT CONTEXT:
{context}

USER QUERY: {query}

INSTRUCTIONS:
1. Answer based ONLY on the provided context.
2. Quote relevant sections when appropriate.
3. If the answer is not in the context, say "Information not found in available documents".
4. For legal/compliance matters, recommend verification with original source.

OUTPUT: Provide a clear, accurate answer with source references.
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "context", "query"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        context=context,
        query=query
    )


def get_web_search_prompt(search_results: str, query: str) -> str:
    """
    Web search synthesis prompt.
    
    Args:
        search_results: Web search snippets
        query: User's question
        
    Returns:
        Formatted prompt for web result synthesis
    """
    template_str = _PROMPT_TEMPLATES.get("web_search", """
{guardrails}

WEB SEARCH RESULTS:
{results}

USER QUERY: {query}

INSTRUCTIONS:
1. Synthesize information from credible sources.
2. Cite sources with URLs.
3. Note the date relevance of information.
4. For tax rates/regulations, recommend verification with official sources.

OUTPUT: Provide factual answer with sources. Add date caveat if time-sensitive.
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "results", "query"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        results=search_results,
        query=query
    )


def get_dataset_match_prompt(query: str, datasets_info: str) -> str:
    """
    Dataset matching prompt to find the best dataset for a query.
    
    Args:
        query: User's question
        datasets_info: Information about available datasets
        
    Returns:
        Formatted prompt for dataset selection
    """
    template_str = _PROMPT_TEMPLATES.get("dataset_match", """
You are a data matching assistant. Match the user's query to the most relevant dataset.

AVAILABLE DATASETS:
{datasets}

USER QUERY: {query}

RULES:
1. Match based on column names, sheet names, and data descriptions.
2. Look for keywords in the query that match dataset content.
3. Consider time periods mentioned (FY21, 9MFY22, etc.).

OUTPUT FORMAT:
Return ONLY the dataset_id of the best match, or "NONE" if no match found.
""")
    
    template = PromptTemplate(
        input_variables=["datasets", "query"],
        template=template_str
    )
    return template.format(datasets=datasets_info, query=query)


def get_metadata_query_prompt(query: str, available_datasets: str, schema_info: str = "") -> str:
    """
    Generate prompt for understanding and answering metadata queries semantically.
    No hardcoding - LLM determines if this is a metadata query and how to answer.
    
    Args:
        query: User's question
        available_datasets: List of available datasets/sheets
        schema_info: Optional schema information
        
    Returns:
        Formatted prompt for metadata query handling
    """
    template_str = _PROMPT_TEMPLATES.get("metadata_query", """
You are a data assistant. Analyze the user's query and determine if they are asking about 
the STRUCTURE or METADATA of loaded data (not the data values themselves).

METADATA QUERIES include questions about:
- Number of sheets/tables/datasets
- Names of sheets/tables/datasets
- Column names or structure
- Data types or schema
- File information

AVAILABLE DATA:
{datasets}

{schema_section}

USER QUERY: "{query}"

INSTRUCTIONS:
1. Determine if this is a METADATA query (about structure) or a DATA query (about values)
   Matches METADATA: "what are the sheet names", "list all tables", "how many columns", "show me schema"
   Matches DATA (NOT metadata): "summary of sheet X", "calculate total", "show me the data", "value of X", "analyze trends"

2. If it is a DATA query (asking for summary, values, analysis), return "is_metadata_query": false
3. If METADATA query, provide a direct, complete answer
4. Include all relevant information (e.g., ALL sheet names, not just some)
5. Format the answer clearly and professionally

OUTPUT FORMAT:
Return JSON:
{{
    "is_metadata_query": true/false,
    "answer": "Your complete answer if metadata query, or null",
    "query_type": "sheet_count|sheet_names|column_info|schema|other|not_metadata"
}}
""")
    
    schema_section = f"SCHEMA INFORMATION:\n{schema_info}" if schema_info else ""
    
    template = PromptTemplate(
        input_variables=["datasets", "query", "schema_section"],
        template=template_str
    )
    return template.format(datasets=available_datasets, query=query, schema_section=schema_section)


# Tool placeholder descriptions for agent prompts
TOOL_DESCRIPTIONS = {
    "tool_benford": "benford_test: Runs Benford's Law test on numeric column to detect anomalies/fraud",
    "tool_3way": "three_way_match: Compares invoice, PO, and ledger for discrepancies (reconciliation)",
    "tool_cohort": "cohort_analysis: Builds cohort retention analysis from transaction data",
    "tool_web_search": "web_search: Multi-provider web search for current tax rates, regulations, benchmarks",
    "tool_search_tax": "search_tax_rate: Specialized search for tax rates from official sources",
    "tool_search_regulation": "search_regulation: Search for regulatory and compliance requirements",
    "tool_search_benchmark": "search_benchmark: Search for industry benchmarks and market data",
    "tool_date_normalize": "date_normalize: Normalizes various date formats to standard format",
}


def get_tool_description(tool_name: str) -> str:
    """Get description for a tool by name."""
    return TOOL_DESCRIPTIONS.get(tool_name, f"{tool_name}: No description available")


def get_tool_selection_prompt(query: str, available_tools: str = "") -> str:
    """
    Generate prompt for LLM to select appropriate tool.
    
    Args:
        query: User's question
        available_tools: Optional list of available tools
        
    Returns:
        Formatted prompt for tool selection
    """
    if not available_tools:
        available_tools = "\n".join([
            f"- {key}: {desc}" for key, desc in TOOL_DESCRIPTIONS.items()
        ])
    
    template_str = _PROMPT_TEMPLATES.get("tool_selection", """
{guardrails}

USER QUERY: {query}

AVAILABLE TOOLS:
{tools}

DETERMINE:
1. Which tool is most appropriate for this query?
2. What parameters are needed?
3. Can this be answered without a tool (pure data query)?

OUTPUT FORMAT:
Return ONLY valid JSON:
{{
    "tool": "tool_name or null",
    "parameters": {{"param1": "value1"}},
    "reason": "Why this tool was selected"
}}
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "query", "tools"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        query=query,
        tools=available_tools
    )


def get_fraud_analysis_prompt(results: str, column: str, dataset: str) -> str:
    """
    Generate prompt for fraud detection analysis interpretation.
    
    Args:
        results: Benford test results
        column: Column that was tested
        dataset: Dataset ID
        
    Returns:
        Formatted prompt for fraud analysis
    """
    template_str = _PROMPT_TEMPLATES.get("fraud_analysis", """
{guardrails}

BENFORD'S LAW TEST RESULTS:
{results}

TESTED COLUMN: {column}
DATASET: {dataset}

INTERPRET FINDINGS:
1. PASS: Data follows expected Benford distribution - no immediate red flags.
2. WARNING: Minor deviations detected - may warrant further investigation.
3. FAIL: Significant deviation from Benford's Law - potential data manipulation or fraud.

FOR FAILURES:
- Identify which digits deviate most from expected distribution
- Suggest specific line items or date ranges to investigate
- Recommend additional audit procedures

OUTPUT: Professional fraud analysis report suitable for audit committee.
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "results", "column", "dataset"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        results=results,
        column=column,
        dataset=dataset
    )


def get_reconciliation_prompt(results: str) -> str:
    """
    Generate prompt for reconciliation summary interpretation.
    
    Args:
        results: Three-way match results
        
    Returns:
        Formatted prompt for reconciliation report
    """
    template_str = _PROMPT_TEMPLATES.get("reconciliation", """
{guardrails}

THREE-WAY MATCH RESULTS:
{results}

RECONCILIATION SUMMARY:
- Matched: Transactions where Invoice, PO, and Ledger agree within tolerance
- Unmatched: Discrepancies requiring investigation

FOR DISCREPANCIES:
1. Categorize by type (price variance, quantity variance, timing difference)
2. Quantify the financial impact
3. Suggest resolution steps
4. Flag any that require management attention

OUTPUT: Professional reconciliation report suitable for CFO review.
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "results"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        results=results
    )


# Output schema definitions for validation
OUTPUT_SCHEMAS = {
    "sql_response": {
        "sql": str,
        "columns_used": list,
        "explanation": str
    },
    "python_response": {
        "code": str,
        "explanation": str
    },
    "analysis_result": {
        "value": (float, int, str, type(None)),
        "method": str,
        "explain": str,
        "confidence": float
    },
    "tool_selection": {
        "tool": str,
        "parameters": dict,
        "reason": str
    },
    "schema_analysis": {
        "label_column": str,
        "period_columns": dict,
        "header_row": int,
        "data_start_row": int,
        "data_type": str
    }
}


def get_output_schema(schema_name: str) -> Dict[str, Any]:
    """Get expected output schema by name."""
    return OUTPUT_SCHEMAS.get(schema_name, {})


def get_dataset_summary_prompt(
    file_name: str,
    dataset_id: str,
    sheet_names: List[str],
    total_rows: int,
    total_cols: int,
    top_columns: List[str],
    sample_rows: str,
    detected_periods: str = "Not detected",
    detected_currency: str = "Not detected",
    detected_fact_types: str = "Not detected"
) -> str:
    """
    Generate prompt for dataset summary queries.
    
    Args:
        file_name: Original filename
        dataset_id: Dataset identifier
        sheet_names: List of sheet names
        total_rows: Number of rows
        total_cols: Number of columns
        top_columns: First N column names
        sample_rows: Sample data (3 rows max)
        detected_periods: Detected time periods
        detected_currency: Detected currency
        detected_fact_types: Detected data types
        
    Returns:
        Formatted prompt for dataset summary
    """
    template_str = _PROMPT_TEMPLATES.get("dataset_summary", """
{guardrails}

You are analyzing an uploaded data file for a client. Generate a professional summary.

FILE INFORMATION:
- File Name: {file_name}
- Dataset ID: {dataset_id}
- Sheet Names: {sheet_names}
- Total Rows: {total_rows}
- Total Columns: {total_cols}
- Top Columns: {top_columns}

SAMPLE DATA (first 3 rows):
{sample_rows}

INSTRUCTIONS:
1. Write a 4-6 sentence overview describing what this dataset contains
2. Identify the apparent purpose (MIS report, financial statement, ledger, etc.)
3. List key columns/metrics with their apparent meaning
4. Note any time periods or currencies detected
5. Be factual - only describe what you observe in the data

OUTPUT FORMAT:
**Dataset Overview:**
[4-6 sentences describing the data]

**Key Fields Identified:**
- [column_name]: [apparent meaning/mapping]

I only use information from the uploaded data - I do not make assumptions about data that is not present.
""")
    
    template = PromptTemplate(
        input_variables=[
            "guardrails", "file_name", "dataset_id", "sheet_names", 
            "total_rows", "total_cols", "top_columns", "sample_rows",
            "detected_periods", "detected_currency", "detected_fact_types"
        ],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        file_name=file_name,
        dataset_id=dataset_id,
        sheet_names=", ".join(sheet_names) if isinstance(sheet_names, list) else str(sheet_names),
        total_rows=total_rows,
        total_cols=total_cols,
        top_columns=", ".join(top_columns[:8]) if isinstance(top_columns, list) else str(top_columns),
        sample_rows=sample_rows,
        detected_periods=detected_periods,
        detected_currency=detected_currency,
        detected_fact_types=detected_fact_types
    )


def get_finalizer_prompt(
    results: str,
    query: str,
    method: str = "unknown",
    provenance: str = ""
) -> str:
    """
    Generate prompt for finalizing results into professional response.
    
    Args:
        results: Raw computation results (JSON or text)
        query: Original user query
        method: Computation method used (sql_duckdb, pandas, llm_direct, summary)
        provenance: Source/provenance information (dataset_id, sheet, row/col)
        
    Returns:
        Formatted prompt for finalizer
    """
    template_str = _PROMPT_TEMPLATES.get("finalizer", """
{guardrails}

You are a Senior Chartered Accountant presenting analysis results to management.

RAW COMPUTATION RESULTS:
{results}

ORIGINAL QUERY: {query}

COMPUTATION METHOD: {method}
PROVENANCE: {provenance}

INSTRUCTIONS:
1. Present the answer in complete sentences with professional CA tone
2. Include the numeric result with appropriate precision (2 decimal places)
3. Express amounts in INR with appropriate scale (Lakhs/Crores/Millions)
4. Reference the data source (dataset/sheet/row-column where applicable)
5. If computing growth/variance, show both values and the calculation
6. Add a brief methodology note
7. Include a CA disclaimer if the data is from uploaded documents

OUTPUT FORMAT:
[Main finding in 1-2 complete sentences with the numeric result]

Methodology: [Brief description of how this was computed]

Sources:
- [Dataset ID] | [Sheet/Table] | [Row/Column reference if applicable]

---
*CA Disclaimer: This calculation uses data from uploaded documents only. Verify with source documents before use as audit evidence.*
""")
    
    template = PromptTemplate(
        input_variables=["guardrails", "results", "query", "method", "provenance"],
        template=template_str
    )
    return template.format(
        guardrails=CA_SYSTEM_GUARDRAILS,
        results=results,
        query=query,
        method=method.upper(),
        provenance=provenance or "Direct computation from uploaded data"
    )


def get_out_of_domain_response() -> str:
    """Get the out-of-domain response template."""
    return _PROMPT_TEMPLATES.get("out_of_domain", """
I am an AI Chartered Accountant assistant focused on document-based accounting and finance queries.

I can help you with:
- Analyzing uploaded financial data (Excel, CSV files)
- Financial calculations (variances, growth rates, aggregations)
- Looking up information in uploaded documents
- Tax rate and compliance queries

For general conversation or non-finance topics, please use a general-purpose assistant.
""")


# ============================================================================
# CONVENIENCE FUNCTIONS FOR TOOL INTEGRATION
# ============================================================================

def format_tool_response_prompt(
    tool_name: str,
    tool_output: Dict[str, Any],
    original_query: str
) -> str:
    """
    Format a tool's output into a professional response using LLM.
    
    Args:
        tool_name: Name of the tool that was used
        tool_output: Raw output from the tool
        original_query: User's original question
        
    Returns:
        Formatted prompt for response generation
    """
    import json
    results_str = json.dumps(tool_output, indent=2, default=str)
    
    if tool_name in ("benford_test", "run_benford_test"):
        return get_fraud_analysis_prompt(
            results_str,
            tool_output.get("column", "unknown"),
            tool_output.get("dataset", "unknown")
        )
    elif tool_name == "three_way_match":
        return get_reconciliation_prompt(results_str)
    elif tool_name in ("web_search", "search_tax_rate", "search_regulation", "search_benchmark"):
        return get_web_search_prompt(results_str, original_query)
    else:
        return get_ca_synthesis_prompt(results_str, original_query, tool_name)


# ============================================================================
# RESPONSE FINALIZER - Human-Like Natural Language Formatting
# Production-grade with comprehensive edge case handling
# ============================================================================

# Keywords for heuristic zero-result handling (avoid LLM calls)
_ZERO_RESULT_PATTERNS = {
    "existence": ["is there", "any mention", "any reference", "utterance", "does it have"],
    "count": ["how many", "count of", "number of", "total count", "count"],
    "search": ["find", "search", "look for", "looking for", "where is"],
    "comparison": ["compare", "difference", "vs", "versus"],
}

# Technical terms to sanitize from raw results
_TECHNICAL_TERMS = [
    "col_", "column_", "nan", "NaN", "null", "None", "undefined",
    "dataset_id", "df_", "row_", "index_", "<NA>", "NaT"
]

# LLM response cleanup patterns
_LLM_CLEANUP_PREFIXES = [
    "Response:", "Answer:", "Result:", "Here's", "Based on", 
    "Natural response:", "Output:", "The response is:", "Here is",
    "According to the data,", "Based on the analysis,", "The data shows that"
]


def _sanitize_result_for_display(raw_result: Any) -> str:
    """
    Sanitize raw result for human-readable display.
    
    Handles:
    - None/NaN/null values
    - Technical column names
    - DataFrame/Series string representations
    - Overly long results
    """
    if raw_result is None:
        return ""
    
    result_str = str(raw_result)
    
    # Handle pandas-style null representations
    null_patterns = ["nan", "NaN", "None", "null", "<NA>", "NaT", "undefined"]
    if result_str.strip().lower() in [p.lower() for p in null_patterns]:
        return ""
    
    # Truncate extremely long results (DataFrame string repr)
    if len(result_str) > 2000:
        result_str = result_str[:2000] + "... [truncated]"
    
    # Clean up technical terms for cleaner LLM input
    for term in _TECHNICAL_TERMS:
        if term in result_str and not result_str.replace(term, "").strip():
            return ""  # Result is just a technical term
    
    return result_str


def _detect_query_type(query: str) -> str:
    """
    Detect query type for optimized heuristic handling.
    
    Returns: 'existence', 'count', 'search', 'comparison', 'value', or 'general'
    """
    query_lower = query.lower()
    
    for query_type, patterns in _ZERO_RESULT_PATTERNS.items():
        if any(p in query_lower for p in patterns):
            return query_type
    
    # Detect value queries
    if any(kw in query_lower for kw in ["what is", "what's", "show me", "give me", "total", "sum", "average"]):
        return "value"
    
    # Detect percentage/growth queries
    if any(kw in query_lower for kw in ["growth", "increase", "decrease", "%", "percent", "margin", "ratio"]):
        return "percentage"
    
    # Detect list queries
    if any(kw in query_lower for kw in ["list", "top", "bottom", "all", "names", "items"]):
        return "list"
    
    return "general"


def _is_empty_result(result_str: str, raw_result: Any) -> bool:
    """
    Determine if a result should be treated as empty/zero.
    
    Handles various representations of empty/null/zero values.
    """
    if not result_str:
        return True
    
    # Common empty representations
    empty_patterns = [
        "0", "0.0", "0.00", "-0", "-0.0",
        "", "[]", "{}", "()", 
        "None", "null", "nan", "NaN", "NaT", "<NA>",
        "undefined", "N/A", "n/a", "-"
    ]
    
    if result_str.strip().lower() in [p.lower() for p in empty_patterns]:
        return True
    
    # Check if it's a list/dict that's empty
    if isinstance(raw_result, (list, tuple)) and len(raw_result) == 0:
        return True
    if isinstance(raw_result, dict) and len(raw_result) == 0:
        return True
    
    # Check numeric zero
    try:
        if float(result_str.replace(",", "").replace("₹", "")) == 0:
            return True
    except (ValueError, TypeError):
        pass
    
    return False


def _generate_heuristic_response(query: str, query_type: str, is_negative: bool = False) -> str:
    """
    Generate heuristic response for empty/zero results without LLM call.
    
    Production-grade: Handles multiple query types with appropriate responses.
    """
    query_lower = query.lower()
    
    if query_type == "existence":
        # Extract what they're looking for
        for pattern in ["mention of", "reference to", "any", "utterance of"]:
            if pattern in query_lower:
                idx = query_lower.find(pattern) + len(pattern)
                subject = query_lower[idx:].strip().rstrip("?").strip()
                if subject:
                    return f"No, there are no mentions of {subject} in this data."
        return "No, the requested information was not found in this data."
    
    elif query_type == "count":
        return "There are no matching items. The count is zero."
    
    elif query_type == "search":
        return "No matching records were found for your search criteria."
    
    elif query_type == "comparison":
        return "Unable to perform comparison - no matching data found."
    
    elif query_type == "value":
        return "The requested value is not available in the current dataset."
    
    elif query_type == "percentage":
        if is_negative:
            return "The value shows zero change or no applicable data was found."
        return "No percentage data was found matching your query."
    
    elif query_type == "list":
        return "No items found matching your criteria."
    
    else:
        return "No matching data was found for your query."


def _format_number_indian(value: float, is_percentage: bool = False) -> str:
    """
    Format number in Indian notation (lakhs/crores) with proper handling.
    """
    if is_percentage:
        if value >= 0:
            return f"{value:.2f}%"
        else:
            return f"negative {abs(value):.2f}%"
    
    abs_val = abs(value)
    sign = "" if value >= 0 else "negative "
    
    if abs_val >= 1e9:  # 100 crores+
        return f"{sign}₹{abs_val/1e7:.2f} crores"
    elif abs_val >= 1e7:  # 1 crore+
        return f"{sign}₹{abs_val/1e7:.2f} crores"
    elif abs_val >= 1e5:  # 1 lakh+
        return f"{sign}₹{abs_val/1e5:.2f} lakhs"
    elif abs_val >= 1000:
        return f"{sign}₹{abs_val:,.2f}"
    elif abs_val > 0:
        return f"{sign}₹{abs_val:.2f}"
    else:
        return "₹0"


def _clean_llm_response(response: str) -> str:
    """
    Clean up common LLM response artifacts and prefixes.
    """
    if not response:
        return ""
    
    cleaned = response.strip()
    
    # Remove common prefixes
    for prefix in _LLM_CLEANUP_PREFIXES:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
            if cleaned.startswith(":"):
                cleaned = cleaned[1:].strip()
    
    # Remove markdown formatting if present
    if cleaned.startswith("**") and cleaned.endswith("**"):
        cleaned = cleaned[2:-2]
    
    # Remove quotes if entire response is quoted
    if (cleaned.startswith('"') and cleaned.endswith('"')) or \
       (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1]
    
    # Ensure proper sentence ending
    if cleaned and not cleaned.endswith(('.', '!', '?')):
        cleaned += "."
    
    return cleaned


def get_response_finalizer_prompt(
    query: str,
    raw_result: Any,
    explanation: str = ""
) -> str:
    """
    Get the prompt for formatting a raw result into human-like natural language.
    
    Uses the 'response_finalizer' template from templates.yaml with fallback.
    
    Args:
        query: User's original question
        raw_result: Raw data/value from analysis
        explanation: Optional explanation of the result
        
    Returns:
        Formatted prompt for LLM to generate natural response
    """
    result_str = _sanitize_result_for_display(raw_result)
    
    if "response_finalizer" in _PROMPT_TEMPLATES:
        template = _PROMPT_TEMPLATES["response_finalizer"]
        return template.format(
            query=query,
            raw_result=result_str or "No data",
            explanation=explanation or "No additional context"
        )
    
    # Fallback if template not loaded
    return f"""ROLE: Senior Chartered Accountant's AI assistant.

INPUT:
- Query: {query}
- Raw Result: {result_str or "No data"}  
- Context: {explanation or "None"}

RULES:
1. Lead with direct answer (1-2 sentences max)
2. No preambles or meta-commentary
3. Indian number format: crores/lakhs for large amounts
4. Never mention technical terms (col_0, SQL, pandas)
5. For zero/empty: explain meaning, don't just say "0"

OUTPUT: Generate ONLY the natural response."""


def format_natural_response(
    query: str,
    raw_result: Any,
    explanation: str,
    llm_wrapper
) -> str:
    """
    Format a raw analysis result into human-like natural language.
    
    Production-grade function with multi-tier fallback:
    1. Heuristic handling for common patterns (fast, no LLM)
    2. LLM-based formatting using template
    3. Numeric formatting fallback
    4. Raw result as last resort
    
    Args:
        query: User's original question
        raw_result: Raw data/value from analysis
        explanation: Explanation from the analysis method
        llm_wrapper: LLM wrapper instance for generating response
        
    Returns:
        Human-like natural language response string
    """
    # Sanitize and analyze input
    result_str = _sanitize_result_for_display(raw_result)
    query_type = _detect_query_type(query)
    is_empty = _is_empty_result(result_str, raw_result)
    
    # TIER 1: Heuristic handling for empty/zero results
    if is_empty:
        return _generate_heuristic_response(query, query_type)
    
    # TIER 2: Try LLM-based natural formatting
    try:
        prompt = get_response_finalizer_prompt(query, raw_result, explanation)
        natural_response = llm_wrapper.invoke(prompt)
        
        if natural_response:
            cleaned = _clean_llm_response(natural_response)
            if cleaned and len(cleaned) > 5:  # Sanity check
                return cleaned
                
    except Exception as e:
        logger.debug(f"LLM natural response formatting failed: {e}")
    
    # TIER 3: Numeric formatting fallback
    try:
        # Clean the result for numeric parsing
        clean_num = str(raw_result).replace(",", "").replace("₹", "").replace("%", "").strip()
        value = float(clean_num)
        
        is_percentage = query_type == "percentage" or "%" in str(raw_result)
        is_negative = value < 0
        
        formatted = _format_number_indian(value, is_percentage)
        
        # Build contextual response based on query type
        if query_type == "percentage":
            if is_negative:
                return f"There was a decline of {abs(value):.2f}%."
            return f"The rate is {formatted}"
        elif is_negative:
            return f"The value shows a loss of {formatted.replace('negative ', '')}."
        else:
            return f"The value is {formatted}."
            
    except (ValueError, TypeError):
        pass
    
    # TIER 4: Last resort - return cleaned result
    if result_str:
        # Try to make it somewhat readable
        if len(result_str) > 200:
            return f"The analysis returned: {result_str[:200]}..."
        return f"The result is: {result_str}"
    
    return "Unable to determine a result from the available data."


# Alias for backward compatibility
get_finalizer_prompt = get_response_finalizer_prompt




__all__ = [
    "CA_SYSTEM_GUARDRAILS",
    "TRACK_DATA",
    "TRACK_DOC", 
    "TRACK_WEB",
    "get_router_prompt",
    "get_data_analyst_sql_prompt",
    "get_data_analyst_python_prompt",
    "get_ca_synthesis_prompt",
    "get_document_rag_prompt",
    "get_web_search_prompt",
    "get_dataset_match_prompt",
    "get_metadata_query_prompt",
    "get_tool_description",
    "get_tool_selection_prompt",
    "get_fraud_analysis_prompt",
    "get_reconciliation_prompt",
    "get_output_schema",
    "get_dataset_summary_prompt",
    "get_finalizer_prompt",
    "get_response_finalizer_prompt",
    "format_natural_response",
    "get_out_of_domain_response",
    "format_tool_response_prompt",
    "TOOL_DESCRIPTIONS",
    "OUTPUT_SCHEMAS",
    "TokenManager",
    "get_token_manager",
]

