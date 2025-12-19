"""
Web Search Tool - Production-grade with multi-provider fallback.
Providers: DuckDuckGo (primary) -> Brave Search (fallback) -> Bing (fallback)
Provides external data for tax rates, regulations, benchmarks.
"""
import logging
import re
import os
from typing import Dict, Any, List, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

try:
    from langchain.tools import tool
except ImportError:
    def tool(name: str = None, return_direct: bool = False):
        def decorator(func):
            func.name = name or func.__name__
            return func
        return decorator

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _contains_non_english(text: str) -> bool:
    """
    Check if text contains significant non-English characters.
    Uses Unicode range detection for: CJK, Arabic, Cyrillic, Thai, Hebrew, etc.
    """
    non_english_count = 0
    total_alpha = 0
    
    for char in text:
        code = ord(char)
        if char.isalpha():
            total_alpha += 1
            # CJK Unified Ideographs
            if 0x4E00 <= code <= 0x9FFF:
                non_english_count += 1
            # CJK Extensions
            elif 0x3400 <= code <= 0x4DBF or 0x20000 <= code <= 0x2B81F:
                non_english_count += 1
            # Japanese Hiragana/Katakana
            elif 0x3040 <= code <= 0x30FF:
                non_english_count += 1
            # Korean Hangul
            elif 0xAC00 <= code <= 0xD7AF:
                non_english_count += 1
            # Arabic
            elif 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
                non_english_count += 1
            # Cyrillic
            elif 0x0400 <= code <= 0x04FF or 0x0500 <= code <= 0x052F:
                non_english_count += 1
            # Thai
            elif 0x0E00 <= code <= 0x0E7F:
                non_english_count += 1
            # Hebrew
            elif 0x0590 <= code <= 0x05FF:
                non_english_count += 1
            # Devanagari (Hindi)
            elif 0x0900 <= code <= 0x097F:
                non_english_count += 1
    
    # If more than 10% of alphabetic chars are non-English, filter it
    if total_alpha > 0 and (non_english_count / total_alpha) > 0.10:
        return True
    return False


def _calculate_relevancy_score(text: str, query: str) -> float:
    """
    Calculate relevancy score between search result and query.
    Uses keyword overlap with IDF-like weighting.
    Returns 0.0-1.0 score.
    """
    # Normalize
    text_lower = text.lower()
    query_lower = query.lower()
    
    # Extract query keywords (remove stopwords)
    stopwords = {'what', 'is', 'the', 'for', 'in', 'a', 'an', 'and', 'or', 'of', 'to', 'with', 'how', 'why'}
    query_words = set(query_lower.split()) - stopwords
    
    if not query_words:
        return 0.5  # Neutral score if no keywords
    
    # Count keyword matches
    matches = 0
    for word in query_words:
        if len(word) >= 3 and word in text_lower:  # Only count words 3+ chars
            matches += 1
    
    return matches / len(query_words) if query_words else 0.0


def _filter_relevant_results(
    results: List[Dict[str, str]], 
    query: str, 
    min_score: float = 0.2,
    max_results: int = 3
) -> List[Dict[str, str]]:
    """
    Filter and rank results by relevancy.
    Removes non-English and low-relevancy results.
    """
    scored_results = []
    
    for result in results:
        combined_text = f"{result.get('title', '')} {result.get('snippet', '')}"
        
        # Skip non-English
        if _contains_non_english(combined_text):
            continue
        
        # Calculate relevancy
        score = _calculate_relevancy_score(combined_text, query)
        
        # Only include if meets threshold
        if score >= min_score:
            result['_relevancy_score'] = score
            scored_results.append(result)
    
    # Sort by relevancy and limit
    scored_results.sort(key=lambda x: x.get('_relevancy_score', 0), reverse=True)
    
    # Remove internal score before returning
    for r in scored_results:
        r.pop('_relevancy_score', None)
    
    return scored_results[:max_results]


# ============================================================================
# SEARCH PROVIDERS
# ============================================================================

def _search_duckduckgo(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """Search DuckDuckGo HTML results with English language preference."""
    if not HTTPX_AVAILABLE or not BS4_AVAILABLE:
        return []
    
    encoded_query = quote_plus(query)
    # Add kl=en-in for English (India) region
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}&kl=en-in"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            results = []
            
            # Collect all results - filtering happens centrally in _web_search_impl
            for div in soup.find_all("div", class_="result")[:num_results]:
                try:
                    title_elem = div.find("a", class_="result__a")
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    href = title_elem.get("href", "")
                    
                    # Extract URL from DuckDuckGo redirect
                    if "uddg=" in href:
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                        result_url = parsed.get("uddg", [href])[0]
                    else:
                        result_url = href
                    
                    snippet_elem = div.find("a", class_="result__snippet")
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if title and result_url:
                        results.append({
                            "title": title,
                            "url": result_url,
                            "snippet": snippet[:300],
                            "source": "duckduckgo"
                        })
                except Exception:
                    continue
            
            return results
            
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []


def _search_brave(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """Search using Brave Search API."""
    if not HTTPX_AVAILABLE:
        return []
    
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        logger.debug("Brave Search API key not configured")
        return []
    
    try:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key
        }
        params = {
            "q": query,
            "count": min(num_results, 10),
            "country": "IN",  # Default to India for CA use case
        }
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("web", {}).get("results", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", "")[:300],
                    "source": "brave"
                })
            
            return results
            
    except Exception as e:
        logger.warning(f"Brave Search failed: {e}")
        return []


def _search_bing_scrape(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """Fallback: Scrape Bing search results with English language preference."""
    if not HTTPX_AVAILABLE or not BS4_AVAILABLE:
        return []
    
    try:
        encoded_query = quote_plus(query)
        # Add setlang=en and mkt=en-IN for English results from India
        url = f"https://www.bing.com/search?q={encoded_query}&setlang=en&mkt=en-IN"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            results = []
            
            # Collect all results - filtering happens centrally in _web_search_impl
            for li in soup.select("li.b_algo")[:num_results]:
                try:
                    title_elem = li.find("h2")
                    if not title_elem:
                        continue
                    
                    link = title_elem.find("a")
                    if not link:
                        continue
                    
                    title = link.get_text(strip=True)
                    href = link.get("href", "")
                    
                    snippet_elem = li.find("p")
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if title and href:
                        results.append({
                            "title": title,
                            "url": href,
                            "snippet": snippet[:300],
                            "source": "bing"
                        })
                except Exception:
                    continue
            
            return results
            
    except Exception as e:
        logger.warning(f"Bing scrape failed: {e}")
        return []


# ============================================================================
# MAIN SEARCH FUNCTION
# ============================================================================

def _web_search_impl(
    query: str,
    num_results: int = 3,
    site_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Internal implementation of web search with multi-provider fallback.
    
    Provider Chain:
    1. DuckDuckGo (primary - no API key needed)
    2. Brave Search (fallback - needs BRAVE_SEARCH_API_KEY)
    3. Bing Scrape (emergency fallback)
    
    Uses relevancy filtering to ensure quality results.
    """
    if not HTTPX_AVAILABLE or not BS4_AVAILABLE:
        return {
            "result": "error",
            "explain": "Required packages not available: httpx, beautifulsoup4"
        }

    # Build search query with site filter
    search_query = query
    if site_filter:
        search_query = f"site:{site_filter} {query}"

    raw_results = []
    provider_used = None
    
    # Get more results than needed to allow for filtering
    fetch_count = num_results * 3
    
    # Try DuckDuckGo first (no API key needed)
    raw_results = _search_duckduckgo(search_query, fetch_count)
    if raw_results:
        provider_used = "duckduckgo"
    
    # Fallback to Brave Search
    if not raw_results:
        raw_results = _search_brave(search_query, fetch_count)
        if raw_results:
            provider_used = "brave"
    
    # Emergency fallback to Bing scraping
    if not raw_results:
        raw_results = _search_bing_scrape(search_query, fetch_count)
        if raw_results:
            provider_used = "bing"
    
    if not raw_results:
        return {
            "result": "no_results",
            "query": query,
            "explain": "No results found from any search provider"
        }
    
    # Apply relevancy filtering
    filtered_results = _filter_relevant_results(
        raw_results, 
        query, 
        min_score=0.15,  # Lower threshold to be more inclusive
        max_results=num_results
    )
    
    # If filtering was too strict, fall back to at least some results
    if not filtered_results and raw_results:
        # Just remove non-English without relevancy check
        filtered_results = [
            r for r in raw_results 
            if not _contains_non_english(f"{r.get('title', '')} {r.get('snippet', '')}")
        ][:num_results]
    
    if not filtered_results:
        return {
            "result": "no_relevant_results",
            "query": query,
            "explain": "Search results were filtered as irrelevant or non-English"
        }

    return {
        "result": "success",
        "query": query,
        "results": filtered_results,
        "num_results": len(filtered_results),
        "provider": provider_used,
        "explain": f"Found {len(filtered_results)} relevant results via {provider_used}"
    }


@tool("web_search", return_direct=True)
def web_search(
    query: str,
    num_results: int = 3,
    site_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search the web for fact-checking using multiple providers.
    
    Args:
        query: Search query string
        num_results: Number of results to return (max 5)
        site_filter: Optional site to filter results (e.g., 'incometax.gov.in')
        
    Returns:
        Dict with results (title, snippet, url) and explanation
    """
    try:
        return _web_search_impl(query, min(num_results, 5), site_filter)
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {
            "result": "error",
            "query": query,
            "explain": f"Search failed: {str(e)}"
        }


# ============================================================================
# SPECIALIZED SEARCH FUNCTIONS
# ============================================================================

def search_tax_rate(
    tax_type: str = "corporate",
    country: str = "India",
    year: int = 2024
) -> Dict[str, Any]:
    """
    Search for tax rates with authority sources.
    
    Args:
        tax_type: Type of tax ('corporate', 'income', 'gst', 'tds')
        country: Country (default India)
        year: Tax year
        
    Returns:
        Search results for the tax query
    """
    query = f"{tax_type} tax rate {country} {year}"
    
    # Authority site filters by country
    site_filters = {
        "india": ["incometax.gov.in", "cbic.gov.in", "taxguru.in", "cleartax.in"],
        "usa": ["irs.gov", "tax.gov"],
        "uk": ["gov.uk"],
    }
    
    country_sites = site_filters.get(country.lower(), [])
    
    # Get the underlying search function
    search_func = web_search.func if hasattr(web_search, 'func') else _web_search_impl
    
    results = []
    
    # Try authority sources first
    for site in country_sites[:2]:
        result = search_func(query, num_results=2, site_filter=site)
        if result.get("result") == "success":
            results.extend(result.get("results", []))
    
    # General search fallback
    if len(results) < 2:
        result = search_func(query, num_results=3)
        results.extend(result.get("results", []))
    
    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in results:
        if r.get("url") not in seen_urls:
            seen_urls.add(r.get("url"))
            unique_results.append(r)
    
    return {
        "result": "success" if unique_results else "no_results",
        "query": query,
        "results": unique_results[:5],
        "disclaimer": "Please verify with official government sources for compliance"
    }


def search_regulation(
    regulation_name: str,
    country: str = "India"
) -> Dict[str, Any]:
    """
    Search for regulatory information and compliance requirements.
    
    Args:
        regulation_name: Name of regulation (e.g., 'GST invoice format', 'TDS rate')
        country: Country (default India)
        
    Returns:
        Search results with regulatory information
    """
    query = f"{regulation_name} {country} official regulation 2024"
    
    search_func = web_search.func if hasattr(web_search, 'func') else _web_search_impl
    result = search_func(query, num_results=3)
    
    if result.get("result") == "success":
        result["disclaimer"] = "Always verify regulations with official government sources"
    
    return result


def search_market_benchmark(
    metric: str,
    industry: str = "",
    region: str = "India"
) -> Dict[str, Any]:
    """
    Search for industry benchmarks and market data.
    
    Args:
        metric: Metric to search (e.g., 'gross margin', 'EBITDA margin')
        industry: Industry sector (optional)
        region: Geographic region (default India)
        
    Returns:
        Search results with benchmark information
    """
    query = f"{industry} {metric} benchmark {region} 2024"
    
    search_func = web_search.func if hasattr(web_search, 'func') else _web_search_impl
    result = search_func(query.strip(), num_results=3)
    
    if result.get("result") == "success":
        result["disclaimer"] = "Benchmarks are indicative. Verify with industry reports."
    
    return result


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "web_search",
    "search_tax_rate",
    "search_regulation",
    "search_market_benchmark",
]
