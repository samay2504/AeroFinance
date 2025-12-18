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
# SEARCH PROVIDERS
# ============================================================================

def _search_duckduckgo(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """Search DuckDuckGo HTML results."""
    if not HTTPX_AVAILABLE or not BS4_AVAILABLE:
        return []
    
    encoded_query = quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            results = []
            
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
                        url = parsed.get("uddg", [href])[0]
                    else:
                        url = href
                    
                    snippet_elem = div.find("a", class_="result__snippet")
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if title and url:
                        results.append({
                            "title": title,
                            "url": url,
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
    """Fallback: Scrape Bing search results."""
    if not HTTPX_AVAILABLE or not BS4_AVAILABLE:
        return []
    
    try:
        encoded_query = quote_plus(query)
        url = f"https://www.bing.com/search?q={encoded_query}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            results = []
            
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

    results = []
    provider_used = None
    
    # Try DuckDuckGo first (no API key needed)
    results = _search_duckduckgo(search_query, num_results)
    if results:
        provider_used = "duckduckgo"
    
    # Fallback to Brave Search
    if not results:
        results = _search_brave(search_query, num_results)
        if results:
            provider_used = "brave"
    
    # Emergency fallback to Bing scraping
    if not results:
        results = _search_bing_scrape(search_query, num_results)
        if results:
            provider_used = "bing"
    
    if not results:
        return {
            "result": "no_results",
            "query": query,
            "explain": "No results found from any search provider"
        }

    return {
        "result": "success",
        "query": query,
        "results": results,
        "num_results": len(results),
        "provider": provider_used,
        "explain": f"Found {len(results)} results via {provider_used} for '{query}'"
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
