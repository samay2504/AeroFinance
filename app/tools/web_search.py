"""
Web Search Tool - Lightweight fact-checking via DuckDuckGo HTML scraping.
Provides external data for tax rates, regulations, benchmarks.
"""
import logging
import re
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


@tool("web_search", return_direct=True)
def web_search(
    query: str,
    num_results: int = 3,
    site_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search the web for fact-checking using DuckDuckGo.
    
    Args:
        query: Search query string
        num_results: Number of results to return (max 5)
        site_filter: Optional site to filter results (e.g., 'incometax.gov.in')
        
    Returns:
        Dict with results (title, snippet, url) and explanation
    """
    if not HTTPX_AVAILABLE or not BS4_AVAILABLE:
        return {
            "result": "error",
            "explain": "Required packages not available: httpx, beautifulsoup4"
        }

    try:
        # Build search query
        search_query = query
        if site_filter:
            search_query = f"site:{site_filter} {query}"

        results = _search_duckduckgo(search_query, min(num_results, 5))
        
        if not results:
            return {
                "result": "no_results",
                "query": query,
                "explain": "No results found for this query"
            }

        return {
            "result": "success",
            "query": query,
            "results": results,
            "num_results": len(results),
            "explain": f"Found {len(results)} results for '{query}'"
        }

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {
            "result": "error",
            "query": query,
            "explain": f"Search failed: {str(e)}"
        }


def _search_duckduckgo(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """Search DuckDuckGo and parse HTML results."""
    
    encoded_query = quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            
            results = []
            
            # DuckDuckGo HTML results structure
            result_divs = soup.find_all("div", class_="result")
            
            for div in result_divs[:num_results]:
                try:
                    # Extract title and URL
                    title_elem = div.find("a", class_="result__a")
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    href = title_elem.get("href", "")
                    
                    # Extract actual URL from DuckDuckGo redirect
                    if "uddg=" in href:
                        # Parse the uddg parameter
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                        url = parsed.get("uddg", [href])[0]
                    else:
                        url = href
                    
                    # Extract snippet
                    snippet_elem = div.find("a", class_="result__snippet")
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if title and url:
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet[:300]  # Limit snippet length
                        })
                        
                except Exception as e:
                    logger.debug(f"Failed to parse result: {e}")
                    continue
            
            return results
            
    except httpx.HTTPError as e:
        logger.error(f"HTTP error during search: {e}")
        return []
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []


def search_tax_rate(
    tax_type: str = "corporate",
    country: str = "India",
    year: int = 2024
) -> Dict[str, Any]:
    """
    Convenience function to search for tax rates.
    
    Args:
        tax_type: Type of tax ('corporate', 'income', 'gst', 'tds')
        country: Country (default India)
        year: Tax year
        
    Returns:
        Search results for the tax query
    """
    query = f"{tax_type} tax rate {country} {year}"
    
    # Add relevant site filters for India
    if country.lower() == "india":
        site_filters = ["incometax.gov.in", "cbic.gov.in", "taxguru.in"]
    else:
        site_filters = []
    
    results = []
    for site in site_filters[:1]:  # Try primary source first
        result = web_search(query, num_results=2, site_filter=site)
        if result.get("result") == "success":
            results.extend(result.get("results", []))
    
    # General search fallback
    if not results:
        result = web_search(query, num_results=3)
        results = result.get("results", [])
    
    return {
        "result": "success" if results else "no_results",
        "query": query,
        "results": results,
        "disclaimer": "Please verify with official sources for compliance"
    }


__all__ = ["web_search", "search_tax_rate"]
