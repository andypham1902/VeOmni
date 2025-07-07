"""
Simple search and visit tools for VeOmni validation.

Just two functions: web_search and web_visit.
"""

import json
import requests
import time
from typing import Dict, List, Any, Optional


def web_search(query: str, top_k: int = 10, preview_char: int = 256, api_config: Optional[Dict] = None) -> str:
    """
    Search the web for information.
    
    Args:
        query: The search query
        top_k: Number of results to return (default: 10)
        preview_char: Number of preview characters (default: 256)
        api_config: Optional API configuration
        
    Returns:
        Formatted string with search results
    """
    try:
        # Use real search endpoint
        search_url = "http://192.168.0.8:10000/search"
        
        payload = {
            "query": query,
            "top_k": top_k,
            "preview_char": preview_char
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(search_url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Handle response that directly contains results or has success field
            if result.get('success') or 'results' in result:
                results = result.get('results', [])
                formatted_results = []
                for i, item in enumerate(results[:top_k]):
                    title = item.get('title', item.get('metadata', {}).get('paper_title', 'No title'))
                    url = item.get('url', 'No URL')
                    preview = item.get('preview', item.get('content', ''))[:preview_char]
                    
                    formatted_results.append(
                        f"{i+1}. {title}\n"
                        f"   URL: {url}\n"
                        f"   Preview: {preview}..."
                    )
                return f"Search results for '{query}':\n" + "\n".join(formatted_results)
            else:
                return f"Search failed: {result.get('error', 'Unknown error')}"
        else:
            return f"Search API error: HTTP {response.status_code}"
        
    except Exception as e:
        import traceback
        return f"Error executing web search: {str(e)}\nTraceback: {traceback.format_exc()}"


def web_visit(url: str, api_config: Optional[Dict] = None) -> str:
    """
    Visit a URL and retrieve the content in markdown.
    
    Args:
        url: The URL to visit
        api_config: Optional API configuration
        
    Returns:
        Formatted string with page content
    """
    try:
        # Use real visit endpoint
        visit_url = "http://192.168.0.8:10000/visit"
        
        payload = {
            "url": url
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(visit_url, json=payload, headers=headers, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            # Handle different response formats
            if result.get('success') or 'content' in result or 'data' in result:
                # Check if the visit was successful (not 404 etc)
                if result.get('status_code') == 404:
                    return f"Visit failed: Page not found (404) for {url}"
                elif result.get('status_code') and result.get('status_code') != 200:
                    return f"Visit failed: HTTP {result.get('status_code')} for {url}"
                
                # Get content from either 'content' or 'data' field
                content = result.get('content', result.get('data', ''))[:1000]
                return f"Content from {url}:\n{content}..."
            else:
                return f"Visit failed: {result.get('error', 'Unknown error')}"
        else:
            return f"Visit API error: HTTP {response.status_code}"
        
    except Exception as e:
        import traceback
        return f"Error visiting URL: {str(e)}\nTraceback: {traceback.format_exc()}"


# Tool schemas for reference
TOOL_SCHEMAS = {
    'web_search': {
        'name': 'web_search',
        'description': 'Search the web for information',
        'parameters': {
            'query': 'string (required)',
            'top_k': 'integer (optional, default: 3)',
            'preview_char': 'integer (optional, default: 256)'
        }
    },
    'web_visit': {
        'name': 'web_visit',
        'description': 'Visit a URL and retrieve the full document content',
        'parameters': {
            'url': 'string (required)'
        }
    }
}