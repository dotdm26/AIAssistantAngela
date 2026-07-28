import os
from typing import Optional
from langchain_tavily import TavilySearch, TavilyExtract, TavilyCrawl
from langchain.tools import tool

TAVILY_API = os.getenv("TAVILY_API_KEY")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", 5))

@tool
def search_web(query: str, topic: Optional[str] = "general", 
               include_answer: Optional[bool] = False,
               include_raw_content: Optional[bool] = False,
               include_images: Optional[bool] = False,
               include_image_descriptions: Optional[bool] = False,
               search_depth: Optional[str] = "basic",
               time_range: Optional[str] = None,
               include_domains: Optional[list] = None,
               exclude_domains: Optional[list] = None) -> str:
    """Search the web using Tavily and return the results."""
    tavily_tool = TavilySearch(max_results=TAVILY_MAX_RESULTS, include_answer=include_answer, include_raw_content=include_raw_content,
                               include_images=include_images, include_image_descriptions=include_image_descriptions,
                               search_depth=search_depth, time_range=time_range, include_domains=include_domains,
                               exclude_domains=exclude_domains)

    return tavily_tool.invoke({"query": query})

@tool
def extract_url(url: str, extract_depth: Optional[str] = "basic", include_images: Optional[bool] = False) -> str:
    """Extract content from a URL using Tavily."""
    tavily_tool = TavilyExtract(extract_depth=extract_depth, include_images=include_images)
    return tavily_tool.invoke({"url": url})

@tool
def crawl_url(url: str, 
              max_depth: Optional[int] = 1, 
              max_breadth: Optional[int] = 20,
              limit: Optional[int] = 50,
              instructions: Optional[str] = None,
              select_paths: Optional[list] = None,
              select_domains: Optional[list] = None,
              exclude_paths: Optional[list] = None,
              exclude_domains: Optional[list] = None,
              allow_external: Optional[bool] = False,
              extract_depth: Optional[str] = "basic",
              format: Optional[str] = "text",
              include_images: Optional[bool] = False) -> str:
    """Crawl a URL using Tavily."""
    tavily_tool = TavilyCrawl(max_depth=max_depth, max_breadth=max_breadth, limit=limit, instructions=instructions,
                              select_paths=select_paths, select_domains=select_domains, exclude_paths=exclude_paths,
                              exclude_domains=exclude_domains, allow_external=allow_external, extract_depth=extract_depth,
                              format=format, include_images=include_images)
    return tavily_tool.invoke({"url": url})

web_search_tools = [search_web, extract_url, crawl_url]