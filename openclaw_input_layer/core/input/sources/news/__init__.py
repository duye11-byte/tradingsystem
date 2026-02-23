"""
新闻事件数据源模块
提供加密货币新闻、社交媒体情绪等
"""

from .cryptopanic_client import CryptoPanicClient
from .newsdata_client import NewsDataClient
from .reddit_client import RedditClient

__all__ = ['CryptoPanicClient', 'NewsDataClient', 'RedditClient']
