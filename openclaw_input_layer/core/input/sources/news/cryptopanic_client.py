"""
CryptoPanic 数据客户端
加密货币新闻聚合平台
免费注册获取 API key
"""

import asyncio
import logging
import aiohttp
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ...input_types import (
    NewsData, DataSourceType
)

logger = logging.getLogger(__name__)


@dataclass
class CryptoPanicConfig:
    """CryptoPanic配置"""
    api_key: Optional[str] = None
    base_url: str = "https://cryptopanic.com/api/v1"
    rate_limit: float = 5.0  # 免费版限制
    timeout: int = 30


class CryptoPanicClient:
    """CryptoPanic新闻客户端"""
    
    # 事件类型映射
    KIND_MAP = {
        'news': '新闻',
        'media': '媒体',
        'all': '全部'
    }
    
    # 情绪关键词
    BULLISH_KEYWORDS = [
        'bull', 'bullish', 'pump', 'moon', 'rally', 'surge', 'soar', 'rocket',
        'breakout', ' ATH', 'all-time high', 'adoption', 'partnership',
        'upgrade', 'mainnet', 'launch', 'listing'
    ]
    
    BEARISH_KEYWORDS = [
        'bear', 'bearish', 'dump', 'crash', 'plunge', 'tank', 'collapse',
        'hack', 'exploit', 'scam', 'fraud', 'SEC', 'regulation', 'ban',
        'lawsuit', 'investigation', 'delist'
    ]
    
    def __init__(self, config: Optional[CryptoPanicConfig] = None):
        self.config = config or CryptoPanicConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self._last_request_time: Optional[datetime] = None
        self._cache: Dict[str, Any] = {}
        self._request_count = 0
        self._error_count = 0
        
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def start(self):
        """启动客户端"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        logger.info("CryptoPanic客户端已启动")
    
    async def stop(self):
        """停止客户端"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info(f"CryptoPanic客户端已停止 (请求: {self._request_count}, 错误: {self._error_count})")
    
    async def _rate_limit(self):
        """速率限制"""
        if self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self.config.rate_limit:
                wait_time = self.config.rate_limit - elapsed
                await asyncio.sleep(wait_time)
        self._last_request_time = datetime.now()
    
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = True,
        cache_ttl: int = 300
    ) -> Optional[Any]:
        """发送API请求"""
        cache_key = f"{endpoint}:{str(params)}"
        
        # 检查缓存
        if use_cache and cache_key in self._cache:
            cached_item = self._cache[cache_key]
            if (datetime.now() - cached_item['timestamp']).seconds < cache_ttl:
                return cached_item['data']
        
        await self._rate_limit()
        
        url = f"{self.config.base_url}{endpoint}"
        
        # 添加认证
        if params is None:
            params = {}
        if self.config.api_key:
            params['auth_token'] = self.config.api_key
        
        try:
            async with self.session.get(url, params=params, timeout=self.config.timeout) as response:
                self._request_count += 1
                
                if response.status == 200:
                    data = await response.json()
                    
                    # 更新缓存
                    if use_cache:
                        self._cache[cache_key] = {
                            'data': data,
                            'timestamp': datetime.now()
                        }
                    
                    return data
                
                elif response.status == 429:
                    logger.warning("CryptoPanic API速率限制，等待后重试...")
                    await asyncio.sleep(30)
                    return None
                
                elif response.status == 401:
                    logger.error("CryptoPanic API认证失败，请检查API key")
                    self._error_count += 1
                    return None
                
                else:
                    error_text = await response.text()
                    logger.error(f"API请求失败: {response.status}, {error_text}")
                    self._error_count += 1
                    return None
        
        except Exception as e:
            logger.error(f"请求错误: {e}")
            self._error_count += 1
            return None
    
    async def get_posts(
        self,
        currencies: Optional[List[str]] = None,
        kind: str = 'news',
        region: str = 'en',
        filter: Optional[str] = None,
        limit: int = 50
    ) -> List[NewsData]:
        """
        获取新闻帖子
        
        Args:
            currencies: 加密货币代码列表，如 ['BTC', 'ETH']
            kind: 类型 (news, media, all)
            region: 地区 (en, de, es, fr, it, pt, ru)
            filter: 过滤 (rising, hot, bullish, bearish, important, saved, lol)
            limit: 返回数量 (最大100)
        """
        params = {
            'kind': kind,
            'region': region,
            'limit': min(limit, 100)
        }
        
        if currencies:
            params['currencies'] = ','.join(currencies)
        if filter:
            params['filter'] = filter
        
        data = await self._make_request('/posts/', params, use_cache=True, cache_ttl=300)
        return self._parse_posts(data or {})
    
    def _parse_posts(self, data: Dict) -> List[NewsData]:
        """解析新闻帖子"""
        posts = []
        results = data.get('results', [])
        
        for post in results:
            try:
                # 提取相关加密货币
                currencies = [
                    c.get('code', '')
                    for c in post.get('currencies', [])
                ]
                
                # 构建完整内容
                title = post.get('title', '')
                content = post.get('description', '') or title
                
                # 简单情绪分析
                sentiment_score = self._analyze_sentiment(title + ' ' + content)
                
                # 提取关键词
                keywords = self._extract_keywords(title + ' ' + content)
                
                news = NewsData(
                    title=title,
                    content=content,
                    timestamp=datetime.fromisoformat(
                        post.get('published_at', '').replace('Z', '+00:00')
                    ),
                    source=post.get('source', {}).get('title', 'CryptoPanic'),
                    url=post.get('url', ''),
                    sentiment_score=sentiment_score,
                    keywords=keywords,
                    related_symbols=currencies,
                    source_type=DataSourceType.CRYPTOPANIC
                )
                
                posts.append(news)
            except Exception as e:
                logger.error(f"解析新闻失败: {e}")
        
        return posts
    
    def _analyze_sentiment(self, text: str) -> float:
        """简单情绪分析"""
        text_lower = text.lower()
        
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw.lower() in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw.lower() in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        # 返回 -1 到 1 之间的分数
        return (bullish_count - bearish_count) / total
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单实现：提取大写的单词（通常是加密货币代码）
        words = re.findall(r'\b[A-Z]{2,5}\b', text)
        return list(set(words))[:10]  # 去重并限制数量
    
    async def get_trending_posts(
        self,
        currencies: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[NewsData]:
        """获取热门新闻"""
        return await self.get_posts(
            currencies=currencies,
            filter='hot',
            limit=limit
        )
    
    async def get_bullish_news(
        self,
        currencies: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[NewsData]:
        """获取看涨新闻"""
        return await self.get_posts(
            currencies=currencies,
            filter='bullish',
            limit=limit
        )
    
    async def get_bearish_news(
        self,
        currencies: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[NewsData]:
        """获取看跌新闻"""
        return await self.get_posts(
            currencies=currencies,
            filter='bearish',
            limit=limit
        )
    
    async def get_important_news(
        self,
        currencies: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[NewsData]:
        """获取重要新闻"""
        return await self.get_posts(
            currencies=currencies,
            filter='important',
            limit=limit
        )
    
    async def search_news(
        self,
        query: str,
        limit: int = 20
    ) -> List[NewsData]:
        """搜索新闻"""
        # CryptoPanic免费版可能不支持搜索，这里用过滤实现
        all_news = await self.get_posts(limit=100)
        query_lower = query.lower()
        
        filtered = [
            news for news in all_news
            if query_lower in news.title.lower() or query_lower in news.content.lower()
        ]
        
        return filtered[:limit]
    
    async def get_news_summary(
        self,
        currencies: Optional[List[str]] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """获取新闻摘要"""
        # 获取不同类型的新闻
        all_news = await self.get_posts(currencies=currencies, limit=100)
        bullish = [n for n in all_news if n.sentiment_score and n.sentiment_score > 0.2]
        bearish = [n for n in all_news if n.sentiment_score and n.sentiment_score < -0.2]
        
        # 统计
        total_count = len(all_news)
        bullish_count = len(bullish)
        bearish_count = len(bearish)
        neutral_count = total_count - bullish_count - bearish_count
        
        # 平均情绪分数
        avg_sentiment = sum(n.sentiment_score or 0 for n in all_news) / max(total_count, 1)
        
        # 热门关键词
        all_keywords = []
        for news in all_news:
            all_keywords.extend(news.keywords)
        
        keyword_counts = {}
        for kw in all_keywords:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        
        top_keywords = sorted(
            keyword_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'timestamp': datetime.now(),
            'period_hours': hours,
            'total_news': total_count,
            'sentiment_distribution': {
                'bullish': bullish_count,
                'neutral': neutral_count,
                'bearish': bearish_count
            },
            'average_sentiment': avg_sentiment,
            'sentiment_label': 'bullish' if avg_sentiment > 0.1 else 'bearish' if avg_sentiment < -0.1 else 'neutral',
            'top_keywords': top_keywords,
            'latest_bullish': bullish[:5] if bullish else [],
            'latest_bearish': bearish[:5] if bearish else []
        }
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取客户端统计信息"""
        return {
            'total_requests': self._request_count,
            'error_count': self._error_count,
            'error_rate': self._error_count / max(self._request_count, 1),
            'cache_size': len(self._cache)
        }
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("CryptoPanic缓存已清空")
