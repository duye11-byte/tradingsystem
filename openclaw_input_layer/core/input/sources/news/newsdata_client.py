"""
NewsData.io 数据客户端
新闻API，免费版 200 req/day
"""

import asyncio
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ...input_types import (
    NewsData, DataSourceType
)

logger = logging.getLogger(__name__)


@dataclass
class NewsDataConfig:
    """NewsData.io配置"""
    api_key: Optional[str] = None
    base_url: str = "https://newsdata.io/api/1"
    rate_limit: float = 432.0  # 200 req/day = 每432秒一次
    timeout: int = 30


class NewsDataClient:
    """NewsData.io新闻客户端"""
    
    # 加密货币相关关键词
    CRYPTO_KEYWORDS = [
        'bitcoin', 'ethereum', 'crypto', 'cryptocurrency', 'blockchain',
        'defi', 'nft', 'altcoin', 'btc', 'eth', 'trading', 'binance',
        'coinbase', 'wallet', 'mining', 'token'
    ]
    
    def __init__(self, config: Optional[NewsDataConfig] = None):
        self.config = config or NewsDataConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self._last_request_time: Optional[datetime] = None
        self._cache: Dict[str, Any] = {}
        self._request_count = 0
        self._error_count = 0
        self._daily_request_count = 0
        self._last_reset_date = datetime.now().date()
        
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def start(self):
        """启动客户端"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        logger.info("NewsData.io客户端已启动")
    
    async def stop(self):
        """停止客户端"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info(f"NewsData.io客户端已停止 (请求: {self._request_count}, 错误: {self._error_count})")
    
    def _check_daily_limit(self) -> bool:
        """检查每日限制"""
        today = datetime.now().date()
        if today != self._last_reset_date:
            self._daily_request_count = 0
            self._last_reset_date = today
        
        return self._daily_request_count < 200
    
    async def _rate_limit(self):
        """速率限制"""
        if self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self.config.rate_limit:
                wait_time = self.config.rate_limit - elapsed
                logger.debug(f"NewsData.io速率限制等待: {wait_time:.0f}秒")
                await asyncio.sleep(wait_time)
        self._last_request_time = datetime.now()
    
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = True,
        cache_ttl: int = 3600
    ) -> Optional[Any]:
        """发送API请求"""
        # 检查每日限制
        if not self._check_daily_limit():
            logger.warning("NewsData.io每日请求限额已用完")
            return None
        
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
            params['apikey'] = self.config.api_key
        
        try:
            async with self.session.get(url, params=params, timeout=self.config.timeout) as response:
                self._request_count += 1
                self._daily_request_count += 1
                
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
                    logger.warning("NewsData.io API速率限制")
                    return None
                
                elif response.status == 401:
                    logger.error("NewsData.io API认证失败")
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
    
    async def get_news(
        self,
        query: Optional[str] = None,
        q_in_title: Optional[str] = None,
        country: Optional[str] = None,
        category: str = 'business',
        language: str = 'en',
        domain: Optional[str] = None,
        exclude_domain: Optional[str] = None,
        timeframe: Optional[int] = None,  # 小时
        limit: int = 10
    ) -> List[NewsData]:
        """
        获取新闻
        
        Args:
            query: 搜索关键词
            q_in_title: 标题中包含的关键词
            country: 国家代码
            category: 类别 (business, entertainment, health, science, sports, technology, top)
            language: 语言代码
            domain: 指定域名
            exclude_domain: 排除域名
            timeframe: 时间范围（小时）
            limit: 返回数量
        """
        params = {
            'language': language,
            'size': min(limit, 50)  # 最大50
        }
        
        if query:
            params['q'] = query
        if q_in_title:
            params['qInTitle'] = q_in_title
        if country:
            params['country'] = country
        if category:
            params['category'] = category
        if domain:
            params['domain'] = domain
        if exclude_domain:
            params['excludeDomain'] = exclude_domain
        if timeframe:
            params['timeframe'] = timeframe
        
        data = await self._make_request('/news', params, use_cache=True, cache_ttl=1800)
        return self._parse_news(data or {})
    
    def _parse_news(self, data: Dict) -> List[NewsData]:
        """解析新闻数据"""
        news_list = []
        results = data.get('results', [])
        
        for item in results:
            try:
                # 解析发布时间
                pub_date = item.get('pubDate')
                if pub_date:
                    try:
                        timestamp = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    except:
                        timestamp = datetime.now()
                else:
                    timestamp = datetime.now()
                
                # 提取加密货币代码
                related_symbols = self._extract_crypto_symbols(
                    item.get('title', '') + ' ' + item.get('description', '')
                )
                
                # 简单情绪分析
                sentiment_score = self._analyze_sentiment(
                    item.get('title', '') + ' ' + item.get('description', '')
                )
                
                news = NewsData(
                    title=item.get('title', ''),
                    content=item.get('description', '') or item.get('content', ''),
                    timestamp=timestamp,
                    source=item.get('source_id', 'Unknown'),
                    url=item.get('link', ''),
                    sentiment_score=sentiment_score,
                    keywords=item.get('keywords', []),
                    related_symbols=related_symbols,
                    source_type=DataSourceType.NEWSDATA
                )
                
                news_list.append(news)
            except Exception as e:
                logger.error(f"解析新闻失败: {e}")
        
        return news_list
    
    def _extract_crypto_symbols(self, text: str) -> List[str]:
        """从文本中提取加密货币代码"""
        # 常见加密货币代码映射
        crypto_map = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'solana': 'SOL',
            'cardano': 'ADA',
            'polkadot': 'DOT',
            'binance': 'BNB',
            'ripple': 'XRP',
            'litecoin': 'LTC',
            'chainlink': 'LINK',
        }
        
        text_lower = text.lower()
        symbols = []
        
        for name, symbol in crypto_map.items():
            if name in text_lower or symbol.lower() in text_lower:
                symbols.append(symbol)
        
        return symbols
    
    def _analyze_sentiment(self, text: str) -> float:
        """简单情绪分析"""
        text_lower = text.lower()
        
        positive_words = [
            'surge', 'rally', 'gain', 'rise', 'bull', 'boom', 'soar', 'rocket',
            'breakout', 'adoption', 'partnership', 'growth', 'positive', 'optimistic'
        ]
        
        negative_words = [
            'crash', 'plunge', 'drop', 'fall', 'bear', 'dump', 'decline', 'tank',
            'hack', 'scam', 'fraud', 'investigation', 'ban', 'negative', 'concern'
        ]
        
        positive_count = sum(1 for w in positive_words if w in text_lower)
        negative_count = sum(1 for w in negative_words if w in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total
    
    async def get_crypto_news(
        self,
        symbols: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[NewsData]:
        """获取加密货币相关新闻"""
        # 构建查询
        if symbols:
            query = ' OR '.join(symbols)
        else:
            query = ' OR '.join(self.CRYPTO_KEYWORDS[:5])  # 使用前5个关键词
        
        return await self.get_news(
            query=query,
            category='business',
            timeframe=24,
            limit=limit
        )
    
    async def get_latest_headlines(
        self,
        limit: int = 10
    ) -> List[NewsData]:
        """获取最新头条"""
        return await self.get_news(
            category='top',
            limit=limit
        )
    
    async def search_by_symbol(
        self,
        symbol: str,
        limit: int = 10
    ) -> List[NewsData]:
        """按加密货币代码搜索新闻"""
        return await self.get_news(
            query=symbol,
            q_in_title=symbol,
            timeframe=48,
            limit=limit
        )
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取客户端统计信息"""
        return {
            'total_requests': self._request_count,
            'error_count': self._error_count,
            'error_rate': self._error_count / max(self._request_count, 1),
            'daily_requests': self._daily_request_count,
            'daily_limit': 200,
            'daily_remaining': max(0, 200 - self._daily_request_count),
            'cache_size': len(self._cache)
        }
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("NewsData.io缓存已清空")
