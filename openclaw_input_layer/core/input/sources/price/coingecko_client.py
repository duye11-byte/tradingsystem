"""
CoinGecko 数据客户端
免费加密货币数据API，适合低频数据获取
"""

import asyncio
import logging
import aiohttp
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ...input_types import (
    PriceData, DataSourceType, DataQualityMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class CoinGeckoConfig:
    """CoinGecko配置"""
    api_key: Optional[str] = None  # 免费版不需要API key
    base_url: str = "https://api.coingecko.com/api/v3"
    pro_url: str = "https://pro-api.coingecko.com/api/v3"
    rate_limit: float = 25.0  # 免费版: 10-30 calls/min，保守设置25秒间隔
    timeout: int = 30
    use_pro: bool = False


class CoinGeckoClient:
    """CoinGecko数据客户端"""
    
    # 常用加密货币ID映射
    SYMBOL_TO_ID = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BNB': 'binancecoin',
        'ADA': 'cardano',
        'DOT': 'polkadot',
        'MATIC': 'matic-network',
        'AVAX': 'avalanche-2',
        'LINK': 'chainlink',
        'UNI': 'uniswap',
        'ATOM': 'cosmos',
        'LTC': 'litecoin',
        'ALGO': 'algorand',
        'VET': 'vechain',
        'FIL': 'filecoin',
        'TRX': 'tron',
        'ETC': 'ethereum-classic',
        'XLM': 'stellar',
        'BCH': 'bitcoin-cash',
        'XRP': 'ripple',
    }
    
    def __init__(self, config: Optional[CoinGeckoConfig] = None):
        self.config = config or CoinGeckoConfig()
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
        logger.info("CoinGecko客户端已启动")
    
    async def stop(self):
        """停止客户端"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info(f"CoinGecko客户端已停止 (请求: {self._request_count}, 错误: {self._error_count})")
    
    async def _rate_limit(self):
        """速率限制"""
        if self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self.config.rate_limit:
                wait_time = self.config.rate_limit - elapsed
                logger.debug(f"速率限制等待: {wait_time:.2f}秒")
                await asyncio.sleep(wait_time)
        self._last_request_time = datetime.now()
    
    def _get_url(self, endpoint: str) -> str:
        """获取完整URL"""
        base = self.config.pro_url if self.config.use_pro else self.config.base_url
        return f"{base}{endpoint}"
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'OpenClaw-Trading-System/1.0'
        }
        if self.config.use_pro and self.config.api_key:
            headers['x-cg-pro-api-key'] = self.config.api_key
        return headers
    
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = True,
        cache_ttl: int = 60
    ) -> Optional[Any]:
        """发送API请求"""
        cache_key = f"{endpoint}:{str(params)}"
        
        # 检查缓存
        if use_cache and cache_key in self._cache:
            cached_item = self._cache[cache_key]
            if (datetime.now() - cached_item['timestamp']).seconds < cache_ttl:
                logger.debug(f"使用缓存数据: {endpoint}")
                return cached_item['data']
        
        await self._rate_limit()
        
        url = self._get_url(endpoint)
        headers = self._get_headers()
        
        try:
            async with self.session.get(url, headers=headers, params=params, timeout=self.config.timeout) as response:
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
                    logger.warning("CoinGecko API速率限制，等待60秒...")
                    await asyncio.sleep(60)
                    return None
                
                else:
                    error_text = await response.text()
                    logger.error(f"API请求失败: {response.status}, {error_text}")
                    self._error_count += 1
                    return None
        
        except asyncio.TimeoutError:
            logger.error(f"请求超时: {endpoint}")
            self._error_count += 1
            return None
        except Exception as e:
            logger.error(f"请求错误: {e}")
            self._error_count += 1
            return None
    
    # ==================== 市场数据 API ====================
    
    async def get_coins_list(self) -> List[Dict[str, Any]]:
        """获取所有支持的加密货币列表"""
        return await self._make_request('/coins/list', use_cache=True, cache_ttl=3600) or []
    
    async def get_coin_data(self, coin_id: str) -> Optional[Dict[str, Any]]:
        """获取单个加密货币详细数据"""
        params = {
            'localization': 'false',
            'tickers': 'false',
            'market_data': 'true',
            'community_data': 'true',
            'developer_data': 'true'
        }
        return await self._make_request(f'/coins/{coin_id}', params, use_cache=True, cache_ttl=300)
    
    async def get_markets(
        self,
        vs_currency: str = 'usd',
        ids: Optional[List[str]] = None,
        category: Optional[str] = None,
        order: str = 'market_cap_desc',
        per_page: int = 100,
        page: int = 1,
        sparkline: bool = False,
        price_change_percentage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取市场数据
        
        Args:
            vs_currency: 计价货币 (usd, eur, jpy, etc.)
            ids: 加密货币ID列表
            category: 类别 (decentralized-finance-defi, non-fungible-tokens-nft, etc.)
            order: 排序方式 (market_cap_desc, market_cap_asc, volume_desc, etc.)
            per_page: 每页数量 (1-250)
            page: 页码
            sparkline: 是否包含7天价格走势
            price_change_percentage: 价格变化时间范围 (1h, 24h, 7d, 14d, 30d, 200d, 1y)
        """
        params = {
            'vs_currency': vs_currency,
            'order': order,
            'per_page': per_page,
            'page': page,
            'sparkline': str(sparkline).lower()
        }
        
        if ids:
            params['ids'] = ','.join(ids)
        if category:
            params['category'] = category
        if price_change_percentage:
            params['price_change_percentage'] = price_change_percentage
        
        return await self._make_request('/coins/markets', params, use_cache=True, cache_ttl=60) or []
    
    async def get_market_chart(
        self,
        coin_id: str,
        vs_currency: str = 'usd',
        days: int = 1,
        interval: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取历史市场数据
        
        Args:
            coin_id: 加密货币ID
            vs_currency: 计价货币
            days: 数据天数 (1, 7, 14, 30, 90, 180, 365, max)
            interval: 数据间隔 (daily, 默认根据days自动选择)
        """
        params = {
            'vs_currency': vs_currency,
            'days': days
        }
        if interval:
            params['interval'] = interval
        
        return await self._make_request(f'/coins/{coin_id}/market_chart', params, use_cache=True, cache_ttl=300) or {}
    
    async def get_market_chart_range(
        self,
        coin_id: str,
        vs_currency: str = 'usd',
        from_timestamp: int = None,
        to_timestamp: int = None
    ) -> Dict[str, Any]:
        """获取指定时间范围的历史数据"""
        params = {
            'vs_currency': vs_currency,
            'from': from_timestamp,
            'to': to_timestamp
        }
        return await self._make_request(f'/coins/{coin_id}/market_chart/range', params, use_cache=True, cache_ttl=300) or {}
    
    # ==================== 全局数据 API ====================
    
    async def get_global_data(self) -> Dict[str, Any]:
        """获取全球加密货币市场数据"""
        return await self._make_request('/global', use_cache=True, cache_ttl=300) or {}
    
    async def get_global_defi_data(self) -> Dict[str, Any]:
        """获取DeFi市场数据"""
        return await self._make_request('/global/decentralized_finance_defi', use_cache=True, cache_ttl=300) or {}
    
    # ==================== 便捷方法 ====================
    
    def symbol_to_id(self, symbol: str) -> Optional[str]:
        """将交易对符号转换为CoinGecko ID"""
        symbol = symbol.upper().replace('USDT', '').replace('USD', '').replace('BUSD', '')
        return self.SYMBOL_TO_ID.get(symbol)
    
    async def get_price_by_symbol(
        self,
        symbol: str,
        vs_currency: str = 'usd'
    ) -> Optional[Decimal]:
        """通过交易对符号获取价格"""
        coin_id = self.symbol_to_id(symbol)
        if not coin_id:
            logger.warning(f"未找到符号对应的CoinGecko ID: {symbol}")
            return None
        
        markets = await self.get_markets(
            vs_currency=vs_currency,
            ids=[coin_id],
            per_page=1
        )
        
        if markets:
            return Decimal(str(markets[0].get('current_price', 0)))
        return None
    
    async def get_multiple_prices(
        self,
        symbols: List[str],
        vs_currency: str = 'usd'
    ) -> Dict[str, Decimal]:
        """获取多个交易对的价格"""
        coin_ids = []
        symbol_map = {}
        
        for symbol in symbols:
            coin_id = self.symbol_to_id(symbol)
            if coin_id:
                coin_ids.append(coin_id)
                symbol_map[coin_id] = symbol
        
        if not coin_ids:
            return {}
        
        markets = await self.get_markets(
            vs_currency=vs_currency,
            ids=coin_ids,
            per_page=len(coin_ids)
        )
        
        prices = {}
        for market in markets:
            coin_id = market.get('id')
            if coin_id in symbol_map:
                symbol = symbol_map[coin_id]
                prices[symbol] = Decimal(str(market.get('current_price', 0)))
        
        return prices
    
    async def get_coin_ohlc(
        self,
        coin_id: str,
        vs_currency: str = 'usd',
        days: int = 1
    ) -> List[List[float]]:
        """
        获取OHLC数据
        
        Returns:
            [[timestamp, open, high, low, close], ...]
        """
        params = {
            'vs_currency': vs_currency,
            'days': days
        }
        data = await self._make_request(f'/coins/{coin_id}/ohlc', params, use_cache=True, cache_ttl=300)
        return data or []
    
    async def get_trending_coins(self) -> List[Dict[str, Any]]:
        """获取热门搜索的加密货币"""
        data = await self._make_request('/search/trending', use_cache=True, cache_ttl=300)
        return data.get('coins', []) if data else []
    
    async def get_exchange_rates(self) -> Dict[str, Any]:
        """获取BTC汇率"""
        return await self._make_request('/exchange_rates', use_cache=True, cache_ttl=300) or {}
    
    # ==================== 数据转换方法 ====================
    
    def market_to_price_data(
        self,
        market: Dict[str, Any],
        symbol: str
    ) -> Optional[PriceData]:
        """将市场数据转换为PriceData"""
        try:
            return PriceData(
                symbol=symbol,
                timestamp=datetime.now(),
                open_price=Decimal(str(market.get('high_24h', 0))),  # 近似值
                high_price=Decimal(str(market.get('high_24h', 0))),
                low_price=Decimal(str(market.get('low_24h', 0))),
                close_price=Decimal(str(market.get('current_price', 0))),
                volume=Decimal(str(market.get('total_volume', 0))),
                quote_volume=Decimal(str(market.get('total_volume', 0))) * Decimal(str(market.get('current_price', 0))),
                trades_count=0,
                source=DataSourceType.COINGECKO,
                metadata={
                    'market_cap': market.get('market_cap'),
                    'market_cap_rank': market.get('market_cap_rank'),
                    'price_change_24h': market.get('price_change_24h'),
                    'price_change_percentage_24h': market.get('price_change_percentage_24h'),
                    'ath': market.get('ath'),
                    'atl': market.get('atl'),
                }
            )
        except Exception as e:
            logger.error(f"转换PriceData失败: {e}")
            return None
    
    # ==================== 统计信息 ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取客户端统计信息"""
        return {
            'total_requests': self._request_count,
            'error_count': self._error_count,
            'error_rate': self._error_count / max(self._request_count, 1),
            'cache_size': len(self._cache),
            'rate_limit': self.config.rate_limit
        }
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("CoinGecko缓存已清空")
