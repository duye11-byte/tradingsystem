"""
Arkham Intelligence 数据客户端
实体标记和聪明钱追踪（免费版有限查询）
"""

import asyncio
import logging
import aiohttp
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ...input_types import (
    OnChainData, DataSourceType
)

logger = logging.getLogger(__name__)


@dataclass
class ArkhamConfig:
    """Arkham配置"""
    api_key: Optional[str] = None
    base_url: str = "https://api.arkhamintelligence.com"
    rate_limit: float = 10.0  # 免费版限制较严格
    timeout: int = 30
    max_retries: int = 3


class ArkhamClient:
    """Arkham Intelligence数据客户端"""
    
    def __init__(self, config: Optional[ArkhamConfig] = None):
        self.config = config or ArkhamConfig()
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
        logger.info("Arkham客户端已启动")
    
    async def stop(self):
        """停止客户端"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info(f"Arkham客户端已停止 (请求: {self._request_count}, 错误: {self._error_count})")
    
    async def _rate_limit(self):
        """速率限制"""
        if self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self.config.rate_limit:
                wait_time = self.config.rate_limit - elapsed
                await asyncio.sleep(wait_time)
        self._last_request_time = datetime.now()
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'OpenClaw-Trading-System/1.0'
        }
        if self.config.api_key:
            headers['API-Key'] = self.config.api_key
        return headers
    
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
                logger.debug(f"使用缓存数据: {endpoint}")
                return cached_item['data']
        
        await self._rate_limit()
        
        url = f"{self.config.base_url}{endpoint}"
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
                    logger.warning("Arkham API速率限制，等待后重试...")
                    await asyncio.sleep(60)
                    return None
                
                elif response.status == 401:
                    logger.error("Arkham API认证失败，请检查API key")
                    self._error_count += 1
                    return None
                
                else:
                    error_text = await response.text()
                    logger.error(f"Arkham API请求失败: {response.status}, {error_text}")
                    self._error_count += 1
                    return None
        
        except Exception as e:
            logger.error(f"Arkham请求错误: {e}")
            self._error_count += 1
            return None
    
    # ==================== 实体 API ====================
    
    async def get_entities(
        self,
        search: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取实体列表"""
        params = {'limit': limit}
        if search:
            params['search'] = search
        if entity_type:
            params['type'] = entity_type
        
        data = await self._make_request('/entity', params, use_cache=True, cache_ttl=3600)
        return data.get('entities', []) if data else []
    
    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """获取单个实体详情"""
        return await self._make_request(f'/entity/{entity_id}', use_cache=True, cache_ttl=3600)
    
    async def get_entity_balances(
        self,
        entity_id: str,
        chain: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取实体持仓"""
        params = {}
        if chain:
            params['chain'] = chain
        
        return await self._make_request(
            f'/entity/{entity_id}/balances',
            params,
            use_cache=True,
            cache_ttl=300
        ) or {}
    
    async def get_entity_transactions(
        self,
        entity_id: str,
        chain: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取实体交易记录"""
        params = {'limit': limit, 'offset': offset}
        if chain:
            params['chain'] = chain
        
        data = await self._make_request(
            f'/entity/{entity_id}/transactions',
            params,
            use_cache=True,
            cache_ttl=60
        )
        return data.get('transactions', []) if data else []
    
    # ==================== 地址 API ====================
    
    async def get_address_info(self, address: str, chain: str) -> Optional[Dict[str, Any]]:
        """获取地址信息"""
        params = {'chain': chain}
        return await self._make_request(
            f'/address/{address}',
            params,
            use_cache=True,
            cache_ttl=3600
        )
    
    async def get_address_transactions(
        self,
        address: str,
        chain: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取地址交易记录"""
        params = {'chain': chain, 'limit': limit}
        data = await self._make_request(
            f'/address/{address}/transactions',
            params,
            use_cache=True,
            cache_ttl=60
        )
        return data.get('transactions', []) if data else []
    
    async def get_address_balances(
        self,
        address: str,
        chain: str
    ) -> Dict[str, Any]:
        """获取地址持仓"""
        params = {'chain': chain}
        return await self._make_request(
            f'/address/{address}/balances',
            params,
            use_cache=True,
            cache_ttl=300
        ) or {}
    
    # ==================== 聪明钱追踪 ====================
    
    async def get_smart_money_flows(
        self,
        token: Optional[str] = None,
        chain: Optional[str] = None,
        timeframe: str = "24h",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        获取聪明钱流向
        
        Args:
            token: 代币地址或符号
            chain: 链名称
            timeframe: 时间范围 (1h, 24h, 7d, 30d)
            limit: 返回数量
        """
        params = {'timeframe': timeframe, 'limit': limit}
        if token:
            params['token'] = token
        if chain:
            params['chain'] = chain
        
        data = await self._make_request(
            '/flows/smart-money',
            params,
            use_cache=True,
            cache_ttl=300
        )
        return data.get('flows', []) if data else []
    
    async def get_whale_transactions(
        self,
        min_value: Decimal = Decimal('100000'),
        chain: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取鲸鱼交易（大额转账）"""
        params = {
            'minValue': str(min_value),
            'limit': limit
        }
        if chain:
            params['chain'] = chain
        
        data = await self._make_request(
            '/transactions/whale',
            params,
            use_cache=True,
            cache_ttl=60
        )
        return data.get('transactions', []) if data else []
    
    # ==================== 投资组合 API ====================
    
    async def get_portfolio_value(
        self,
        addresses: List[str],
        chain: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取投资组合价值"""
        params = {'addresses': ','.join(addresses)}
        if chain:
            params['chain'] = chain
        
        return await self._make_request(
            '/portfolio/value',
            params,
            use_cache=True,
            cache_ttl=300
        ) or {}
    
    async def get_portfolio_history(
        self,
        addresses: List[str],
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """获取投资组合历史"""
        params = {
            'addresses': ','.join(addresses),
            'days': days
        }
        
        data = await self._make_request(
            '/portfolio/history',
            params,
            use_cache=True,
            cache_ttl=3600
        )
        return data.get('history', []) if data else []
    
    # ==================== 警报 API ====================
    
    async def get_alerts(
        self,
        alert_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取链上警报"""
        params = {'limit': limit}
        if alert_type:
            params['type'] = alert_type
        
        data = await self._make_request('/alerts', params, use_cache=True, cache_ttl=60)
        return data.get('alerts', []) if data else []
    
    async def create_alert(self, alert_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """创建自定义警报"""
        # 注意：免费版可能不支持创建警报
        logger.warning("免费版Arkham可能不支持创建自定义警报")
        return None
    
    # ==================== 便捷方法 ====================
    
    async def find_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """通过名称查找实体"""
        entities = await self.get_entities(search=name, limit=10)
        
        for entity in entities:
            if entity.get('name', '').lower() == name.lower():
                return entity
        
        return entities[0] if entities else None
    
    async def get_exchange_holdings(
        self,
        exchange_name: str,
        token: Optional[str] = None
    ) -> Dict[str, Decimal]:
        """获取交易所持仓"""
        entity = await self.find_entity_by_name(exchange_name)
        if not entity:
            return {}
        
        balances = await self.get_entity_balances(entity.get('id'))
        holdings = {}
        
        for chain, tokens in balances.get('balances', {}).items():
            for token_data in tokens:
                token_symbol = token_data.get('symbol', 'UNKNOWN')
                if token and token_symbol.upper() != token.upper():
                    continue
                
                balance = Decimal(str(token_data.get('balance', 0)))
                price = Decimal(str(token_data.get('price', 0)))
                
                if token_symbol in holdings:
                    holdings[token_symbol] += balance * price
                else:
                    holdings[token_symbol] = balance * price
        
        return holdings
    
    async def track_smart_money_buying(
        self,
        token: str,
        min_buy_amount: Decimal = Decimal('50000'),
        timeframe: str = "24h"
    ) -> List[Dict[str, Any]]:
        """追踪聪明钱买入信号"""
        flows = await self.get_smart_money_flows(
            token=token,
            timeframe=timeframe,
            limit=100
        )
        
        buying_signals = []
        for flow in flows:
            if flow.get('type') == 'buy':
                amount = Decimal(str(flow.get('amount', 0)))
                if amount >= min_buy_amount:
                    buying_signals.append(flow)
        
        return buying_signals
    
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
        logger.info("Arkham缓存已清空")
