"""
Dune Analytics 数据客户端
通过社区共享查询获取链上数据
"""

import asyncio
import logging
import aiohttp
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ...input_types import (
    OnChainData, ExchangeFlowData, HolderBehaviorData,
    DataSourceType
)

logger = logging.getLogger(__name__)


@dataclass
class DuneConfig:
    """Dune配置"""
    api_key: Optional[str] = None  # 社区版可能不需要
    base_url: str = "https://core-api.dune.com/public"
    graphql_url: str = "https://core-api.dune.com/public/graphql"
    rate_limit: float = 5.0  # 查询间隔
    timeout: int = 60
    max_retries: int = 3


class DuneClient:
    """Dune Analytics数据客户端"""
    
    # 社区共享查询ID（这些ID是示例，实际使用时需要验证）
    DEFAULT_QUERIES = {
        # 交易所净流入（BTC）
        'btc_exchange_flows': {
            'query_id': '3940735',
            'blockchain': 'bitcoin',
            'refresh': '1h'
        },
        # 长期持有者行为
        'holder_behavior': {
            'query_id': '2453981',
            'refresh': '4h'
        },
        # 稳定币交易所储备
        'stablecoin_reserves': {
            'query_id': '2893922',
            'refresh': '30m'
        },
        # 鲸鱼钱包追踪
        'whale_wallets': {
            'query_id': '3156789',
            'refresh': '1h'
        },
        # 活跃地址数
        'active_addresses': {
            'query_id': '2847561',
            'refresh': '1d'
        }
    }
    
    def __init__(self, config: Optional[DuneConfig] = None):
        self.config = config or DuneConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self._last_request_time: Optional[datetime] = None
        self._cache: Dict[str, Any] = {}
        self._query_cache: Dict[str, Dict] = {}
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
        logger.info("Dune Analytics客户端已启动")
    
    async def stop(self):
        """停止客户端"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info(f"Dune Analytics客户端已停止 (请求: {self._request_count}, 错误: {self._error_count})")
    
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
            'Content-Type': 'application/json',
            'User-Agent': 'OpenClaw-Trading-System/1.0'
        }
        if self.config.api_key:
            headers['x-dune-api-key'] = self.config.api_key
        return headers
    
    async def execute_query(
        self,
        query_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        cache_ttl: int = 3600
    ) -> Optional[Dict[str, Any]]:
        """
        执行Dune查询
        
        Args:
            query_id: 查询ID
            parameters: 查询参数
            use_cache: 是否使用缓存
            cache_ttl: 缓存时间（秒）
        """
        cache_key = f"query_{query_id}:{str(parameters)}"
        
        # 检查缓存
        if use_cache and cache_key in self._cache:
            cached_item = self._cache[cache_key]
            if (datetime.now() - cached_item['timestamp']).seconds < cache_ttl:
                logger.debug(f"使用缓存数据: query {query_id}")
                return cached_item['data']
        
        await self._rate_limit()
        
        url = f"{self.config.base_url}/queries/{query_id}/execute"
        headers = self._get_headers()
        
        payload = {}
        if parameters:
            payload['query_parameters'] = parameters
        
        try:
            # 提交查询
            async with self.session.post(url, headers=headers, json=payload) as response:
                self._request_count += 1
                
                if response.status == 200:
                    execution_data = await response.json()
                    execution_id = execution_data.get('execution_id')
                    
                    if execution_id:
                        # 等待查询完成并获取结果
                        result = await self._get_execution_result(execution_id)
                        
                        # 更新缓存
                        if result and use_cache:
                            self._cache[cache_key] = {
                                'data': result,
                                'timestamp': datetime.now()
                            }
                        
                        return result
                
                elif response.status == 429:
                    logger.warning("Dune API速率限制，等待后重试...")
                    await asyncio.sleep(30)
                    return None
                
                else:
                    error_text = await response.text()
                    logger.error(f"Dune查询失败: {response.status}, {error_text}")
                    self._error_count += 1
                    return None
        
        except Exception as e:
            logger.error(f"Dune查询错误: {e}")
            self._error_count += 1
            return None
    
    async def _get_execution_result(
        self,
        execution_id: str,
        max_wait: int = 300,
        poll_interval: int = 5
    ) -> Optional[Dict[str, Any]]:
        """获取查询执行结果"""
        url = f"{self.config.base_url}/execution/{execution_id}/results"
        headers = self._get_headers()
        
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < max_wait:
            try:
                async with self.session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        state = data.get('state', '')
                        
                        if state == 'QUERY_STATE_COMPLETED':
                            return data
                        elif state in ['QUERY_STATE_FAILED', 'QUERY_STATE_CANCELLED']:
                            logger.error(f"查询执行失败: {state}")
                            return None
                        
                        # 继续等待
                        logger.debug(f"查询执行中: {state}, 等待 {poll_interval}秒...")
                        await asyncio.sleep(poll_interval)
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"获取结果失败: {response.status}, {error_text}")
                        return None
            
            except Exception as e:
                logger.error(f"获取结果错误: {e}")
                return None
        
        logger.warning(f"查询超时: {execution_id}")
        return None
    
    async def get_exchange_flows(
        self,
        blockchain: str = 'bitcoin',
        days: int = 30
    ) -> List[ExchangeFlowData]:
        """获取交易所资金流向"""
        query_info = self.DEFAULT_QUERIES.get('btc_exchange_flows', {})
        query_id = query_info.get('query_id')
        
        if not query_id:
            logger.warning("未配置交易所流向查询ID")
            return []
        
        parameters = {
            'blockchain': blockchain,
            'days': days
        }
        
        result = await self.execute_query(query_id, parameters)
        return self._parse_exchange_flows(result, blockchain)
    
    def _parse_exchange_flows(
        self,
        result: Optional[Dict],
        blockchain: str
    ) -> List[ExchangeFlowData]:
        """解析交易所流向数据"""
        flows = []
        
        if not result or 'result' not in result:
            return flows
        
        rows = result['result'].get('rows', [])
        
        for row in rows:
            try:
                flow = ExchangeFlowData(
                    symbol=row.get('symbol', 'BTC'),
                    timestamp=datetime.fromisoformat(row.get('timestamp', '').replace('Z', '+00:00')),
                    exchange=row.get('exchange', 'unknown'),
                    inflow=Decimal(str(row.get('inflow', 0))),
                    outflow=Decimal(str(row.get('outflow', 0))),
                    netflow=Decimal(str(row.get('netflow', 0))),
                    source=DataSourceType.DUNE
                )
                flows.append(flow)
            except Exception as e:
                logger.error(f"解析流向数据失败: {e}")
        
        return flows
    
    async def get_holder_behavior(
        self,
        symbol: str = 'BTC',
        days: int = 90
    ) -> Optional[HolderBehaviorData]:
        """获取持有者行为数据"""
        query_info = self.DEFAULT_QUERIES.get('holder_behavior', {})
        query_id = query_info.get('query_id')
        
        if not query_id:
            logger.warning("未配置持有者行为查询ID")
            return None
        
        parameters = {
            'symbol': symbol,
            'days': days
        }
        
        result = await self.execute_query(query_id, parameters)
        return self._parse_holder_behavior(result, symbol)
    
    def _parse_holder_behavior(
        self,
        result: Optional[Dict],
        symbol: str
    ) -> Optional[HolderBehaviorData]:
        """解析持有者行为数据"""
        if not result or 'result' not in result:
            return None
        
        rows = result['result'].get('rows', [])
        if not rows:
            return None
        
        latest = rows[-1]
        
        try:
            return HolderBehaviorData(
                symbol=symbol,
                timestamp=datetime.fromisoformat(latest.get('timestamp', '').replace('Z', '+00:00')),
                long_term_holders=Decimal(str(latest.get('long_term_holders', 0))),
                short_term_holders=Decimal(str(latest.get('short_term_holders', 0))),
                new_addresses=int(latest.get('new_addresses', 0)),
                active_addresses=int(latest.get('active_addresses', 0)),
                source=DataSourceType.DUNE
            )
        except Exception as e:
            logger.error(f"解析持有者行为数据失败: {e}")
            return None
    
    async def get_custom_query(
        self,
        query_id: str,
        parameters: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """执行自定义查询"""
        return await self.execute_query(query_id, parameters)
    
    async def get_query_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """获取查询执行状态"""
        url = f"{self.config.base_url}/execution/{execution_id}/status"
        headers = self._get_headers()
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"获取查询状态失败: {response.status}, {error_text}")
                    return None
        except Exception as e:
            logger.error(f"获取查询状态错误: {e}")
            return None
    
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
        logger.info("Dune缓存已清空")
