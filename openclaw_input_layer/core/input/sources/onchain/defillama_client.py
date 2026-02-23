"""
DeFiLlama 数据客户端
获取TVL、收益率、稳定币等DeFi数据
完全免费，无API key限制
"""

import asyncio
import logging
import aiohttp
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ...input_types import (
    TVLData, OnChainData, DataSourceType
)

logger = logging.getLogger(__name__)


@dataclass
class DeFiLlamaConfig:
    """DeFiLlama配置"""
    base_url: str = "https://api.llama.fi"
    yields_url: str = "https://yields.llama.fi"
    stablecoins_url: str = "https://stablecoins.llama.fi"
    rate_limit: float = 0.5  # 2 req/sec 保守设置
    timeout: int = 30


class DeFiLlamaClient:
    """DeFiLlama数据客户端"""
    
    def __init__(self, config: Optional[DeFiLlamaConfig] = None):
        self.config = config or DeFiLlamaConfig()
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
        logger.info("DeFiLlama客户端已启动")
    
    async def stop(self):
        """停止客户端"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info(f"DeFiLlama客户端已停止 (请求: {self._request_count}, 错误: {self._error_count})")
    
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
        url: str,
        use_cache: bool = True,
        cache_ttl: int = 300
    ) -> Optional[Any]:
        """发送HTTP请求"""
        cache_key = url
        
        # 检查缓存
        if use_cache and cache_key in self._cache:
            cached_item = self._cache[cache_key]
            if (datetime.now() - cached_item['timestamp']).seconds < cache_ttl:
                return cached_item['data']
        
        await self._rate_limit()
        
        try:
            async with self.session.get(url, timeout=self.config.timeout) as response:
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
                else:
                    error_text = await response.text()
                    logger.error(f"请求失败: {response.status}, {error_text}")
                    self._error_count += 1
                    return None
        
        except Exception as e:
            logger.error(f"请求错误: {e}")
            self._error_count += 1
            return None
    
    # ==================== TVL API ====================
    
    async def get_protocols(self) -> List[Dict[str, Any]]:
        """获取所有协议列表"""
        url = f"{self.config.base_url}/protocols"
        return await self._make_request(url, use_cache=True, cache_ttl=3600) or []
    
    async def get_protocol_tvl(self, protocol: str) -> Optional[Dict[str, Any]]:
        """获取单个协议的TVL历史"""
        url = f"{self.config.base_url}/protocol/{protocol}"
        return await self._make_request(url, use_cache=True, cache_ttl=300)
    
    async def get_chain_tvl(self, chain: str) -> Optional[Dict[str, Any]]:
        """获取单个链的TVL历史"""
        url = f"{self.config.base_url}/v2/chains"
        data = await self._make_request(url, use_cache=True, cache_ttl=300)
        
        if data:
            for chain_data in data:
                if chain_data.get('name', '').lower() == chain.lower():
                    return chain_data
        return None
    
    async def get_all_chains_tvl(self) -> List[Dict[str, Any]]:
        """获取所有链的TVL"""
        url = f"{self.config.base_url}/v2/chains"
        return await self._make_request(url, use_cache=True, cache_ttl=300) or []
    
    async def get_historical_tvl(
        self,
        protocol: Optional[str] = None,
        chain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取历史TVL数据"""
        if protocol:
            url = f"{self.config.base_url}/protocol/{protocol}"
            data = await self._make_request(url, use_cache=True, cache_ttl=300)
            return data.get('tvl', []) if data else []
        elif chain:
            url = f"{self.config.base_url}/v2/historicalChainTvl/{chain}"
            return await self._make_request(url, use_cache=True, cache_ttl=300) or []
        else:
            url = f"{self.config.base_url}/v2/historicalTvl"
            return await self._make_request(url, use_cache=True, cache_ttl=300) or []
    
    def parse_tvl_data(
        self,
        protocol: str,
        data: Dict[str, Any],
        chain: str = ""
    ) -> Optional[TVLData]:
        """解析TVL数据"""
        try:
            tvl_list = data.get('tvl', [])
            if not tvl_list:
                return None
            
            latest = tvl_list[-1]
            previous = tvl_list[-2] if len(tvl_list) > 1 else latest
            
            current_tvl = Decimal(str(latest.get('totalLiquidityUSD', 0)))
            previous_tvl = Decimal(str(previous.get('totalLiquidityUSD', 0)))
            
            # 计算24小时变化（近似）
            tvl_change_24h = Decimal('0')
            if previous_tvl > 0:
                tvl_change_24h = (current_tvl - previous_tvl) / previous_tvl
            
            # 计算7天变化
            tvl_change_7d = Decimal('0')
            if len(tvl_list) >= 8:
                week_ago = tvl_list[-8]
                week_ago_tvl = Decimal(str(week_ago.get('totalLiquidityUSD', 0)))
                if week_ago_tvl > 0:
                    tvl_change_7d = (current_tvl - week_ago_tvl) / week_ago_tvl
            
            return TVLData(
                protocol=protocol,
                timestamp=datetime.fromtimestamp(latest.get('date', 0)),
                tvl_usd=current_tvl,
                tvl_change_24h=tvl_change_24h,
                tvl_change_7d=tvl_change_7d,
                chain=chain,
                source=DataSourceType.DEFILLAMA
            )
        except Exception as e:
            logger.error(f"解析TVL数据失败: {e}")
            return None
    
    # ==================== 收益率 API ====================
    
    async def get_pools(self) -> List[Dict[str, Any]]:
        """获取所有收益池"""
        url = f"{self.config.yields_url}/pools"
        data = await self._make_request(url, use_cache=True, cache_ttl=300)
        return data.get('data', []) if data else []
    
    async def get_pool_chart(self, pool_id: str) -> List[Dict[str, Any]]:
        """获取收益池历史数据"""
        url = f"{self.config.yields_url}/chart/{pool_id}"
        data = await self._make_request(url, use_cache=True, cache_ttl=300)
        return data.get('data', []) if data else []
    
    async def get_top_yield_pools(
        self,
        chain: Optional[str] = None,
        limit: int = 10,
        min_tvl: Decimal = Decimal('1000000')
    ) -> List[Dict[str, Any]]:
        """获取高收益池"""
        pools = await self.get_pools()
        
        filtered = []
        for pool in pools:
            # 过滤条件
            if chain and pool.get('chain') != chain:
                continue
            
            tvl = Decimal(str(pool.get('tvlUsd', 0)))
            if tvl < min_tvl:
                continue
            
            filtered.append(pool)
        
        # 按APY排序
        filtered.sort(key=lambda x: float(x.get('apy', 0)), reverse=True)
        
        return filtered[:limit]
    
    # ==================== 稳定币 API ====================
    
    async def get_stablecoins(self) -> List[Dict[str, Any]]:
        """获取稳定币列表"""
        url = f"{self.config.stablecoins_url}/stablecoins"
        data = await self._make_request(url, use_cache=True, cache_ttl=300)
        return data.get('peggedAssets', []) if data else []
    
    async def get_stablecoin_charts(
        self,
        stablecoin_id: str,
        chain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取稳定币流通量历史"""
        if chain:
            url = f"{self.config.stablecoins_url}/stablecoincharts/{chain}?stablecoin={stablecoin_id}"
        else:
            url = f"{self.config.stablecoins_url}/stablecoincharts/all?stablecoin={stablecoin_id}"
        
        return await self._make_request(url, use_cache=True, cache_ttl=300) or []
    
    async def get_stablecoin_prices(self) -> List[Dict[str, Any]]:
        """获取稳定币价格"""
        url = f"{self.config.stablecoins_url}/stablecoinprices"
        return await self._make_request(url, use_cache=True, cache_ttl=60) or []
    
    async def get_stablecoin_mcap(self, stablecoin_id: str) -> Optional[Decimal]:
        """获取稳定币市值"""
        charts = await self.get_stablecoin_charts(stablecoin_id)
        if charts:
            latest = charts[-1]
            total_circulating = latest.get('totalCirculatingUSD', {})
            # 汇总所有链的流通量
            total = sum(Decimal(str(v)) for v in total_circulating.values())
            return total
        return None
    
    # ==================== 便捷方法 ====================
    
    async def get_total_tvl(self) -> Decimal:
        """获取全链总TVL"""
        url = f"{self.config.base_url}/v2/chains"
        data = await self._make_request(url, use_cache=True, cache_ttl=300)
        
        if data:
            total = sum(
                Decimal(str(chain.get('tvl', 0)))
                for chain in data
            )
            return total
        return Decimal('0')
    
    async def get_chain_dominance(self) -> Dict[str, Decimal]:
        """获取各链TVL占比"""
        chains = await self.get_all_chains_tvl()
        total_tvl = await self.get_total_tvl()
        
        if total_tvl == 0:
            return {}
        
        dominance = {}
        for chain in chains:
            name = chain.get('name', 'Unknown')
            tvl = Decimal(str(chain.get('tvl', 0)))
            dominance[name] = (tvl / total_tvl) * 100
        
        return dominance
    
    async def get_protocol_comparison(
        self,
        protocols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """对比多个协议的TVL"""
        comparison = {}
        
        for protocol in protocols:
            data = await self.get_protocol_tvl(protocol)
            if data:
                tvl_data = self.parse_tvl_data(protocol, data)
                if tvl_data:
                    comparison[protocol] = {
                        'tvl_usd': tvl_data.tvl_usd,
                        'tvl_change_24h': tvl_data.tvl_change_24h,
                        'tvl_change_7d': tvl_data.tvl_change_7d,
                        'chain': tvl_data.chain
                    }
        
        return comparison
    
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
        logger.info("DeFiLlama缓存已清空")
