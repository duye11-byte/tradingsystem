"""
Binance 数据客户端
支持 WebSocket 实时流和 REST API
"""

import asyncio
import json
import logging
import aiohttp
import websockets
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any, AsyncGenerator
from dataclasses import dataclass

from ...input_types import (
    PriceData, OrderBookData, OrderBookLevel, TradeData,
    DataSourceType, DataFrequency, DataSourceConfig
)

logger = logging.getLogger(__name__)


@dataclass
class BinanceConfig:
    """Binance配置"""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    use_testnet: bool = False
    websocket_reconnect_interval: int = 5
    rest_base_url: str = "https://api.binance.com"
    futures_base_url: str = "https://fapi.binance.com"
    websocket_base_url: str = "wss://stream.binance.com:9443/ws"
    futures_websocket_url: str = "wss://fstream.binance.com/ws"


class BinanceClient:
    """Binance数据客户端"""
    
    def __init__(self, config: Optional[BinanceConfig] = None):
        self.config = config or BinanceConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._callbacks: Dict[str, List[Callable]] = {
            'kline': [],
            'trade': [],
            'orderbook': [],
            'ticker': []
        }
        self._subscribed_streams: set = set()
        self._ws_task: Optional[asyncio.Task] = None
        
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def start(self):
        """启动客户端"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        self._running = True
        logger.info("Binance客户端已启动")
    
    async def stop(self):
        """停止客户端"""
        self._running = False
        
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("Binance客户端已停止")
    
    # ==================== REST API 方法 ====================
    
    async def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[PriceData]:
        """
        获取K线数据
        
        Args:
            symbol: 交易对，如 "BTCUSDT"
            interval: 时间间隔 (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: 返回数量 (默认100，最大1000)
            start_time: 开始时间戳 (毫秒)
            end_time: 结束时间戳 (毫秒)
        """
        url = f"{self.config.rest_base_url}/api/v3/klines"
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._parse_klines(symbol, data)
            else:
                error_text = await response.text()
                logger.error(f"获取K线数据失败: {response.status}, {error_text}")
                return []
    
    def _parse_klines(self, symbol: str, data: List[List]) -> List[PriceData]:
        """解析K线数据"""
        klines = []
        for item in data:
            # Binance Kline格式:
            # [开盘时间, 开盘价, 最高价, 最低价, 收盘价, 成交量, 收盘时间, 成交额, 成交笔数, 
            #  主动买入成交量, 主动买入成交额, 忽略]
            klines.append(PriceData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(item[0] / 1000),
                open_price=Decimal(str(item[1])),
                high_price=Decimal(str(item[2])),
                low_price=Decimal(str(item[3])),
                close_price=Decimal(str(item[4])),
                volume=Decimal(str(item[5])),
                quote_volume=Decimal(str(item[7])),
                trades_count=int(item[8]),
                source=DataSourceType.BINANCE
            ))
        return klines
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[OrderBookData]:
        """获取订单簿"""
        url = f"{self.config.rest_base_url}/api/v3/depth"
        params = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._parse_orderbook(symbol, data)
            else:
                error_text = await response.text()
                logger.error(f"获取订单簿失败: {response.status}, {error_text}")
                return None
    
    def _parse_orderbook(self, symbol: str, data: Dict) -> OrderBookData:
        """解析订单簿数据"""
        bids = [
            OrderBookLevel(
                price=Decimal(str(item[0])),
                quantity=Decimal(str(item[1]))
            )
            for item in data.get('bids', [])
        ]
        asks = [
            OrderBookLevel(
                price=Decimal(str(item[0])),
                quantity=Decimal(str(item[1]))
            )
            for item in data.get('asks', [])
        ]
        
        return OrderBookData(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
            source=DataSourceType.BINANCE
        )
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[TradeData]:
        """获取近期成交"""
        url = f"{self.config.rest_base_url}/api/v3/trades"
        params = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._parse_trades(symbol, data)
            else:
                error_text = await response.text()
                logger.error(f"获取成交数据失败: {response.status}, {error_text}")
                return []
    
    def _parse_trades(self, symbol: str, data: List[Dict]) -> List[TradeData]:
        """解析成交数据"""
        trades = []
        for item in data:
            trades.append(TradeData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(item['time'] / 1000),
                price=Decimal(str(item['price'])),
                quantity=Decimal(str(item['qty'])),
                is_buyer_maker=item['isBuyerMaker'],
                trade_id=str(item['id']),
                source=DataSourceType.BINANCE
            ))
        return trades
    
    async def get_ticker_24h(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """获取24小时价格变动统计"""
        url = f"{self.config.rest_base_url}/api/v3/ticker/24hr"
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                logger.error(f"获取24小时统计失败: {response.status}, {error_text}")
                return {}
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """获取交易对信息"""
        url = f"{self.config.rest_base_url}/api/v3/exchangeInfo"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                logger.error(f"获取交易对信息失败: {response.status}, {error_text}")
                return {}
    
    # ==================== 期货数据 API ====================
    
    async def get_funding_rate(self, symbol: str, limit: int = 1) -> List[Dict[str, Any]]:
        """获取资金费率"""
        url = f"{self.config.futures_base_url}/fapi/v1/fundingRate"
        params = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                logger.error(f"获取资金费率失败: {response.status}, {error_text}")
                return []
    
    async def get_long_short_ratio(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 1
    ) -> List[Dict[str, Any]]:
        """获取多空比"""
        url = f"{self.config.futures_base_url}/futures/data/globalLongShortAccountRatio"
        params = {
            'symbol': symbol.upper(),
            'period': period,
            'limit': limit
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                logger.error(f"获取多空比失败: {response.status}, {error_text}")
                return []
    
    async def get_taker_volume(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 1
    ) -> List[Dict[str, Any]]:
        """获取吃单量统计"""
        url = f"{self.config.futures_base_url}/futures/data/takerlongshortRatio"
        params = {
            'symbol': symbol.upper(),
            'period': period,
            'limit': limit
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                logger.error(f"获取吃单量失败: {response.status}, {error_text}")
                return []
    
    # ==================== WebSocket 方法 ====================
    
    def on(self, event: str, callback: Callable):
        """注册事件回调"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        else:
            logger.warning(f"未知事件类型: {event}")
    
    def off(self, event: str, callback: Callable):
        """移除事件回调"""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
    
    async def subscribe_kline(self, symbols: List[str], interval: str = "1m"):
        """订阅K线数据"""
        streams = [f"{s.lower()}@kline_{interval}" for s in symbols]
        await self._subscribe_streams(streams)
    
    async def subscribe_orderbook(self, symbols: List[str], level: int = 20):
        """订阅订单簿"""
        streams = [f"{s.lower()}@depth{level}@100ms" for s in symbols]
        await self._subscribe_streams(streams)
    
    async def subscribe_trades(self, symbols: List[str]):
        """订阅成交数据"""
        streams = [f"{s.lower()}@aggTrade" for s in symbols]
        await self._subscribe_streams(streams)
    
    async def subscribe_ticker(self, symbols: List[str]):
        """订阅24小时统计"""
        streams = [f"{s.lower()}@ticker" for s in symbols]
        await self._subscribe_streams(streams)
    
    async def _subscribe_streams(self, streams: List[str]):
        """订阅数据流"""
        for stream in streams:
            self._subscribed_streams.add(stream)
        
        # 如果WebSocket已连接，发送订阅消息
        if self.websocket:
            await self._send_subscribe_message(list(self._subscribed_streams))
        else:
            # 启动WebSocket连接
            await self._start_websocket()
    
    async def _start_websocket(self):
        """启动WebSocket连接"""
        if self._ws_task and not self._ws_task.done():
            return
        
        self._ws_task = asyncio.create_task(self._websocket_loop())
    
    async def _websocket_loop(self):
        """WebSocket主循环"""
        while self._running:
            try:
                await self._connect_websocket()
                await self._handle_messages()
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket连接断开，准备重连...")
            except Exception as e:
                logger.error(f"WebSocket错误: {e}")
            
            if self._running:
                await asyncio.sleep(self.config.websocket_reconnect_interval)
    
    async def _connect_websocket(self):
        """连接WebSocket"""
        if not self._subscribed_streams:
            # 等待订阅请求
            await asyncio.sleep(0.1)
            return
        
        streams = list(self._subscribed_streams)
        stream_path = '/'.join(streams)
        ws_url = f"{self.config.websocket_base_url}/{stream_path}"
        
        logger.info(f"连接WebSocket: {ws_url}")
        self.websocket = await websockets.connect(ws_url)
        logger.info("WebSocket连接成功")
    
    async def _send_subscribe_message(self, streams: List[str]):
        """发送订阅消息"""
        if not self.websocket:
            return
        
        message = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(datetime.now().timestamp())
        }
        await self.websocket.send(json.dumps(message))
        logger.info(f"发送订阅消息: {streams}")
    
    async def _handle_messages(self):
        """处理WebSocket消息"""
        async for message in self.websocket:
            if not self._running:
                break
            
            try:
                data = json.loads(message)
                await self._process_message(data)
            except json.JSONDecodeError:
                logger.error(f"JSON解析错误: {message}")
            except Exception as e:
                logger.error(f"处理消息错误: {e}")
    
    async def _process_message(self, data: Dict[str, Any]):
        """处理收到的消息"""
        if 'stream' not in data:
            return
        
        stream = data['stream']
        payload = data.get('data', {})
        
        # 提取symbol
        symbol = stream.split('@')[0].upper()
        
        if '@kline' in stream:
            kline_data = self._parse_ws_kline(symbol, payload)
            await self._emit('kline', kline_data)
        
        elif '@depth' in stream:
            orderbook_data = self._parse_ws_orderbook(symbol, payload)
            await self._emit('orderbook', orderbook_data)
        
        elif '@aggTrade' in stream:
            trade_data = self._parse_ws_trade(symbol, payload)
            await self._emit('trade', trade_data)
        
        elif '@ticker' in stream:
            await self._emit('ticker', payload)
    
    def _parse_ws_kline(self, symbol: str, data: Dict) -> PriceData:
        """解析WebSocket K线数据"""
        k = data.get('k', {})
        return PriceData(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(k.get('t', 0) / 1000),
            open_price=Decimal(str(k.get('o', 0))),
            high_price=Decimal(str(k.get('h', 0))),
            low_price=Decimal(str(k.get('l', 0))),
            close_price=Decimal(str(k.get('c', 0))),
            volume=Decimal(str(k.get('v', 0))),
            quote_volume=Decimal(str(k.get('q', 0))),
            trades_count=k.get('n', 0),
            source=DataSourceType.BINANCE,
            metadata={'is_closed': k.get('x', False)}
        )
    
    def _parse_ws_orderbook(self, symbol: str, data: Dict) -> OrderBookData:
        """解析WebSocket订单簿数据"""
        bids = [
            OrderBookLevel(price=Decimal(str(item[0])), quantity=Decimal(str(item[1])))
            for item in data.get('bids', [])
        ]
        asks = [
            OrderBookLevel(price=Decimal(str(item[0])), quantity=Decimal(str(item[1])))
            for item in data.get('asks', [])
        ]
        
        return OrderBookData(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
            source=DataSourceType.BINANCE
        )
    
    def _parse_ws_trade(self, symbol: str, data: Dict) -> TradeData:
        """解析WebSocket成交数据"""
        return TradeData(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(data.get('T', 0) / 1000),
            price=Decimal(str(data.get('p', 0))),
            quantity=Decimal(str(data.get('q', 0))),
            is_buyer_maker=data.get('m', False),
            trade_id=str(data.get('a', '')),
            source=DataSourceType.BINANCE
        )
    
    async def _emit(self, event: str, data: Any):
        """触发事件回调"""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"回调执行错误: {e}")
    
    # ==================== 流式生成器 ====================
    
    async def stream_klines(
        self,
        symbol: str,
        interval: str = "1m"
    ) -> AsyncGenerator[PriceData, None]:
        """K线数据流生成器"""
        queue: asyncio.Queue = asyncio.Queue()
        
        async def on_kline(data):
            await queue.put(data)
        
        self.on('kline', on_kline)
        await self.subscribe_kline([symbol], interval)
        
        try:
            while self._running:
                data = await queue.get()
                yield data
        finally:
            self.off('kline', on_kline)
    
    async def stream_orderbook(
        self,
        symbol: str,
        level: int = 20
    ) -> AsyncGenerator[OrderBookData, None]:
        """订单簿数据流生成器"""
        queue: asyncio.Queue = asyncio.Queue()
        
        async def on_orderbook(data):
            await queue.put(data)
        
        self.on('orderbook', on_orderbook)
        await self.subscribe_orderbook([symbol], level)
        
        try:
            while self._running:
                data = await queue.get()
                yield data
        finally:
            self.off('orderbook', on_orderbook)
    
    async def stream_trades(
        self,
        symbol: str
    ) -> AsyncGenerator[TradeData, None]:
        """成交数据流生成器"""
        queue: asyncio.Queue = asyncio.Queue()
        
        async def on_trade(data):
            await queue.put(data)
        
        self.on('trade', on_trade)
        await self.subscribe_trades([symbol])
        
        try:
            while self._running:
                data = await queue.get()
                yield data
        finally:
            self.off('trade', on_trade)
