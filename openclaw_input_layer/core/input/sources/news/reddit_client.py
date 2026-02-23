"""
Reddit 数据客户端
加密货币社区情绪分析
使用 PRAW (Python Reddit API Wrapper)
免费版 60 req/min
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re
from collections import Counter

from ...input_types import (
    SocialSentimentData, NewsData, DataSourceType
)

logger = logging.getLogger(__name__)


@dataclass
class RedditConfig:
    """Reddit配置"""
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    user_agent: str = "OpenClaw-Trading-System/1.0"
    username: Optional[str] = None
    password: Optional[str] = None
    rate_limit: float = 1.0  # 60 req/min


class RedditClient:
    """Reddit数据客户端"""
    
    # 加密货币相关subreddit
    CRYPTO_SUBREDDITS = [
        'BitcoinMarkets',
        'CryptoCurrency',
        'CryptoMarkets',
        'Solana',
        'ethereum',
        'btc',
        'defi',
        'NFTs',
        'altcoin',
        'CryptoTechnology'
    ]
    
    # 情绪关键词
    BULLISH_KEYWORDS = [
        'moon', 'pump', 'bull', 'bullish', 'hodl', 'buy', 'long',
        'rocket', 'lambo', 'gain', 'profit', 'breakout', 'ATH'
    ]
    
    BEARISH_KEYWORDS = [
        'dump', 'bear', 'bearish', 'sell', 'short', 'crash', 'dip',
        'loss', 'panic', 'fear', 'bear market', 'correction'
    ]
    
    def __init__(self, config: Optional[RedditConfig] = None):
        self.config = config or RedditConfig()
        self.reddit = None
        self._request_count = 0
        self._error_count = 0
        self._cache: Dict[str, Any] = {}
        
    async def start(self):
        """启动客户端"""
        try:
            import praw
            
            if not all([self.config.client_id, self.config.client_secret]):
                logger.warning("Reddit API凭证未配置，客户端将以受限模式运行")
                return
            
            self.reddit = praw.Reddit(
                client_id=self.config.client_id,
                client_secret=self.config.client_secret,
                user_agent=self.config.user_agent,
                username=self.config.username,
                password=self.config.password
            )
            
            logger.info("Reddit客户端已启动")
        
        except ImportError:
            logger.error("praw库未安装，请运行: pip install praw")
        except Exception as e:
            logger.error(f"Reddit客户端启动失败: {e}")
    
    async def stop(self):
        """停止客户端"""
        if self.reddit:
            # PRAW没有明确的关闭方法
            pass
        logger.info(f"Reddit客户端已停止 (请求: {self._request_count}, 错误: {self._error_count})")
    
    def _run_sync(self, func, *args, **kwargs):
        """同步运行异步函数"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return func(*args, **kwargs)
    
    async def get_subreddit_posts(
        self,
        subreddit_name: str,
        sort: str = 'hot',
        time_filter: str = 'day',
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """获取subreddit帖子"""
        if not self.reddit:
            logger.warning("Reddit客户端未初始化")
            return []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            if sort == 'hot':
                posts = subreddit.hot(limit=limit)
            elif sort == 'new':
                posts = subreddit.new(limit=limit)
            elif sort == 'top':
                posts = subreddit.top(time_filter=time_filter, limit=limit)
            elif sort == 'rising':
                posts = subreddit.rising(limit=limit)
            else:
                posts = subreddit.hot(limit=limit)
            
            result = []
            for post in posts:
                result.append({
                    'id': post.id,
                    'title': post.title,
                    'content': post.selftext,
                    'author': str(post.author),
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'url': post.url,
                    'permalink': post.permalink,
                    'is_self': post.is_self
                })
            
            self._request_count += 1
            return result
        
        except Exception as e:
            logger.error(f"获取subreddit帖子失败: {e}")
            self._error_count += 1
            return []
    
    async def search_posts(
        self,
        query: str,
        subreddit: Optional[str] = None,
        sort: str = 'relevance',
        time_filter: str = 'week',
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """搜索帖子"""
        if not self.reddit:
            logger.warning("Reddit客户端未初始化")
            return []
        
        try:
            if subreddit:
                subreddit_obj = self.reddit.subreddit(subreddit)
                posts = subreddit_obj.search(query, sort=sort, time_filter=time_filter, limit=limit)
            else:
                posts = self.reddit.subreddit('all').search(query, sort=sort, time_filter=time_filter, limit=limit)
            
            result = []
            for post in posts:
                result.append({
                    'id': post.id,
                    'title': post.title,
                    'content': post.selftext,
                    'author': str(post.author),
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'url': post.url,
                    'permalink': post.permalink,
                    'subreddit': str(post.subreddit)
                })
            
            self._request_count += 1
            return result
        
        except Exception as e:
            logger.error(f"搜索帖子失败: {e}")
            self._error_count += 1
            return []
    
    async def analyze_sentiment(
        self,
        subreddits: Optional[List[str]] = None,
        limit: int = 100
    ) -> SocialSentimentData:
        """分析subreddit情绪"""
        if not subreddits:
            subreddits = ['BitcoinMarkets', 'CryptoCurrency']
        
        all_posts = []
        for subreddit in subreddits:
            posts = await self.get_subreddit_posts(subreddit, sort='hot', limit=limit // len(subreddits))
            all_posts.extend(posts)
        
        if not all_posts:
            return SocialSentimentData(
                platform='reddit',
                timestamp=datetime.now(),
                mention_count=0,
                sentiment_score=0.0,
                trending_keywords=[],
                source=DataSourceType.REDDIT
            )
        
        # 分析情绪
        sentiment_scores = []
        all_texts = []
        
        for post in all_posts:
            text = post['title'] + ' ' + post['content']
            all_texts.append(text)
            
            # 简单情绪分析
            score = self._calculate_sentiment(text)
            sentiment_scores.append(score)
        
        # 提取热门关键词
        trending_keywords = self._extract_trending_keywords(all_texts)
        
        # 计算平均情绪
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        return SocialSentimentData(
            platform='reddit',
            timestamp=datetime.now(),
            mention_count=len(all_posts),
            sentiment_score=avg_sentiment,
            trending_keywords=trending_keywords,
            source=DataSourceType.REDDIT
        )
    
    def _calculate_sentiment(self, text: str) -> float:
        """计算文本情绪分数"""
        text_lower = text.lower()
        
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        # 返回 -1 到 1 之间的分数
        return (bullish_count - bearish_count) / total
    
    def _extract_trending_keywords(self, texts: List[str]) -> List[str]:
        """提取热门关键词"""
        all_text = ' '.join(texts).lower()
        
        # 提取单词
        words = re.findall(r'\b[a-z]{3,15}\b', all_text)
        
        # 过滤常见词
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
            'did', 'she', 'use', 'her', 'way', 'many', 'oil', 'sit', 'set', 'run',
            'eat', 'far', 'sea', 'eye', 'ago', 'off', 'too', 'any', 'say', 'man',
            'try', 'ask', 'end', 'why', 'let', 'put', 'say', 'she', 'try', 'way'
        }
        
        # 统计词频
        word_counts = Counter(w for w in words if w not in stop_words and len(w) > 3)
        
        # 返回最常见的词
        return [word for word, count in word_counts.most_common(10)]
    
    async def get_crypto_sentiment(
        self,
        symbol: str,
        limit: int = 50
    ) -> Dict[str, Any]:
        """获取特定加密货币的社区情绪"""
        # 搜索相关帖子
        posts = await self.search_posts(
            query=symbol,
            subreddit='CryptoCurrency',
            limit=limit
        )
        
        if not posts:
            return {
                'symbol': symbol,
                'mention_count': 0,
                'sentiment_score': 0,
                'sentiment_label': 'neutral'
            }
        
        # 分析情绪
        sentiment_scores = []
        for post in posts:
            text = post['title'] + ' ' + post['content']
            score = self._calculate_sentiment(text)
            sentiment_scores.append(score)
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # 标签
        if avg_sentiment > 0.2:
            label = 'bullish'
        elif avg_sentiment < -0.2:
            label = 'bearish'
        else:
            label = 'neutral'
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'mention_count': len(posts),
            'sentiment_score': avg_sentiment,
            'sentiment_label': label,
            'top_posts': posts[:5]
        }
    
    async def get_trending_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取热门话题"""
        all_posts = []
        
        for subreddit in ['CryptoCurrency', 'BitcoinMarkets']:
            posts = await self.get_subreddit_posts(subreddit, sort='hot', limit=50)
            all_posts.extend(posts)
        
        if not all_posts:
            return []
        
        # 提取关键词
        all_texts = [p['title'] for p in all_posts]
        keywords = self._extract_trending_keywords(all_texts)
        
        # 为每个关键词计算热度
        topics = []
        for keyword in keywords[:limit]:
            mention_count = sum(1 for text in all_texts if keyword in text.lower())
            avg_score = sum(p['score'] for p in all_posts if keyword in p['title'].lower()) / max(mention_count, 1)
            
            topics.append({
                'keyword': keyword,
                'mention_count': mention_count,
                'avg_score': avg_score,
                'hotness': mention_count * avg_score
            })
        
        # 按热度排序
        topics.sort(key=lambda x: x['hotness'], reverse=True)
        
        return topics
    
    async def get_posts_as_news(
        self,
        subreddits: Optional[List[str]] = None,
        limit: int = 25
    ) -> List[NewsData]:
        """将帖子转换为新闻格式"""
        if not subreddits:
            subreddits = ['CryptoCurrency']
        
        all_posts = []
        for subreddit in subreddits:
            posts = await self.get_subreddit_posts(subreddit, sort='hot', limit=limit // len(subreddits))
            all_posts.extend(posts)
        
        news_list = []
        for post in all_posts:
            sentiment = self._calculate_sentiment(post['title'] + ' ' + post['content'])
            keywords = self._extract_trending_keywords([post['title']])
            
            news = NewsData(
                title=post['title'],
                content=post['content'][:500] if post['content'] else post['title'],
                timestamp=datetime.fromtimestamp(post['created_utc']),
                source=f"r/{post.get('subreddit', 'unknown')}",
                url=f"https://reddit.com{post['permalink']}",
                sentiment_score=sentiment,
                keywords=keywords,
                related_symbols=[],
                source_type=DataSourceType.REDDIT
            )
            news_list.append(news)
        
        return news_list
    
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
        logger.info("Reddit缓存已清空")
