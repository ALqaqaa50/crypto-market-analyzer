"""
Claude Analyzer - AI-powered market analysis using Claude API
"""
import anthropic
import json
from datetime import datetime
from typing import Dict, Any, Optional

from ..utils.logger import get_logger
from ..config.loader import get_config

logger = get_logger(__name__)


class ClaudeAnalyzer:
    """
    AI-powered market analyzer using Claude API.
    
    Features:
    - Comprehensive market analysis
    - Opportunity evaluation
    - Risk assessment
    - Trade recommendations
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude Analyzer.
        
        Args:
            api_key: Anthropic API key (if None, reads from config)
        """
        if api_key is None:
            config = get_config()
            api_key = config.get("claude", "api_key")
            
        if not api_key:
            raise ValueError(
                "Claude API key not provided. Set CLAUDE_API_KEY environment variable "
                "or provide api_key parameter"
            )
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"  # Latest Claude model
        
        logger.info(f"Claude Analyzer initialized with model: {self.model}")
    
    def build_market_context(self, data: Dict[str, Any]) -> str:
        """
        Build comprehensive market context from available data.
        
        Args:
            data: Market data dictionary
            
        Returns:
            Formatted context string for Claude
        """
        context = f"""
أنت محلل سوق محترف متخصص في تداول البيتكوين. مهمتك تحليل الوضع الحالي للسوق وتقديم رؤية واضحة ومفيدة للمتداول.

📊 **الوضع الحالي للسوق:**

**السعر والحركة:**
- السعر الحالي: ${data['price']:,.2f}
- التغير خلال ٢٤ ساعة: {data['change_24h']:+.2f}%
- أعلى سعر خلال ٢٤ ساعة: ${data['high_24h']:,.2f}
- أدنى سعر خلال ٢٤ ساعة: ${data['low_24h']:,.2f}

**حجم التداول:**
- حجم التداول خلال ٢٤ ساعة: ${data['volume_24h']:,.0f}
- متوسط حجم التداول خلال ٧ أيام: ${data['avg_volume_7d']:,.0f}
- النسبة: {(data['volume_24h'] / data['avg_volume_7d'] * 100):,.1f}% من المتوسط

**المؤشرات التقنية:**
- مؤشر القوة النسبية RSI(14): {data['rsi']:.1f}
- MACD: {data['macd']:.2f}
- حجم التداول التراكمي CVD: {data['cvd']:+,.0f}

**معدل التمويل والعقود:**
- معدل التمويل الحالي: {data['funding_rate']:.4f}%
- الاهتمام المفتوح: ${data['open_interest']:,.0f}

**نشاط الحيتان:**
"""
        
        # Add whale transfers if available
        if data.get('whale_transfers'):
            context += "\n**تحويلات كبيرة رصدت خلال الساعات الأخيرة:**\n"
            for transfer in data['whale_transfers']:
                context += (
                    f"- {transfer['amount']:,.0f} BTC بقيمة ${transfer['usd_value']:,.0f} "
                    f"من {transfer['from']} إلى {transfer['to']} قبل {transfer['time_ago']}\n"
                )
        else:
            context += "\n- لم يتم رصد تحويلات كبيرة خلال الساعات الأخيرة.\n"
        
        # Add liquidation clusters if available
        if data.get('liquidation_clusters'):
            context += "\n**تجمعات التصفيات:**\n"
            context += (
                f"- تجمع كبير للبائعين على المكشوف عند "
                f"${data['liquidation_clusters']['shorts_above']:,.0f}\n"
            )
            context += (
                f"- تجمع كبير للمشترين عند "
                f"${data['liquidation_clusters']['longs_below']:,.0f}\n"
            )
        
        # Add orderbook information if available
        if data.get('orderbook'):
            context += "\n**دفتر الأوامر:**\n"
            context += (
                f"- أكبر جدار شراء: {data['orderbook']['biggest_bid_size']:.2f} BTC "
                f"عند ${data['orderbook']['biggest_bid_price']:,.0f}\n"
            )
            context += (
                f"- أكبر جدار بيع: {data['orderbook']['biggest_ask_size']:.2f} BTC "
                f"عند ${data['orderbook']['biggest_ask_price']:,.0f}\n"
            )
        
        context += """

**المطلوب منك:**
١. قيم الوضع الحالي للسوق بشكل شامل
٢. حدد الاتجاه المحتمل (صاعد، هابط، جانبي)
٣. اذكر العوامل الإيجابية والسلبية
٤. حدد نسبة ثقتك في التحليل (من ١ إلى ١٠٠)
٥. إذا كانت هناك فرصة تداول، حدد:
   - نقاط الدخول المقترحة
   - إيقاف الخسارة المنطقي
   - أهداف الربح المحتملة
٦. إذا كان الموقف غامضاً أو خطيراً، قل ذلك بوضوح

**مهم:** كن صريحاً ومباشراً. إذا كان الموقف غير واضح، قل "يفضل الانتظار" بدلاً من إعطاء توصية ضعيفة.
"""
        
        return context
    
    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current market conditions using Claude.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Analysis result dictionary
        """
        try:
            # Build context
            context = self.build_market_context(market_data)
            
            logger.info("Requesting market analysis from Claude...")
            
            # Send request to Claude
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": context
                }]
            )
            
            # Extract analysis text
            analysis = message.content[0].text
            
            # Build result
            result = {
                "timestamp": datetime.now().isoformat(),
                "market_price": market_data['price'],
                "analysis": analysis,
                "model_used": self.model,
                "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
                "error": False
            }
            
            logger.info(
                f"Market analysis completed. Tokens used: {result['tokens_used']}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get analysis from Claude: {e}")
            return {
                "error": True,
                "message": f"فشل في الحصول على تحليل من كلود: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def analyze_opportunity(self, opportunity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a specific trading opportunity detected by the system.
        
        Args:
            opportunity_data: Dictionary containing opportunity details
            
        Returns:
            Opportunity analysis result
        """
        try:
            prompt = f"""
أنت محلل سوق خبير. النظام رصد فرصة تداول محتملة، وأنت مطلوب منك تقييمها بدقة.

**تفاصيل الفرصة:**
- النوع: {opportunity_data['type']}
- السعر الحالي: ${opportunity_data['current_price']:,.2f}
- الوصف: {opportunity_data['description']}

**السياق:**
{self.build_market_context(opportunity_data['market_context'])}

**المطلوب:**
١. هل هذه فرصة حقيقية أم إشارة خاطئة؟
٢. ما هي نسبة نجاح هذه الفرصة في رأيك؟ (من ١ إلى ١٠٠)
٣. ما هي المخاطر الرئيسية؟
٤. إذا كانت الفرصة جيدة:
   - أفضل نقطة دخول
   - إيقاف خسارة محكم
   - هدف ربح واقعي
   - حجم صفقة مقترح (نسبة من رأس المال)
٥. ما هو السيناريو البديل إذا فشلت الفرصة؟

كن صريحاً جداً في تقييمك.
"""
            
            logger.info(f"Analyzing opportunity: {opportunity_data.get('type')}")
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = {
                "opportunity_id": opportunity_data.get('id'),
                "analysis": message.content[0].text,
                "timestamp": datetime.now().isoformat(),
                "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
                "error": False
            }
            
            logger.info(f"Opportunity analysis completed. Tokens: {result['tokens_used']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze opportunity: {e}")
            return {
                "error": True,
                "message": f"فشل تحليل الفرصة: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model"""
        return {
            "model": self.model,
            "provider": "Anthropic",
            "description": "Claude Sonnet 4 - Advanced AI model for market analysis"
        }


# Example usage
if __name__ == "__main__":
    import os
    
    # Initialize analyzer (API key should come from environment in production)
    api_key = os.getenv('CLAUDE_API_KEY', 'your-api-key-here')
    analyzer = ClaudeAnalyzer(api_key=api_key)
    
    # Example market data
    market_data = {
        "price": 95234.50,
        "change_24h": -1.23,
        "high_24h": 96800,
        "low_24h": 94100,
        "volume_24h": 28500000000,
        "avg_volume_7d": 24000000000,
        "rsi": 42.3,
        "macd": -145.6,
        "cvd": -12500,
        "funding_rate": 0.0085,
        "open_interest": 15600000000,
        "whale_transfers": [
            {
                "amount": 3200,
                "usd_value": 304750000,
                "from": "Coinbase",
                "to": "Unknown Wallet",
                "time_ago": "ساعتين"
            }
        ],
        "liquidation_clusters": {
            "shorts_above": 97000,
            "longs_below": 93000
        },
        "orderbook": {
            "biggest_bid_size": 145.6,
            "biggest_bid_price": 95100,
            "biggest_ask_size": 203.4,
            "biggest_ask_price": 95350
        }
    }
    
    # Get analysis
    result = analyzer.analyze_market(market_data)
    
    if not result.get('error'):
        print("=" * 80)
        print("MARKET ANALYSIS")
        print("=" * 80)
        print(result['analysis'])
        print("=" * 80)
        print(f"Tokens used: {result['tokens_used']}")
    else:
        print(f"Error: {result['message']}")
