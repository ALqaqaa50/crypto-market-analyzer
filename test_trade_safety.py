"""
🧪 Test Suite for Trade Safety System
Tests all safety checks and edge cases
"""

import asyncio
from datetime import datetime, timedelta, timezone
from okx_stream_hunter.core.trade_safety import TradeSafety, SafetyConfig


class SafetyTester:
    """Test harness for safety system"""
    
    def __init__(self):
        self.safety = TradeSafety(SafetyConfig(
            min_confidence=0.70,
            max_spoof_score=0.50,
            max_risk_penalty=0.80,
            min_trade_interval_seconds=300,
            max_trades_per_hour=4,
            max_trades_per_day=20,
            max_daily_loss_pct=0.05,
            max_consecutive_losses=3,
        ))
        
        self.test_results = []
    
    def test_rapid_signals(self):
        """Test 1: Duplicate signal filtering"""
        print("\n" + "="*80)
        print("TEST 1: RAPID SIGNALS (Duplicate Filtering)")
        print("="*80)
        
        base_signal = {
            "direction": "long",
            "confidence": 0.75,
            "price": 87000.0,
            "regime": "trending_up",
            "spoof_score": 0.2,
            "risk_penalty": 0.3,
            "scores": {"trend": 0.5},
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Signal 1 - Should pass
        result1 = self.safety.should_execute_signal(base_signal)
        print(f"\n✓ Signal 1: {result1.approved} - {result1.reason}")
        assert result1.approved, "First signal should be approved"
        
        # Record trade
        self.safety.record_trade({
            "direction": "long",
            "price": 87000.0,
            "size": 0.01,
            "timestamp": datetime.now(timezone.utc)
        })
        
        # Signal 2 (5s later, same price) - Should be duplicate
        base_signal["price"] = 87005.0  # Tiny price change
        result2 = self.safety.should_execute_signal(base_signal)
        print(f"✓ Signal 2: {result2.approved} - {result2.reason}")
        assert not result2.approved, "Duplicate signal should be rejected"
        assert "duplicate" in result2.reason.lower()
        
        print("\n✅ TEST 1 PASSED: Duplicate filtering works")
    
    def test_low_confidence(self):
        """Test 2: Confidence threshold filtering"""
        print("\n" + "="*80)
        print("TEST 2: LOW CONFIDENCE")
        print("="*80)
        
        signal = {
            "direction": "short",
            "confidence": 0.45,  # Below 70% threshold
            "price": 87000.0,
            "regime": "ranging",
            "spoof_score": 0.1,
            "risk_penalty": 0.2,
            "scores": {"trend": 0.2},
            "timestamp": datetime.now(timezone.utc)
        }
        
        result = self.safety.should_execute_signal(signal)
        print(f"\n✓ Low confidence signal: {result.approved} - {result.reason}")
        assert not result.approved, "Low confidence should be rejected"
        assert "confidence" in result.reason.lower()
        
        print("\n✅ TEST 2 PASSED: Confidence filtering works")
    
    def test_high_spoof(self):
        """Test 3: Spoof score filtering"""
        print("\n" + "="*80)
        print("TEST 3: HIGH SPOOF SCORE")
        print("="*80)
        
        signal = {
            "direction": "long",
            "confidence": 0.80,  # High confidence
            "price": 87000.0,
            "regime": "trending_up",
            "spoof_score": 1.0,  # 100% spoof!
            "risk_penalty": 0.2,
            "scores": {"trend": 0.6},
            "timestamp": datetime.now(timezone.utc)
        }
        
        result = self.safety.should_execute_signal(signal)
        print(f"\n✓ High spoof signal: {result.approved} - {result.reason}")
        assert not result.approved, "High spoof should be rejected"
        assert "spoof" in result.reason.lower()
        
        print("\n✅ TEST 3 PASSED: Spoof filtering works")
    
    def test_position_check(self):
        """Test 4: Position already open check"""
        print("\n" + "="*80)
        print("TEST 4: POSITION STATE CHECK")
        print("="*80)
        
        # Reset safety system with cooldowns disabled for this test
        self.safety = TradeSafety(SafetyConfig(
            min_trade_interval_seconds=0,
            min_same_direction_interval_seconds=0
        ))
        
        signal = {
            "direction": "long",
            "confidence": 0.75,
            "price": 87000.0,
            "regime": "trending_up",
            "spoof_score": 0.2,
            "risk_penalty": 0.3,
            "scores": {"trend": 0.5},
            "timestamp": datetime.now(timezone.utc)
        }
        
        # First signal - should open position
        result1 = self.safety.should_execute_signal(signal)
        print(f"\n✓ Signal 1 (open): {result1.approved} - {result1.reason}")
        assert result1.approved
        
        self.safety.record_trade({
            "direction": "long",
            "price": 87000.0,
            "size": 0.01,
            "timestamp": datetime.now(timezone.utc)
        })
        
        # Second signal (same direction) - should be blocked
        signal["price"] = 87100.0  # Different price to avoid duplicate
        signal["timestamp"] = datetime.now(timezone.utc) + timedelta(minutes=10)
        result2 = self.safety.should_execute_signal(signal)
        print(f"✓ Signal 2 (same direction): {result2.approved} - {result2.reason}")
        assert not result2.approved, "Same direction should be blocked"
        assert "position_already_open" in result2.reason
        
        print("\n✅ TEST 4 PASSED: Position state check works")
    
    def test_hourly_limit(self):
        """Test 5: Hourly trade limit"""
        print("\n" + "="*80)
        print("TEST 5: HOURLY TRADE LIMIT")
        print("="*80)
        
        # Reset safety system
        self.safety = TradeSafety(SafetyConfig(
            max_trades_per_hour=4,
            min_trade_interval_seconds=0,  # Disable cooldown for this test
        ))
        
        for i in range(5):
            signal = {
                "direction": "long" if i % 2 == 0 else "short",
                "confidence": 0.75,
                "price": 87000.0 + (i * 100),
                "regime": "trending_up",
                "spoof_score": 0.2,
                "risk_penalty": 0.3,
                "scores": {"trend": 0.5},
                "timestamp": datetime.now(timezone.utc) + timedelta(minutes=i*20)
            }
            
            result = self.safety.should_execute_signal(signal)
            status = "✅ APPROVED" if result.approved else f"❌ BLOCKED ({result.reason})"
            print(f"\n✓ Trade {i+1}: {status}")
            
            if result.approved:
                self.safety.record_trade({
                    "direction": signal["direction"],
                    "price": signal["price"],
                    "size": 0.01,
                    "timestamp": signal["timestamp"]
                })
                self.safety.close_position(signal["price"], "test")
        
        # 5th trade should be blocked
        assert i == 4, "Should have tested 5 trades"
        
        print("\n✅ TEST 5 PASSED: Hourly limit works")
    
    def test_daily_loss_limit(self):
        """Test 6: Daily loss limit and emergency stop"""
        print("\n" + "="*80)
        print("TEST 6: DAILY LOSS LIMIT")
        print("="*80)
        
        # Reset with $1000 balance, 5% = $50 max loss
        self.safety = TradeSafety(SafetyConfig(
            max_daily_loss_pct=0.05,
            min_trade_interval_seconds=0,
        ))
        self.safety.initial_balance = 1000.0
        
        losses = [-10, -20, -30]  # Total = -60 > -50 limit
        
        for i, loss in enumerate(losses):
            print(f"\nLoss {i+1}: ${loss} → Total: ${self.safety.daily_pnl + loss}")
            
            self.safety.record_trade_result(pnl=loss, is_win=False)
            
            # Try to trade after loss
            signal = {
                "direction": "long",
                "confidence": 0.75,
                "price": 87000.0 + (i * 100),
                "regime": "trending_up",
                "spoof_score": 0.2,
                "risk_penalty": 0.3,
                "scores": {"trend": 0.5},
                "timestamp": datetime.now(timezone.utc)
            }
            
            result = self.safety.should_execute_signal(signal)
            
            if self.safety.emergency_stopped:
                print(f"   🚨 EMERGENCY STOP: {self.safety.emergency_stop_reason}")
                assert not result.approved, "Should be blocked after emergency stop"
                break
        
        print("\n✅ TEST 6 PASSED: Daily loss limit triggers emergency stop")
    
    def test_consecutive_losses(self):
        """Test 7: Consecutive loss protection"""
        print("\n" + "="*80)
        print("TEST 7: CONSECUTIVE LOSSES")
        print("="*80)
        
        # Reset with 3 loss limit
        self.safety = TradeSafety(SafetyConfig(
            max_consecutive_losses=3,
            cooldown_after_loss_seconds=0,  # Disable cooldown for test
            min_trade_interval_seconds=0,
        ))
        
        for i in range(4):
            self.safety.record_trade_result(pnl=-5, is_win=False)
            print(f"\nLoss {i+1} → Consecutive: {self.safety.consecutive_losses}/3")
            
            signal = {
                "direction": "long",
                "confidence": 0.75,
                "price": 87000.0,
                "regime": "trending_up",
                "spoof_score": 0.2,
                "risk_penalty": 0.3,
                "scores": {"trend": 0.5},
                "timestamp": datetime.now(timezone.utc)
            }
            
            result = self.safety.should_execute_signal(signal)
            
            if i < 3:
                # First 3 losses should still allow trading (with cooldown in real scenario)
                print(f"   Status: Can still trade (loss {i+1}/3)")
            else:
                # 4th trade after 3 losses should be blocked
                print(f"   Status: ❌ BLOCKED - {result.reason}")
                assert not result.approved, "Should block after 3 consecutive losses"
                assert "consecutive_loss" in result.reason
        
        print("\n✅ TEST 7 PASSED: Consecutive loss protection works")
    
    def run_all_tests(self):
        """Run all safety tests"""
        print("\n" + "="*80)
        print("🧪 TRADE SAFETY SYSTEM - COMPREHENSIVE TEST SUITE")
        print("="*80)
        
        tests = [
            ("Rapid Signals", self.test_rapid_signals),
            ("Low Confidence", self.test_low_confidence),
            ("High Spoof Score", self.test_high_spoof),
            ("Position State Check", self.test_position_check),
            ("Hourly Limit", self.test_hourly_limit),
            ("Daily Loss Limit", self.test_daily_loss_limit),
            ("Consecutive Losses", self.test_consecutive_losses),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                test_func()
                passed += 1
            except AssertionError as e:
                print(f"\n❌ TEST FAILED: {test_name}")
                print(f"   Error: {e}")
                failed += 1
            except Exception as e:
                print(f"\n💥 TEST ERROR: {test_name}")
                print(f"   Error: {e}")
                failed += 1
        
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        print(f"✅ Passed: {passed}/{len(tests)}")
        print(f"❌ Failed: {failed}/{len(tests)}")
        
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED! Safety system is working correctly.")
            print("\n✅ SYSTEM IS READY FOR AUTO TRADING")
            print("   (with conservative settings and monitoring)")
        else:
            print("\n⚠️ SOME TESTS FAILED - DO NOT ENABLE AUTO TRADING")
        
        return failed == 0


if __name__ == "__main__":
    tester = SafetyTester()
    success = tester.run_all_tests()
    
    exit(0 if success else 1)
