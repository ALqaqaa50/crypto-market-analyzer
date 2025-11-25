"""
PROMETHEUS AI BRAIN v7 (OMEGA EDITION)
Dashboard API Endpoint

Exposes real-time AI decisions via REST API
"""

from flask import jsonify
import logging

logger = logging.getLogger(__name__)


def register_ai_endpoints(app):
    """Register AI endpoints to Flask app"""
    
    @app.route('/api/ai/live')
    def get_ai_live():
        """Get live AI decision"""
        try:
            from okx_stream_hunter.ai.brain_ultra import get_brain
            
            brain = get_brain()
            decision = brain.get_live_decision()
            
            if decision is None:
                return jsonify({
                    'status': 'warming_up',
                    'message': 'AI brain warming up - collecting data'
                }), 200
            
            return jsonify(decision), 200
        
        except Exception as e:
            logger.error(f"Error in /api/ai/live: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/ai/status')
    def get_ai_status():
        """Get AI brain status"""
        try:
            from okx_stream_hunter.ai.brain_ultra import get_brain
            
            brain = get_brain()
            status = brain.get_status()
            
            return jsonify(status), 200
        
        except Exception as e:
            logger.error(f"Error in /api/ai/status: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/ai/learning_status')
    def get_learning_status():
        """Get self-learning and model status (PHASE 4)"""
        try:
            from okx_stream_hunter.ai.self_learning_controller import get_self_learning_controller
            
            controller = get_self_learning_controller()
            status = controller.get_learning_status()
            
            return jsonify(status), 200
        
        except Exception as e:
            logger.error(f"Error in /api/ai/learning_status: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    logger.info("✅ AI endpoints registered: /api/ai/live, /api/ai/status, /api/ai/learning_status")
