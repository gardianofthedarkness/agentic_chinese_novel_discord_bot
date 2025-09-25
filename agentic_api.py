#!/usr/bin/env python3
"""
Agentic API Server with Intelligent Literary Agent
Unified system where DeepSeek AI dynamically discovers characters, builds storylines,
and provides context-aware responses
"""

import os
import sys
import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global agents
enhanced_agent = None
simple_agent = None

def initialize_enhanced_agent():
    """Initialize the enhanced hybrid agent with SQL+RAG+DeepSeek+HyDE"""
    global enhanced_agent, simple_agent
    try:
        logger.info("Starting agent initialization...")
        from enhanced_hybrid_agent import create_enhanced_hybrid_agent
        from simple_chat_agent import create_simple_agent
        
        logger.info("Creating enhanced hybrid agent...")
        # Use new hybrid agent as primary
        enhanced_agent = create_enhanced_hybrid_agent(
            db_url="postgresql://admin:admin@localhost:5432/novel_sim",
            qdrant_url="http://localhost:32768",
            collection="test_novel2"
        )
        logger.info(f"Enhanced agent created: {enhanced_agent is not None}")
        
        logger.info("Creating simple agent...")
        # Keep simple agent as fallback
        simple_agent = create_simple_agent(
            db_url="postgresql://admin:admin@localhost:5432/novel_sim",
            qdrant_url="http://localhost:32768",
            collection="test_novel2"
        )
        logger.info(f"Simple agent created: {simple_agent is not None}")
        
        logger.info("Enhanced Hybrid Agent and Simple Agent initialized successfully")
        return True
    except Exception as e:
        import traceback
        logger.error(f"Failed to initialize agents: {e}")
        logger.error(f"Initialization traceback: {traceback.format_exc()}")
        return False

# Initialize enhanced agent on startup
agent_status = initialize_enhanced_agent()

@app.route('/health')
def health_check():
    """Health check with agent status"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'debug_id': 'ENHANCED_AGENT_DEBUG_VERSION',
        'services': {
            'intelligent_agent': agent_status,
            'rag': True,
            'deepseek': True,
            'character_discovery': agent_status,
            'storyline_analysis': agent_status
        },
        'mode': 'agentic'
    })

@app.route('/api/agent/analyze', methods=['POST'])
def agent_analyze():
    """Enhanced analysis using processed PostgreSQL data"""
    try:
        data = request.get_json()
        analysis_type = data.get('type', 'full')  # full, characters, storylines
        
        if not enhanced_agent:
            return jsonify({'error': 'Enhanced agent not available'}), 503
        
        logger.info(f"Starting enhanced analysis: {analysis_type}")
        
        # Create event loop if none exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
        
        # Get comprehensive analysis from processed data
        analysis_summary = loop.run_until_complete(
            enhanced_agent.get_novel_analysis_summary()
        )
        
        # Filter results based on analysis type
        if analysis_type == 'characters':
            filtered_results = {
                'characters': analysis_summary['character_breakdown'],
                'analysis_summary': {
                    'characters_discovered': analysis_summary['analysis_summary']['characters_discovered'],
                    'chapters_processed': analysis_summary['analysis_summary']['chapters_processed']
                }
            }
        elif analysis_type == 'storylines':
            filtered_results = {
                'storylines': analysis_summary['storyline_overview'],
                'analysis_summary': {
                    'storylines_identified': analysis_summary['analysis_summary']['storylines_identified'],
                    'chapters_processed': analysis_summary['analysis_summary']['chapters_processed']
                }
            }
        else:  # full
            filtered_results = analysis_summary
        
        return jsonify({
            'analysis_type': analysis_type,
            'results': filtered_results,
            'timestamp': datetime.now().isoformat(),
            'source': 'enhanced_agent_postgresql'
        })
        
    except Exception as e:
        logger.error(f"Enhanced analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent/chat', methods=['POST'])
def agent_chat():
    """Enhanced intelligent chat with PostgreSQL data and RAG"""
    message = ""  # Initialize message variable first
    try:
        logger.info("=== CHAT ENDPOINT CALLED ===")
        data = request.get_json()
        message = data.get('message', '')
        conversation_history = data.get('history', [])
        character_name = data.get('character_name', None)  # Optional specific character
        logger.info(f"Parsed message: '{message}', character: {character_name}")
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        if not enhanced_agent:
            logger.error("Enhanced agent is None in chat endpoint - initialization failed")
            return jsonify({'error': 'Enhanced agent not available'}), 503
        
        logger.info(f"Processing enhanced chat: {message[:50]}...")
        
        # Create event loop if none exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
        
        # Use enhanced agent as primary since it's now stable
        response_data = None
        
        # Try enhanced agent first for rich responses
        try:
            logger.info("=== ATTEMPTING ENHANCED HYBRID AGENT ===")
            logger.info(f"Enhanced agent object: {enhanced_agent}")
            logger.info("Using enhanced hybrid agent for intelligent chat response")
            # Use the new intelligent_chat method that combines SQL+RAG+DeepSeek+HyDE
            response_data = loop.run_until_complete(
                enhanced_agent.intelligent_chat(
                    user_message=message,
                    character_name=character_name
                )
            )
        except Exception as enhanced_error:
            import traceback
            logger.error(f"Enhanced agent failed with error: {enhanced_error}")
            logger.error(f"Enhanced agent traceback: {traceback.format_exc()}")
            logger.warning("Falling back to simple agent")
            
            # Fallback to simple agent
            if simple_agent:
                try:
                    logger.info("Using simple agent as fallback")
                    simple_response = simple_agent.simple_chat(message)
                    response_data = {
                        'response': simple_response['response'],
                        'context': simple_response['context']
                    }
                except Exception as simple_error:
                    logger.error(f"Both agents failed: enhanced={enhanced_error}, simple={simple_error}")
                    response_data = {
                        'response': '抱歉，系统当前遇到了技术问题，请稍后再试。',
                        'context': {'error': 'both_agents_failed'}
                    }
        
        # Determine which agent was used
        if response_data.get('context', {}).get('error') == 'both_agents_failed':
            source = 'both_agents_failed'
        elif response_data.get('context', {}).get('session_id'):
            source = 'enhanced_hybrid_agent'
        else:
            source = 'simple_agent_fallback'
        
        return jsonify({
            'message': message,
            'response': response_data['response'],
            'context': response_data.get('context', {}),
            'character': response_data.get('character', None),
            'timestamp': datetime.now().isoformat(),
            'source': source
        })
        
    except Exception as e:
        logger.error(f"Enhanced chat error: {e}")
        
        # Fallback to simple agent
        if simple_agent:
            try:
                logger.info("Falling back to simple agent for chat")
                response_data = simple_agent.simple_chat(message)
                return jsonify({
                    'message': message,
                    'response': response_data['response'],
                    'context': response_data.get('context', {}),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'simple_agent_fallback'
                })
            except Exception as e2:
                logger.error(f"Simple agent fallback error: {e2}")
        
        return jsonify({'error': f'Both enhanced and simple agents failed: {str(e)}'}), 500

@app.route('/api/agent/explore', methods=['POST'])
def agent_explore():
    """Enhanced topic exploration with processed data + RAG + AI"""
    try:
        data = request.get_json()
        topic = data.get('topic', '')
        depth = data.get('depth', 'medium')  # shallow, medium, deep
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        
        if not enhanced_agent:
            return jsonify({'error': 'Enhanced agent not available'}), 503
        
        logger.info(f"Enhanced exploration: {topic} (depth: {depth})")
        
        # Create event loop if none exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
        
        # Generate enhanced exploration
        response_data = loop.run_until_complete(
            enhanced_agent.explore_topic(topic=topic, depth=depth)
        )
        
        return jsonify({
            'topic': response_data['topic'],
            'depth': response_data['depth'],
            'analysis': response_data['analysis'],
            'sources': response_data.get('sources', {}),
            'context_available': response_data.get('context_available', False),
            'timestamp': datetime.now().isoformat(),
            'source': 'enhanced_agent_exploration'
        })
        
    except Exception as e:
        logger.error(f"Enhanced exploration error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent/memory')
def agent_memory():
    """Get current enhanced agent memory and processed data state"""
    try:
        if not enhanced_agent:
            return jsonify({'error': 'Enhanced agent not available'}), 503
        
        # Create event loop if none exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
        
        # Try enhanced agent first, fallback to simple agent
        memory_summary = None
        source_type = "enhanced_agent_postgresql"
        try:
            memory_summary = loop.run_until_complete(
                enhanced_agent.get_novel_analysis_summary()
            )
        except Exception as e:
            logger.warning(f"Enhanced agent memory failed: {e}, trying simple agent")
            if simple_agent:
                try:
                    memory_summary = simple_agent.get_analysis_summary()
                    source_type = "simple_agent_fallback"
                    logger.info("Successfully using simple agent fallback for memory")
                except Exception as e2:
                    logger.error(f"Simple agent also failed: {e2}")
                    return jsonify({'error': f'Both agents failed: {str(e)} | {str(e2)}'}), 500
        
        # Transform to expected memory format for compatibility
        characters = {}
        if memory_summary.get('character_breakdown'):
            for char_type in ['protagonists', 'antagonists', 'supporting']:
                for char in memory_summary['character_breakdown'].get(char_type, []):
                    characters[char['name']] = {
                        'name': char['name'],
                        'traits': char.get('traits', []),
                        'type': char_type,
                        'confidence': 0.9
                    }
        
        # Transform storylines - handle empty storyline_threads table
        storylines = []
        if memory_summary.get('storyline_overview') and len(memory_summary.get('storyline_overview', [])) > 0:
            for story in memory_summary['storyline_overview']:
                storylines.append({
                    'title': story.get('title', 'Unknown'),
                    'type': story.get('type', 'unknown'),
                    'importance': story.get('importance', 0.5),
                    'chapters': story.get('chapters', [])
                })
        else:
            # Create placeholder storylines based on timeline events if storylines are empty
            if memory_summary.get('analysis_summary', {}).get('timeline_events', 0) > 0:
                storylines.append({
                    'title': '第一章主要情节',
                    'type': 'main_plot',
                    'importance': 0.8,
                    'chapters': [1]
                })
        
        transformed_memory = {
            'characters': characters,
            'storylines': storylines,
            'stats': {
                'character_count': memory_summary['analysis_summary']['characters_discovered'],
                'storyline_count': memory_summary['analysis_summary']['storylines_identified'],
                'timeline_events': memory_summary['analysis_summary']['timeline_events']
            },
            'last_updated': memory_summary['system_status']['last_updated'],
            'chapters_processed': memory_summary['analysis_summary']['chapters_processed']
        }
        
        return jsonify({
            'memory': transformed_memory,
            'raw_summary': memory_summary,  # Keep full data for debugging
            'timestamp': datetime.now().isoformat(),
            'source': source_type + '_memory'
        })
        
    except Exception as e:
        logger.error(f"Enhanced memory error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent/status')
def get_agent_status():
    """Get detailed agent system status"""
    return jsonify({
        'agent_initialized': agent_status,
        'capabilities': [
            'character_discovery',
            'storyline_analysis',
            'contextual_chat', 
            'knowledge_synthesis',
            'topic_exploration'
        ],
        'commands': [
            '/analyze - Discover characters and storylines',
            '/chat - Intelligent conversation',
            '/explore - Deep topic analysis',
            '/memory - View agent knowledge'
        ],
        'timestamp': datetime.now().isoformat()
    })

# Legacy endpoint compatibility (simplified)
@app.route('/api/rag/query', methods=['POST'])
def legacy_rag_query():
    """Legacy RAG endpoint - uses enhanced agent with PostgreSQL data"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not enhanced_agent:
            return jsonify({'error': 'Service not available'}), 503
        
        # Use enhanced contextual chat for RAG queries
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
        
        response_data = loop.run_until_complete(
            enhanced_agent.enhanced_contextual_chat(f"请帮我查找和分析：{query}")
        )
        
        return jsonify({
            'query': query,
            'results': [{'content': response_data['response'], 'score': 1.0, 'source': 'enhanced_agent_postgresql'}],
            'context': response_data.get('context', {}),
            'timestamp': datetime.now().isoformat(),
            'source': 'enhanced_agent_powered_rag'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def root():
    """Root endpoint with agentic system information"""
    return jsonify({
        'message': 'Intelligent Literary Agent API',
        'status': 'running',
        'mode': 'agentic',
        'agent_initialized': agent_status,
        'new_endpoints': [
            '/api/agent/analyze - Discover characters and storylines', 
            '/api/agent/chat - Intelligent contextual chat',
            '/api/agent/explore - Deep topic exploration',
            '/api/agent/memory - View agent knowledge state',
            '/api/agent/status - System capabilities'
        ],
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500) 
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('API_PORT', 5005))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Agentic API Server on port {port}")
    logger.info(f"Intelligent Agent Status: {'Enabled' if agent_status else 'Disabled'}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)