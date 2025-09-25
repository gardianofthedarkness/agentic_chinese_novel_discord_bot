#!/usr/bin/env node
/**
 * Test script for Discord.js bot and API integration
 * Verifies all connections and services are working
 */

const axios = require('axios');
require('dotenv').config();

const API_SERVER_URL = process.env.API_SERVER_URL || 'http://localhost:5000';
const DISCORD_TOKEN = process.env.DISCORD_TOKEN;

console.log('🧪 Testing Discord.js Bot and API Integration');
console.log('=' .repeat(50));

// Test results
let testResults = {
    envConfig: false,
    apiHealth: false,
    ragQuery: false,
    characterList: false,
    textGeneration: false,
    discordToken: false
};

// Test 1: Environment Configuration
function testEnvironmentConfig() {
    console.log('\n📋 Test 1: Environment Configuration');
    
    const requiredVars = ['DISCORD_TOKEN'];
    const optionalVars = ['DISCORD_CLIENT_ID', 'API_SERVER_URL', 'DEEPSEEK_API_KEY'];
    
    console.log('Required variables:');
    requiredVars.forEach(varName => {
        const value = process.env[varName];
        const status = value ? '✅' : '❌';
        console.log(`   ${status} ${varName}: ${value ? 'Set' : 'Not set'}`);
    });
    
    console.log('Optional variables:');
    optionalVars.forEach(varName => {
        const value = process.env[varName];
        const status = value ? '✅' : '⚠️ ';
        console.log(`   ${status} ${varName}: ${value || 'Not set'}`);
    });
    
    const hasRequired = requiredVars.every(varName => process.env[varName]);
    testResults.envConfig = hasRequired;
    testResults.discordToken = !!process.env.DISCORD_TOKEN;
    
    console.log(`Result: ${hasRequired ? '✅ PASS' : '❌ FAIL'}`);
    
    return hasRequired;
}

// Test 2: API Server Health Check
async function testAPIServerHealth() {
    console.log('\n🏥 Test 2: API Server Health Check');
    console.log(`Testing connection to: ${API_SERVER_URL}`);
    
    try {
        const response = await axios.get(`${API_SERVER_URL}/health`, { timeout: 5000 });
        
        if (response.status === 200) {
            console.log('✅ API server is responding');
            console.log('Services status:');
            
            const services = response.data.services || {};
            Object.entries(services).forEach(([service, status]) => {
                const icon = status ? '✅' : '❌';
                console.log(`   ${icon} ${service}: ${status ? 'Available' : 'Unavailable'}`);
            });
            
            testResults.apiHealth = true;
            console.log('Result: ✅ PASS');
            return response.data;
        } else {
            console.log(`❌ API server responded with status ${response.status}`);
            console.log('Result: ❌ FAIL');
            return null;
        }
    } catch (error) {
        console.log(`❌ Failed to connect to API server: ${error.message}`);
        console.log('Result: ❌ FAIL');
        console.log('💡 Make sure to start the API server: python api_server.py');
        return null;
    }
}

// Test 3: RAG Query Test
async function testRAGQuery() {
    console.log('\n🔍 Test 3: RAG Query Test');
    
    try {
        const testQuery = 'What is the main theme?';
        console.log(`Testing query: "${testQuery}"`);
        
        const response = await axios.post(`${API_SERVER_URL}/api/rag/query`, {
            query: testQuery,
            collection: 'test_novel2',
            limit: 3
        }, { timeout: 10000 });
        
        if (response.status === 200) {
            const data = response.data;
            console.log(`✅ RAG query successful`);
            console.log(`   Query: ${data.query}`);
            console.log(`   Results: ${data.results ? data.results.length : 0} items`);
            
            testResults.ragQuery = true;
            console.log('Result: ✅ PASS');
            return data;
        } else {
            console.log(`❌ RAG query failed with status ${response.status}`);
            console.log('Result: ❌ FAIL');
            return null;
        }
    } catch (error) {
        console.log(`❌ RAG query error: ${error.message}`);
        console.log('Result: ❌ FAIL (Expected if RAG service not configured)');
        return null;
    }
}

// Test 4: Character List Test
async function testCharacterList() {
    console.log('\n👥 Test 4: Character List Test');
    
    try {
        const response = await axios.get(`${API_SERVER_URL}/api/character/list`, { timeout: 5000 });
        
        if (response.status === 200) {
            const data = response.data;
            console.log(`✅ Character list retrieved`);
            console.log(`   Characters available: ${data.count || 0}`);
            
            if (data.characters && data.characters.length > 0) {
                console.log('   Character names:');
                data.characters.forEach(char => {
                    console.log(`     - ${char.name || char}`);
                });
            }
            
            testResults.characterList = true;
            console.log('Result: ✅ PASS');
            return data;
        } else {
            console.log(`❌ Character list failed with status ${response.status}`);
            console.log('Result: ❌ FAIL');
            return null;
        }
    } catch (error) {
        console.log(`❌ Character list error: ${error.message}`);
        console.log('Result: ❌ FAIL (Expected if character service not configured)');
        return null;
    }
}

// Test 5: Text Generation Test
async function testTextGeneration() {
    console.log('\n✍️  Test 5: Text Generation Test');
    
    try {
        const testPrompt = 'Hello, how are you?';
        console.log(`Testing prompt: "${testPrompt}"`);
        
        const response = await axios.post(`${API_SERVER_URL}/api/deepseek/generate`, {
            prompt: testPrompt,
            max_tokens: 100,
            temperature: 0.7
        }, { timeout: 15000 });
        
        if (response.status === 200) {
            const data = response.data;
            console.log(`✅ Text generation successful`);
            console.log(`   Response: ${data.response ? data.response.substring(0, 100) + '...' : 'No response'}`);
            
            testResults.textGeneration = true;
            console.log('Result: ✅ PASS');
            return data;
        } else {
            console.log(`❌ Text generation failed with status ${response.status}`);
            console.log('Result: ❌ FAIL');
            return null;
        }
    } catch (error) {
        console.log(`❌ Text generation error: ${error.message}`);
        console.log('Result: ❌ FAIL (Expected if DeepSeek API key not configured)');
        return null;
    }
}

// Test 6: Discord Token Validation
function testDiscordToken() {
    console.log('\n🔑 Test 6: Discord Token Validation');
    
    if (!DISCORD_TOKEN) {
        console.log('❌ Discord token not found');
        console.log('Result: ❌ FAIL');
        return false;
    }
    
    if (DISCORD_TOKEN === 'your_discord_bot_token_here') {
        console.log('❌ Discord token is still placeholder value');
        console.log('Result: ❌ FAIL');
        return false;
    }
    
    // Basic token format validation
    if (!DISCORD_TOKEN.includes('.')) {
        console.log('❌ Discord token format appears invalid');
        console.log('Result: ❌ FAIL');
        return false;
    }
    
    console.log('✅ Discord token is configured');
    console.log(`   Token: ...${DISCORD_TOKEN.slice(-10)}`);
    console.log('Result: ✅ PASS');
    
    return true;
}

// Summary and recommendations
function printSummary() {
    console.log('\n📊 Test Summary');
    console.log('=' .repeat(50));
    
    const tests = [
        { name: 'Environment Configuration', result: testResults.envConfig },
        { name: 'API Server Health', result: testResults.apiHealth },
        { name: 'RAG Query', result: testResults.ragQuery },
        { name: 'Character List', result: testResults.characterList },
        { name: 'Text Generation', result: testResults.textGeneration },
        { name: 'Discord Token', result: testResults.discordToken }
    ];
    
    tests.forEach(test => {
        const status = test.result ? '✅' : '❌';
        console.log(`${status} ${test.name}`);
    });
    
    const passedTests = tests.filter(test => test.result).length;
    const totalTests = tests.length;
    
    console.log(`\n🎯 Overall: ${passedTests}/${totalTests} tests passed`);
    
    // Recommendations
    console.log('\n💡 Recommendations:');
    
    if (!testResults.envConfig) {
        console.log('   - Set up your .env file with required variables');
    }
    
    if (!testResults.apiHealth) {
        console.log('   - Start the API server: python api_server.py');
        console.log('   - Check if all Python dependencies are installed');
    }
    
    if (!testResults.discordToken) {
        console.log('   - Set a valid Discord bot token in your .env file');
    }
    
    if (!testResults.ragQuery) {
        console.log('   - Configure Qdrant database and RAG service');
    }
    
    if (!testResults.textGeneration) {
        console.log('   - Set DEEPSEEK_API_KEY in your .env file');
    }
    
    console.log('\n🚀 Ready to start?');
    if (testResults.envConfig && testResults.discordToken) {
        console.log('   You can start the bot with: npm start');
        console.log('   Or use the startup script: node start.js');
    } else {
        console.log('   Fix the failing tests above first');
    }
}

// Main test runner
async function runAllTests() {
    try {
        // Run all tests
        testEnvironmentConfig();
        await testAPIServerHealth();
        await testRAGQuery();
        await testCharacterList();
        await testTextGeneration();
        testDiscordToken();
        
        // Print summary
        printSummary();
        
        // Exit code
        const criticalTestsPassed = testResults.envConfig && testResults.discordToken;
        process.exit(criticalTestsPassed ? 0 : 1);
        
    } catch (error) {
        console.error('\n💥 Test runner error:', error);
        process.exit(1);
    }
}

// Run tests
runAllTests();