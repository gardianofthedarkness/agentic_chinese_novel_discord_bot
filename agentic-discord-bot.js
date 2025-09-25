#!/usr/bin/env node
/**
 * Agentic Discord Bot with Intelligent Literary Agent
 * Unified commands powered by DeepSeek AI + RAG for dynamic character discovery
 * and context-aware responses
 */

const { Client, GatewayIntentBits, SlashCommandBuilder, REST, Routes, EmbedBuilder } = require('discord.js');
const axios = require('axios');
const winston = require('winston');
require('dotenv').config();

// Configure logging
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
    ),
    defaultMeta: { service: 'agentic-discord-bot' },
    transports: [
        new winston.transports.File({ filename: 'agentic-bot-error.log', level: 'error' }),
        new winston.transports.File({ filename: 'agentic-bot.log' }),
        new winston.transports.Console({
            format: winston.format.simple()
        })
    ]
});

// Bot configuration
const DISCORD_TOKEN = process.env.DISCORD_TOKEN;
const API_SERVER_URL = process.env.API_SERVER_URL || 'http://localhost:5005';
const CLIENT_ID = process.env.DISCORD_CLIENT_ID;
const GUILD_ID = process.env.DISCORD_GUILD_ID;

if (!DISCORD_TOKEN) {
    logger.error('DISCORD_TOKEN is required');
    process.exit(1);
}

// Create Discord client
const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent,
        GatewayIntentBits.DirectMessages
    ]
});

// Agentic API client
class AgenticAPIClient {
    constructor(baseURL) {
        this.baseURL = baseURL;
        this.client = axios.create({
            baseURL: baseURL,
            timeout: 120000, // Extended timeout for complex AI processing (2 minutes)
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }

    async healthCheck() {
        try {
            const response = await this.client.get('/health');
            return response.data;
        } catch (error) {
            logger.error('API health check failed:', error.message);
            throw error;
        }
    }

    async analyzeNovel(analysisType = 'full', limit = 10) {
        try {
            const response = await this.client.post('/api/agent/analyze', {
                type: analysisType,
                limit: limit
            });
            return response.data;
        } catch (error) {
            logger.error('Novel analysis failed:', error.message);
            throw error;
        }
    }

    async intelligentChat(message, conversationHistory = [], characterName = null) {
        try {
            const requestData = {
                message: message,
                history: conversationHistory
            };
            
            if (characterName) {
                requestData.character_name = characterName;
            }
            
            const response = await this.client.post('/api/agent/chat', requestData);
            return response.data;
        } catch (error) {
            logger.error('Intelligent chat failed:', error.message);
            throw error;
        }
    }

    async exploreTopic(topic, depth = 'medium') {
        try {
            const response = await this.client.post('/api/agent/explore', {
                topic: topic,
                depth: depth
            });
            return response.data;
        } catch (error) {
            logger.error('Topic exploration failed:', error.message);
            throw error;
        }
    }

    async getAgentMemory() {
        try {
            const response = await this.client.get('/api/agent/memory');
            return response.data;
        } catch (error) {
            logger.error('Agent memory retrieval failed:', error.message);
            throw error;
        }
    }

    async getAgentStatus() {
        try {
            const response = await this.client.get('/api/agent/status');
            return response.data;
        } catch (error) {
            logger.error('Agent status retrieval failed:', error.message);
            throw error;
        }
    }
}

// Initialize API client
const apiClient = new AgenticAPIClient(API_SERVER_URL);

// Global state for conversations
const conversationHistories = new Map(); // channelId -> messages[]

// Enhanced function to handle Discord message limits intelligently
function processLongResponse(text, maxEmbedLength = 4000) {
    if (text.length <= maxEmbedLength) {
        return { mainContent: text, additionalChunks: [] };
    }
    
    // For very long responses, create a summary for the embed
    const summaryLength = Math.min(500, maxEmbedLength - 200);
    const summary = text.substring(0, summaryLength);
    const lastSentenceEnd = Math.max(
        summary.lastIndexOf('。'),
        summary.lastIndexOf('！'),
        summary.lastIndexOf('？'),
        summary.lastIndexOf('.'),
        summary.lastIndexOf('\n')
    );
    
    let mainContent;
    if (lastSentenceEnd > summaryLength * 0.6) {
        mainContent = summary.substring(0, lastSentenceEnd + 1) + '\n\n*[Continue reading in follow-up messages...]*';
    } else {
        mainContent = summary + '...\n\n*[Continue reading in follow-up messages...]*';
    }
    
    // Split remaining content into Discord-friendly chunks
    const remainingText = text.substring(lastSentenceEnd + 1 || summaryLength);
    const chunks = splitIntoChunks(remainingText, 1900);
    
    return { mainContent, additionalChunks: chunks };
}

// Helper function to split text into chunks at natural breakpoints
function splitIntoChunks(text, maxChunkSize = 1900) {
    if (text.length <= maxChunkSize) {
        return text.trim() ? [text.trim()] : [];
    }
    
    const chunks = [];
    let currentChunk = '';
    
    // Split by paragraphs first, then sentences
    const paragraphs = text.split(/\n\s*\n/);
    
    for (const paragraph of paragraphs) {
        if (currentChunk.length + paragraph.length + 2 <= maxChunkSize) {
            currentChunk += (currentChunk ? '\n\n' : '') + paragraph;
        } else {
            // If current chunk has content, save it
            if (currentChunk.trim()) {
                chunks.push(currentChunk.trim());
            }
            
            // If paragraph is too long, split by sentences
            if (paragraph.length > maxChunkSize) {
                const sentences = paragraph.split(/([。！？.])/);
                currentChunk = '';
                
                for (let i = 0; i < sentences.length; i += 2) {
                    const sentence = sentences[i] + (sentences[i + 1] || '');
                    
                    if (currentChunk.length + sentence.length <= maxChunkSize) {
                        currentChunk += sentence;
                    } else {
                        if (currentChunk.trim()) {
                            chunks.push(currentChunk.trim());
                        }
                        currentChunk = sentence;
                    }
                }
            } else {
                currentChunk = paragraph;
            }
        }
    }
    
    // Add any remaining content
    if (currentChunk.trim()) {
        chunks.push(currentChunk.trim());
    }
    
    return chunks;
}

// New Agentic Slash Commands
const commands = [
    new SlashCommandBuilder()
        .setName('analyze')
        .setDescription('AI-powered analysis of the novel')
        .addStringOption(option =>
            option.setName('type')
                .setDescription('Type of analysis to perform')
                .setRequired(false)
                .addChoices(
                    { name: 'Full Analysis (Characters + Storylines)', value: 'full' },
                    { name: 'Character Discovery', value: 'characters' },
                    { name: 'Storyline Analysis', value: 'storylines' }
                ))
        .addIntegerOption(option =>
            option.setName('depth')
                .setDescription('Analysis depth (number of samples)')
                .setRequired(false)
                .setMinValue(1)
                .setMaxValue(20)),

    new SlashCommandBuilder()
        .setName('chat')
        .setDescription('Intelligent conversation with AI about the novel')
        .addStringOption(option =>
            option.setName('message')
                .setDescription('Your message or question')
                .setRequired(true))
        .addStringOption(option =>
            option.setName('character')
                .setDescription('Chat with a specific character (optional)')
                .setRequired(false)),

    new SlashCommandBuilder()
        .setName('explore')
        .setDescription('Deep exploration of specific topics')
        .addStringOption(option =>
            option.setName('topic')
                .setDescription('Topic to explore (e.g., "character development", "主题分析")')
                .setRequired(true))
        .addStringOption(option =>
            option.setName('depth')
                .setDescription('Analysis depth')
                .setRequired(false)
                .addChoices(
                    { name: 'Surface Level', value: 'shallow' },
                    { name: 'Medium Depth', value: 'medium' },
                    { name: 'Deep Analysis', value: 'deep' }
                )),

    new SlashCommandBuilder()
        .setName('memory')
        .setDescription('View what the AI has learned about the novel'),

    new SlashCommandBuilder()
        .setName('status')
        .setDescription('Check AI agent system status'),

    new SlashCommandBuilder()
        .setName('clear')
        .setDescription('Clear conversation history in this channel')
];

// Event handlers
client.once('ready', async () => {
    logger.info(`Agentic bot logged in as ${client.user.tag}`);
    
    // Check API server connection
    try {
        const health = await apiClient.healthCheck();
        logger.info('Agentic API server connection successful:', health);
    } catch (error) {
        logger.warn('API server connection failed - some features may not work');
    }
    
    // Register slash commands
    if (CLIENT_ID) {
        try {
            await registerCommands();
            logger.info('Agentic slash commands registered successfully');
        } catch (error) {
            logger.error('Failed to register slash commands:', error);
        }
    }
});

client.on('messageCreate', async (message) => {
    // Ignore bot messages
    if (message.author.bot) return;
    
    // For now, don't auto-respond to all messages (can be enabled later)
    // The new system focuses on intentional commands
});

client.on('interactionCreate', async (interaction) => {
    if (!interaction.isChatInputCommand()) return;

    const { commandName } = interaction;

    try {
        switch (commandName) {
            case 'analyze':
                await handleAnalyzeCommand(interaction);
                break;
            case 'chat':
                await handleChatCommand(interaction);
                break;
            case 'explore':
                await handleExploreCommand(interaction);
                break;
            case 'memory':
                await handleMemoryCommand(interaction);
                break;
            case 'status':
                await handleStatusCommand(interaction);
                break;
            case 'clear':
                await handleClearCommand(interaction);
                break;
            default:
                await interaction.reply('Unknown command');
        }
    } catch (error) {
        logger.error('Command error:', error);
        const errorMessage = 'An error occurred while processing your command.';
        
        if (interaction.replied || interaction.deferred) {
            await interaction.editReply(errorMessage);
        } else {
            await interaction.reply(errorMessage);
        }
    }
});

// Command handlers
async function handleAnalyzeCommand(interaction) {
    await interaction.deferReply();
    
    const analysisType = interaction.options.getString('type') || 'full';
    const depth = interaction.options.getInteger('depth') || 10;
    
    try {
        const result = await apiClient.analyzeNovel(analysisType, depth);
        
        const embed = new EmbedBuilder()
            .setTitle(`📊 Enhanced Novel Analysis: ${analysisType}`)
            .setColor(0x00ff00)
            .setTimestamp();
        
        // Handle new enhanced format with processed PostgreSQL data
        if (result.results.analysis_summary) {
            const summary = result.results.analysis_summary;
            embed.addFields({
                name: '📈 Processing Summary',
                value: `Chapters: ${summary.chapters_processed}\nCharacters: ${summary.characters_discovered}\nStorylines: ${summary.storylines_identified}\nTimeline Events: ${summary.timeline_events}`,
                inline: true
            });
            
            if (summary.recent_chapter) {
                embed.addFields({
                    name: '📖 Latest Chapter',
                    value: `${summary.recent_chapter.index}: ${summary.recent_chapter.title}`,
                    inline: true
                });
            }
        }
        
        // Character breakdown
        if (result.results.character_breakdown) {
            const breakdown = result.results.character_breakdown;
            const characterParts = [];
            
            if (breakdown.protagonists.length > 0) {
                characterParts.push(`**主角**: ${breakdown.protagonists.map(c => c.name).join(', ')}`);
            }
            if (breakdown.antagonists.length > 0) {
                characterParts.push(`**反派**: ${breakdown.antagonists.map(c => c.name).join(', ')}`);
            }
            if (breakdown.supporting.length > 0) {
                characterParts.push(`**配角**: ${breakdown.supporting.map(c => c.name).slice(0, 3).join(', ')}`);
            }
            
            if (characterParts.length > 0) {
                embed.addFields({
                    name: '🎭 Character Breakdown',
                    value: characterParts.join('\n'),
                    inline: false
                });
            }
        }
        
        // Storyline overview
        if (result.results.storyline_overview && result.results.storyline_overview.length > 0) {
            const storylineInfo = result.results.storyline_overview.slice(0, 3).map(s => 
                `**${s.title}** (${s.type}) - Importance: ${s.importance}`
            ).join('\n');
            
            embed.addFields({
                name: '📚 Key Storylines',
                value: storylineInfo,
                inline: false
            });
        }
        
        // System status
        if (result.results.system_status) {
            embed.addFields({
                name: '🔧 System Info',
                value: `Source: ${result.results.system_status.data_source}\nRAG: ${result.results.system_status.rag_enabled ? '✅' : '❌'}\nAI: ${result.results.system_status.ai_model}`,
                inline: true
            });
        }
        
        await interaction.editReply({ embeds: [embed] });
        
    } catch (error) {
        await interaction.editReply('Failed to analyze novel. The AI system may be processing or unavailable.');
    }
}

async function handleChatCommand(interaction) {
    await interaction.deferReply();
    
    const message = interaction.options.getString('message');
    
    // Send a progress indicator for complex requests
    if (message.length > 50) {
        await interaction.editReply('🧠 AI is processing your complex request... This may take up to 2 minutes.');
    }
    const characterName = interaction.options.getString('character');
    const channelId = interaction.channel.id;
    
    // Get conversation history
    const history = conversationHistories.get(channelId) || [];
    
    try {
        // Add character_name to request if specified
        const chatData = {
            message: message,
            history: history
        };
        
        if (characterName) {
            chatData.character_name = characterName;
        }
        
        const result = await apiClient.intelligentChat(message, history, characterName);
        
        const embed = new EmbedBuilder()
            .setTimestamp();
        
        // Process the response for Discord limits
        const { mainContent, additionalChunks } = processLongResponse(result.response);
        
        // Different styling based on whether it's character chat or general chat
        if (result.character) {
            // Character roleplay response
            embed.setTitle(`🎭 ${result.character.name}`)
                .setDescription(mainContent)
                .setColor(0xff6b6b);
            
            if (result.character.traits) {
                embed.addFields({
                    name: '✨ Character Traits',
                    value: result.character.traits.join(', '),
                    inline: true
                });
            }
            
            embed.addFields({
                name: '🎪 Roleplay Mode',
                value: `Type: ${result.character.type || 'Unknown'}`,
                inline: true
            });
        } else {
            // General AI assistant response
            embed.setTitle('🤖 AI Literary Assistant')
                .setDescription(mainContent)
                .setColor(0x0099ff);
        }
        
        // Add context information
        if (result.context) {
            const contextParts = [];
            if (result.context.characters_available && result.context.characters_available.length > 0) {
                contextParts.push(`Characters: ${result.context.characters_available.slice(0, 5).join(', ')}`);
            }
            if (result.context.storylines_tracked) {
                contextParts.push(`Storylines: ${result.context.storylines_tracked}`);
            }
            if (result.context.rag_sources) {
                contextParts.push(`Sources: ${result.context.rag_sources}`);
            }
            
            if (contextParts.length > 0) {
                embed.addFields({
                    name: '📊 Context',
                    value: contextParts.join(' | '),
                    inline: false
                });
            }
        }
        
        // Send the main embed
        await interaction.editReply({ embeds: [embed] });
        
        // Send additional chunks if any
        if (additionalChunks.length > 0) {
            // Add a small delay to ensure proper message ordering
            await new Promise(resolve => setTimeout(resolve, 500));
            
            for (let i = 0; i < additionalChunks.length; i++) {
                const chunk = additionalChunks[i];
                if (chunk.trim()) {
                    // Add chunk numbers for very long responses
                    const chunkHeader = additionalChunks.length > 1 ? `**[Part ${i + 1}/${additionalChunks.length}]**\n\n` : '';
                    await interaction.followUp({ 
                        content: chunkHeader + chunk.trim(), 
                        ephemeral: false 
                    });
                    
                    // Progressive delay between chunks to avoid rate limits
                    if (i < additionalChunks.length - 1) {
                        const delay = Math.min(1000 + (i * 200), 3000); // Progressive delay up to 3s
                        await new Promise(resolve => setTimeout(resolve, delay));
                    }
                }
            }
        }
        
        // Update conversation history
        history.push(
            { role: 'user', content: message },
            { role: 'assistant', content: result.response }
        );
        
        // Keep only last 10 messages
        if (history.length > 10) {
            history.splice(0, history.length - 10);
        }
        
        conversationHistories.set(channelId, history);
        
    } catch (error) {
        logger.error('Chat command error:', error);
        
        // More specific error handling for different failure types
        let errorMessage = 'Failed to process your message. ';
        
        if (error.message.includes('timeout') || error.message.includes('ECONNRESET')) {
            errorMessage += 'The AI system is taking longer than expected. Please try again with a shorter message.';
        } else if (error.message.includes('413') || error.message.includes('too large')) {
            errorMessage += 'Your message may be too long. Please try a shorter question.';
        } else if (error.message.includes('408') || error.message.includes('Request Timeout')) {
            errorMessage += 'The AI system may be busy processing. Please wait a moment and try again.';
        } else if (error.message.includes('429') || error.message.includes('rate limit')) {
            errorMessage += 'Rate limit reached. Please wait a moment before trying again.';
        } else if (error.message.includes('50013') || error.message.includes('Missing Permissions')) {
            errorMessage += 'Bot missing permissions to send messages or embeds.';
        } else {
            errorMessage += 'The AI system may be busy or characters may not be available yet. Please try again.';
        }
        
        await interaction.editReply(errorMessage);
    }
}

async function handleExploreCommand(interaction) {
    await interaction.deferReply();
    
    const topic = interaction.options.getString('topic');
    const depth = interaction.options.getString('depth') || 'medium';
    
    try {
        const result = await apiClient.exploreTopic(topic, depth);
        
        // Process exploration response for Discord limits
        const { mainContent, additionalChunks } = processLongResponse(result.analysis);

        const embed = new EmbedBuilder()
            .setTitle(`🔍 Deep Exploration: ${topic}`)
            .setDescription(mainContent)
            .setColor(0xff9900)
            .setFooter({ text: `Analysis Depth: ${depth}` })
            .setTimestamp();
        
        // Send main embed
        await interaction.editReply({ embeds: [embed] });
        
        // Send additional chunks if any
        if (additionalChunks.length > 0) {
            await new Promise(resolve => setTimeout(resolve, 500));
            
            for (let i = 0; i < additionalChunks.length; i++) {
                const chunk = additionalChunks[i];
                if (chunk.trim()) {
                    const chunkHeader = additionalChunks.length > 1 ? `**[Analysis Part ${i + 1}/${additionalChunks.length}]**\n\n` : '';
                    await interaction.followUp({ 
                        content: chunkHeader + chunk.trim(), 
                        ephemeral: false 
                    });
                    
                    if (i < additionalChunks.length - 1) {
                        const delay = Math.min(1000 + (i * 200), 3000); // Progressive delay up to 3s
                        await new Promise(resolve => setTimeout(resolve, delay));
                    }
                }
            }
        }
        
    } catch (error) {
        await interaction.editReply('Failed to explore the topic. Please try again later.');
    }
}

async function handleMemoryCommand(interaction) {
    try {
        await interaction.deferReply();
        
        const result = await apiClient.getAgentMemory();
        
        const embed = new EmbedBuilder()
            .setTitle('🧠 Enhanced AI Agent Memory')
            .setColor(0x800080)
            .setTimestamp();
        
        const stats = result.memory.stats || {};
        embed.addFields({
            name: '📊 Memory Statistics',
            value: `Characters: ${stats.character_count || 0}\nStorylines: ${stats.storyline_count || 0}\nTimeline Events: ${stats.timeline_events || 0}\nChapters: ${result.memory.chapters_processed || 0}`,
            inline: true
        });
        
        // Show characters if available
        const characters = result.memory.characters || {};
        if (Object.keys(characters).length > 0) {
            const charList = Object.keys(characters).slice(0, 8).map(name => {
                const char = characters[name];
                return `**${name}** (${char.type || 'unknown'})`;
            }).join('\n');
            
            embed.addFields({
                name: '👥 Discovered Characters',
                value: charList || 'Loading characters...',
                inline: false
            });
        } else {
            embed.addFields({
                name: '👥 Characters',
                value: 'Characters are being processed from the novel...',
                inline: false
            });
        }
        
        // Show storylines if available
        const storylines = result.memory.storylines || [];
        if (storylines.length > 0) {
            const storylineList = storylines.slice(0, 3)
                .map(s => `• **${s.title}** (${s.type}) - Importance: ${s.importance}`)
                .join('\n');
            embed.addFields({
                name: '📚 Active Storylines',
                value: storylineList,
                inline: false
            });
        } else {
            embed.addFields({
                name: '📚 Storylines',
                value: 'Storyline analysis in progress...',
                inline: false
            });
        }
        
        embed.addFields({
            name: '🔧 System Status',
            value: `Data Source: PostgreSQL\nLast Updated: ${result.memory.last_updated || 'Recently'}`,
            inline: false
        });
        
        await interaction.editReply({ embeds: [embed] });
        
    } catch (error) {
        logger.error('Memory command error:', error);
        if (!interaction.replied && !interaction.deferred) {
            await interaction.reply('Failed to retrieve agent memory. The system may still be loading.');
        } else {
            await interaction.editReply('Failed to retrieve agent memory. The system may still be loading.');
        }
    }
}

async function handleStatusCommand(interaction) {
    await interaction.deferReply();
    
    try {
        const health = await apiClient.healthCheck();
        const agentStatus = await apiClient.getAgentStatus();
        
        const embed = new EmbedBuilder()
            .setTitle('🤖 Agentic System Status')
            .setColor(health.services.intelligent_agent ? 0x00ff00 : 0xff0000)
            .setTimestamp()
            .addFields(
                { name: 'System Mode', value: health.mode || 'Unknown', inline: true },
                { name: 'Intelligent Agent', value: health.services.intelligent_agent ? '✅ Active' : '❌ Inactive', inline: true },
                { name: 'RAG System', value: health.services.rag ? '✅ Active' : '❌ Inactive', inline: true },
                { name: 'DeepSeek AI', value: health.services.deepseek ? '✅ Active' : '❌ Inactive', inline: true },
                { name: 'Character Discovery', value: health.services.character_discovery ? '✅ Active' : '❌ Inactive', inline: true },
                { name: 'Storyline Analysis', value: health.services.storyline_analysis ? '✅ Active' : '❌ Inactive', inline: true }
            );
        
        if (agentStatus.capabilities) {
            embed.addFields({
                name: '🔧 Agent Capabilities',
                value: agentStatus.capabilities.join(', '),
                inline: false
            });
        }
        
        await interaction.editReply({ embeds: [embed] });
        
    } catch (error) {
        const embed = new EmbedBuilder()
            .setTitle('🤖 System Status')
            .setColor(0xff0000)
            .setTimestamp()
            .addFields(
                { name: 'Discord Bot', value: '✅ Online', inline: true },
                { name: 'API Server', value: '❌ Offline', inline: true }
            );
        
        await interaction.editReply({ embeds: [embed] });
    }
}

async function handleClearCommand(interaction) {
    const channelId = interaction.channel.id;
    conversationHistories.delete(channelId);
    
    await interaction.reply('Conversation history cleared for this channel.');
}

// Register slash commands
async function registerCommands() {
    const rest = new REST().setToken(DISCORD_TOKEN);
    
    try {
        const data = await rest.put(
            GUILD_ID 
                ? Routes.applicationGuildCommands(CLIENT_ID, GUILD_ID)
                : Routes.applicationCommands(CLIENT_ID),
            { body: commands }
        );
        
        logger.info(`Successfully registered ${data.length} agentic commands.`);
    } catch (error) {
        logger.error('Error registering commands:', error);
        throw error;
    }
}

// Error handling
process.on('unhandledRejection', (reason, promise) => {
    logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (error) => {
    logger.error('Uncaught Exception:', error);
    process.exit(1);
});

// Graceful shutdown
process.on('SIGINT', () => {
    logger.info('Received SIGINT, shutting down gracefully');
    client.destroy();
    process.exit(0);
});

process.on('SIGTERM', () => {
    logger.info('Received SIGTERM, shutting down gracefully');
    client.destroy();
    process.exit(0);
});

// Start the bot
client.login(DISCORD_TOKEN);