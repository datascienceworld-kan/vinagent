import React, { useState, useEffect, useRef } from 'react';
import './ChatPanel.css';
import string from "../../config/string.ts";
import {Box, Button, TextField, Typography, Alert, CircularProgress} from "@mui/material";
import AgentToolSelector from "./AgentToolSelector/AgentToolSelector.tsx";
import ChatMessageDisplay from "./ChatMessageDisplay/ChatMessageDisplay.tsx";
import type {
    ArtifactDataPayload,
    ChatMessage,
    OutgoingQueryMessage,
    IncomingWebSocketMessage
} from '../../types/types.ts';
import SendIcon from '@mui/icons-material/Send';
import websocketService from "../../services/websocketService.ts";
import {fetchAgentTools} from "../../services/apiService.ts";
import { v4 as uuidv4 } from 'uuid';

interface ChatPanelProps {
    setDisplayedArtifactData: (data: ArtifactDataPayload | null) => void;
}

const getArtifactDisplayText = (artifactData: ArtifactDataPayload): string => {
    if (!artifactData) return string.leftPanel.artifactLinkDefaultText;

    switch (artifactData.type) {
        case 'table':
            return `View Table (${artifactData.rows.length} rows)`;
        case 'plotly':
            return 'View Plot';
        case 'markdown':
            return 'View Notes';
        default:
            return string.leftPanel.artifactLinkDefaultText;
    }
};

const ChatPanel: React.FC<ChatPanelProps> = ({ setDisplayedArtifactData }) => {

    const [selectedTool, setSelectedTool] = useState<string>('');
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [inputText, setInputText] = useState<string>('');
    const [connectionStatus, setConnectionStatus] = useState<string>(string.leftPanel.connectionStatus.disconnected);
    const [showConnectedStatusTemporarily, setShowConnectedStatusTemporarily] = useState<boolean>(false);
    const openTimerIdRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const [availableTools, setAvailableTools] = useState<string[]>([]);
    const [isLoadingTools, setIsLoadingTools] = useState<boolean>(true);
    const [toolsError, setToolsError] = useState<string | null>(null);

    const typingMessageIdRef = useRef<string | null>(null);

    // Effect for fetching agent tools
    useEffect(() => {
        const getTools = async () => {
            setIsLoadingTools(true);
            setToolsError(null);
            try {
                const tools = await fetchAgentTools();
                setAvailableTools(tools);
            } catch (error) {
                console.error('Failed to fetch tools:', error);
                setToolsError(string.leftPanel.noToolAvailableMessage);
                setAvailableTools([]);
                setSelectedTool('');
            } finally {
                setIsLoadingTools(false);
            }
        };

        getTools();

        return () => {
        }
    }, []);

    // Effect for WebSocket connection management
    useEffect(() => {
        websocketService.connect();

        const clearConnectedStatusTimer = () => {
            if (openTimerIdRef.current) {
                clearTimeout(openTimerIdRef.current);
                openTimerIdRef.current = null;
            }
            setShowConnectedStatusTemporarily(false);
        };

        const unsubscribeOpen = websocketService.on('open', () => {
            setConnectionStatus(string.leftPanel.connectionStatus.connected);
            setShowConnectedStatusTemporarily(true);
            openTimerIdRef.current = setTimeout(() => {
                setShowConnectedStatusTemporarily(false);
            }, 3000);
        });

        const unsubscribeConnecting = websocketService.on('connecting', () => {
            setConnectionStatus(string.leftPanel.connectionStatus.connecting);
            clearConnectedStatusTimer();
        });

        const unsubscribeReconnecting = websocketService.on('reconnecting', (delay) => {
            const delayInSeconds = typeof delay === 'number' ? Math.ceil(delay / 1000) : '...';
            setConnectionStatus(string.leftPanel.connectionStatus.reconnecting.replace('{0}', delayInSeconds.toString()));
            clearConnectedStatusTimer();
        });

        const unsubscribeError = websocketService.on('error', (error) => {
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            if (errorMsg === string.leftPanel.alertMessages.maxReconnectAttemptsReached) {
                setConnectionStatus(string.leftPanel.alertMessages.maxReconnectAttemptsReached);
            } else {
                setConnectionStatus(string.leftPanel.connectionStatus.connectionError);
            }
            clearConnectedStatusTimer();
            console.error("WebSocket Error from Service.", error);
        });

        const unsubscribeClose = websocketService.on('close', (event) => {
            setConnectionStatus(string.leftPanel.connectionStatus.closed);
            clearConnectedStatusTimer();
            console.log("WebSocket closed from Service.", event);
        });

        const unsubscribeDisconnected = websocketService.on('disconnected', () => {
            setConnectionStatus(string.leftPanel.connectionStatus.disconnected);
            clearConnectedStatusTimer();
        });

        return () => {
            unsubscribeOpen();
            unsubscribeError();
            unsubscribeClose();
            unsubscribeConnecting();
            unsubscribeReconnecting();
            unsubscribeDisconnected();
            if (openTimerIdRef.current) {
                clearTimeout(openTimerIdRef.current);
            }
        };
    }, []); // Empty dependency array ensures this runs once on mount and cleans up on unmount

    // Effect for handling incoming WebSocket messages
    useEffect(() => {
        const unsubscribeMessage = websocketService.onMessage((data: IncomingWebSocketMessage) => {
            console.log("Enter message via Service (typed):", data);

            if (typingMessageIdRef.current && data.query_id === typingMessageIdRef.current) {
                setMessages(prevMessages => {
                    const updatedMessages = prevMessages.filter(msg => msg.id !== typingMessageIdRef.current);
                    return updatedMessages;
                });
                typingMessageIdRef.current = null;
            } else if (typingMessageIdRef.current && data.query_id !== typingMessageIdRef.current) {
                console.warn("Received message for a different query while typing indicator is active.");
                setMessages(prevMessages => {
                    const updatedMessages = prevMessages.filter(msg => msg.id !== typingMessageIdRef.current);
                    return updatedMessages;
                });
                typingMessageIdRef.current = null; // Reset ref
            }

            if (data.chat_message && data.chat_message.from && data.chat_message.text !== undefined) {
                const newMessage: ChatMessage = {
                    id: data.query_id || uuidv4(),
                    sender: data.chat_message.from === 'user' ? 'user' : 'agent',
                    text: data.chat_message.text,
                    timestamp: new Date(),
                    artifactLink: data.artifact_data
                        ? {
                            displayText: getArtifactDisplayText(data.artifact_data),
                            data: data.artifact_data,
                        }
                        : undefined,
                };

                setMessages((prevMessages) => [...prevMessages, newMessage]);

            } else {
                console.warn("Service: Định dạng tin nhắn đến không khớp:", data);
            }

            if (data.artifact_data ) {
                console.log("Processing artifact data:", data.artifact_data);

                setMessages(prevMessages => {
                    const messageIndex = prevMessages.findIndex(msg => msg.id === data.query_id);
                    if (messageIndex !== -1 && data.artifact_data) {
                        const updatedMessages = [...prevMessages];

                        updatedMessages[messageIndex] = {
                            ...updatedMessages[messageIndex],
                            artifactLink: {
                                displayText: getArtifactDisplayText(data.artifact_data),
                                data: data.artifact_data,
                            },
                        };
                        console.log("Updated message with artifact link:", updatedMessages[messageIndex]);
                        return updatedMessages;
                    }
                    return prevMessages;
                });
            } else if (Object.prototype.hasOwnProperty.call(data,'artifact_data') && data.artifact_data === null) {
                console.log("Clearing artifact data as artifact_data is null.");
                setDisplayedArtifactData(null);
            } else {
                console.log("Received message without artifact_data or with non-array artifact_data (but not null).");
            }
        });

        return () => {
            unsubscribeMessage();
        };
    }, [setDisplayedArtifactData, setMessages]);


    const handleToolChange = (tool: string) => {
        setSelectedTool(tool);
        console.log("Agent tool selected:", tool);
    };

    const handleArtifactLinkClick = (data: ArtifactDataPayload | null) => {
        setDisplayedArtifactData(data);
        console.log("Artifact link clicked, data:", data);
    };

    const handleSendMessage = () => {
        const queryText = inputText.trim();
        if (queryText !== '') {
            const uuid = uuidv4()
            const newUserMessage: ChatMessage = {
                id: uuid,
                sender: 'user',
                text: queryText,
                timestamp: new Date(),
            };

            const typingIndicatorMessage: ChatMessage = {
                id: uuid,
                sender: 'agent',
                text: '',
                timestamp: new Date(),
                isTypingIndicator: true,
            };

            setMessages((prevMessages) => [...prevMessages, newUserMessage, typingIndicatorMessage]);

            typingMessageIdRef.current = uuid;

            const outgoingMessage: OutgoingQueryMessage = {
                query: queryText,
                query_id: uuid,
                selected_tools: selectedTool ? [selectedTool] : [],
            };
            websocketService.sendMessage(outgoingMessage);

            setInputText('');
        }
    };

    const handleInputKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSendMessage();
        }
    };

    return (
        <Box className="chat-panel">
            {/* Connection Status Display - Could be extracted to a component */}
            {connectionStatus !== string.leftPanel.connectionStatus.connected && (
                <Alert
                    severity={
                        connectionStatus === string.leftPanel.connectionStatus.connectionError ||
                        connectionStatus === string.leftPanel.connectionStatus.disconnected ||
                        connectionStatus === string.leftPanel.alertMessages.maxReconnectAttemptsReached
                            ? 'error'
                            : 'warning'
                    }
                    sx={{ mb: 2 }}
                >
                    {connectionStatus === string.leftPanel.connectionStatus.connecting && string.leftPanel.connectionStatus.connecting}
                    {connectionStatus.startsWith(string.leftPanel.connectionStatus.reconnecting.substring(0, string.leftPanel.connectionStatus.reconnecting.indexOf('{0}')))
                        ? connectionStatus
                        : (
                            connectionStatus === string.leftPanel.connectionStatus.connectionError && string.leftPanel.alertMessages.connectionError ||
                            connectionStatus === string.leftPanel.alertMessages.maxReconnectAttemptsReached && string.leftPanel.alertMessages.maxReconnectAttemptsReached ||
                            connectionStatus === string.leftPanel.connectionStatus.closed && string.leftPanel.alertMessages.connectionClosed ||
                            connectionStatus === string.leftPanel.connectionStatus.disconnected && string.leftPanel.alertMessages.disconnecting
                        )
                    }
                </Alert>
            )}

            {(showConnectedStatusTemporarily || connectionStatus !== string.leftPanel.connectionStatus.connected) && (
                <Typography variant="body2" sx={{
                    textAlign: 'center',
                    mt: connectionStatus !== string.leftPanel.connectionStatus.connected ? 0 : 1,
                    color:
                        showConnectedStatusTemporarily ? 'success.main' :
                            connectionStatus === string.leftPanel.connectionStatus.connectionError ||
                            connectionStatus === string.leftPanel.alertMessages.maxReconnectAttemptsReached ||
                            connectionStatus === string.leftPanel.alertMessages.connectionNotOpen ? 'error.main' :
                                connectionStatus === string.leftPanel.connectionStatus.closed ||
                                connectionStatus === string.leftPanel.connectionStatus.disconnected ? 'warning.main' :
                                    'info.main'
                }}>
                    {string.leftPanel.connectionStatus.statusLabel} {
                    showConnectedStatusTemporarily
                        ? string.leftPanel.connectionStatus.connected
                        : connectionStatus
                }
                </Typography>
            )}

            <Typography variant="h2" gutterBottom>{string.leftPanel.title}</Typography>

            {/* Agent Tool Selector Section */}
            {isLoadingTools ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                    <CircularProgress size={24} />
                    <Typography variant="body2" sx={{ ml: 1 }}>Loading tools...</Typography>
                </Box>
            ) : toolsError ? (
                <Alert severity="error" sx={{ my: 2 }}>{toolsError}</Alert>
            ): (
                <AgentToolSelector
                    selectedTool={selectedTool}
                    onToolChange={handleToolChange}
                    agentTools={availableTools}
                    isLoadingTools={isLoadingTools}
                    toolsError={toolsError}
                />
            )}

            {/* Chat Message Display Section */}
            <ChatMessageDisplay messages={messages} onArtifactLinkClick={handleArtifactLinkClick} />

            {/* Chat Input Area - Could be extracted to a component */}
            <Box className="chat-input-area" sx={{
                display: 'flex',
                gap: '10px',
                alignItems: 'center',
                mt: 'auto',
            }}>
                <TextField
                    fullWidth
                    variant="outlined"
                    size="small"
                    placeholder={string.leftPanel.chatInputPlaceholder}
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={handleInputKeyPress}
                />
                <Button
                    variant="contained"
                    color="primary"
                    onClick={handleSendMessage}
                    size="medium"
                    sx={{ px: 3 }}
                    startIcon={<SendIcon />}
                >
                    {string.leftPanel.sendMessageButton}
                </Button>
            </Box>
        </Box>
    );
};

export default ChatPanel;