import React, {useRef, useEffect} from 'react';

import {Box, Typography, Paper, type Theme, type SxProps, Button, keyframes, styled} from '@mui/material';

import './ChatMessageDisplay.css';
import type {ArtifactDataPayload, ChatMessage} from "../../../types/types.ts";


// Define keyframes for the dot animation
const blink = keyframes`
  0% { opacity: 0.2; }
  20% { opacity: 1; }
  100% { opacity: 0.2; }
`;

// Styled component for the typing indicator dots
const TypingDot = styled('span')({
    display: 'inline-block',
    width: '6px',
    height: '6px',
    margin: '0 1px',
    borderRadius: '50%',
    backgroundColor: 'text.secondary', // Use theme color
    animation: `${blink} 1.4s infinite both`,
});

// Adjust animation delay for each dot
const TypingDot1 = styled(TypingDot)({
    animationDelay: '0s',
});

const TypingDot2 = styled(TypingDot)({
    animationDelay: '0.2s',
});

const TypingDot3 = styled(TypingDot)({
    animationDelay: '0.4s',
});

interface ChatMessageDisplayProps {
    messages: ChatMessage[];
    sx?: SxProps<Theme>;
    onArtifactLinkClick: (data: ArtifactDataPayload | null) => void;
}

const ChatMessageDisplay: React.FC<ChatMessageDisplayProps> = ({messages, sx, onArtifactLinkClick}) => {
    const messagesEndRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({behavior: 'smooth'});
    }, [messages]);

    const formatTime = (date: Date) => {
        const hours = date.getHours().toString().padStart(2, '0');
        const minutes = date.getMinutes().toString().padStart(2, '0');
        return `${hours}:${minutes}`;
    };

    // Function to render the typing indicator
    const renderTypingIndicator = () => (
        <Box sx={{ display: 'flex', alignItems: 'center', height: '20px' }}> {/* Adjust height as needed */}
            <Typography variant="body2" sx={{ mr: 0.5, fontStyle: 'italic', color: 'text.secondary' }}>Agent is typing</Typography> {/* Optional text */}
            <TypingDot1 />
            <TypingDot2 />
            <TypingDot3 />
        </Box>
    );


    return (
        <Paper elevation={3} className="chat-messages-container" sx={sx}>
            {messages.map((message) => (
                <Box
                    key={message.id}
                    className={`chat-message ${message.sender}`}
                >
                    {message.sender === 'agent' && message.isTypingIndicator ? (
                        renderTypingIndicator()
                    ) : (
                        // Render normal message text
                        <Typography variant="body2" component="div" sx={{ mb: 0.5 }}>{message.text}</Typography>
                    )}

                    {message.artifactLink && (
                        <Button
                            variant="text"
                            size="small"
                            sx={{ mt: 1, textTransform: 'none' }}
                            onClick={() => onArtifactLinkClick(message.artifactLink!.data)}
                        >
                            {message.artifactLink.displayText}
                        </Button>
                    )}

                    <Typography variant="caption" sx={{
                        color: 'text.secondary',
                        fontSize: '0.75rem',
                        display: 'block',
                        textAlign: message.sender === 'user' ? 'right' : 'left',
                    }}>
                        {message.sender === 'user' ? 'You' : 'Agent'} - {formatTime(message.timestamp)}
                    </Typography>
                </Box>
            ))}
            <div ref={messagesEndRef}/>
        </Paper>
    )
};

export default ChatMessageDisplay;
