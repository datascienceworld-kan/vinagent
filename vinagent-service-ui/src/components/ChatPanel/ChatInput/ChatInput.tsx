import React, { useState } from 'react';
import { TextField, Button, Box } from '@mui/material';
import './ChatInput.css';
import string from "../../../config/string.ts";

interface ChatInputProps {
    onSendMessage: (text: string) => void;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage }) => {
    const [inputText, setInputText] = useState<string>('');

    const handleSendClick = () => {
        if (inputText.trim() !== '') {
            onSendMessage(inputText.trim());
            setInputText('');
        }
    };

    const handleInputKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            handleSendClick();
        }
    };

    return (
        <Box className="chat-input-area">
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
                onClick={handleSendClick}
            >
                {string.leftPanel.sendMessageButton}
            </Button>
        </Box>
    );
};

export default ChatInput;