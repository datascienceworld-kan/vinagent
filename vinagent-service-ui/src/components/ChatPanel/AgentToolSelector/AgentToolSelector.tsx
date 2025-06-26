import React, {useState} from 'react';
import './AgentToolSelector.css';
import {
    Box,
    FormControl, InputAdornment,
    InputLabel,
    MenuItem,
    Select,
    type SelectChangeEvent,
    TextField,
    Typography
} from "@mui/material";
import string from "../../../config/string.ts";
import SearchIcon from '@mui/icons-material/Search';

interface AgentToolSelectorProps {
    selectedTool: string;
    onToolChange: (tool: string) => void;
    agentTools: string[];
    isLoadingTools: boolean;
    toolsError: string | null;
}

const AgentToolSelector: React.FC<AgentToolSelectorProps> = ({
                                                                 selectedTool,
                                                                 onToolChange,
                                                                 agentTools,
                                                                 isLoadingTools,
                                                                 toolsError
                                                             }) => {

    const [searchQuery, setSearchQuery] = useState('');
    const [open, setOpen] = useState(false);

    const handleChange = (event: SelectChangeEvent) => {
        onToolChange(event.target.value as string);
        setOpen(false);
        setSearchQuery('');
    };

    const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setSearchQuery(event.target.value);
        setOpen(true);
    };

    const handleSearchFocus = () => {
        // setOpen(true);
    };

    const isSelectDisabled = isLoadingTools || toolsError !== null || agentTools.length === 0;

    const filteredTools = agentTools.filter(technicalName => {
        const friendlyName = string.toolNameMapping[technicalName] || technicalName;
        return technicalName.toLowerCase().includes(searchQuery.toLowerCase()) ||
            friendlyName.toLowerCase().includes(searchQuery.toLowerCase());
    });


    return (
        <Box sx={{mb: 2, width: '100%', flexShrink: 0}}>

            <TextField
                fullWidth
                size="small"
                placeholder="Search Agent Tools..."
                value={searchQuery}
                onChange={handleSearchChange}
                onFocus={handleSearchFocus}
                InputProps={{
                    startAdornment: (
                        <InputAdornment position="start">
                            <SearchIcon/>
                        </InputAdornment>
                    ),
                }}
                sx={{mb: 1}}
                disabled={isLoadingTools || toolsError !== null}
            />

            <FormControl fullWidth className="agent-tool-selector-container" disabled={isSelectDisabled}>
                <InputLabel id="agent-tool-select-label">{string.leftPanel.toolSelectLabel}</InputLabel>
                <Select
                    labelId="agent-tool-select"
                    id="agent-tool-select"
                    value={selectedTool}
                    label={string.leftPanel.toolSelectLabel}
                    onChange={handleChange}
                    open={open}
                    onClose={() => setOpen(false)}
                    onOpen={() => setOpen(true)}


                    MenuProps={{
                        PaperProps: {
                            sx: {
                                maxHeight: 250,
                                width: '25%',
                                overflowY: 'auto',
                            },
                        },
                    }}
                >

                    {filteredTools.map((technicalName) => (
                        <MenuItem key={technicalName} value={technicalName}>
                            {string.toolNameMapping[technicalName] || technicalName}
                        </MenuItem>
                    ))}

                    {searchQuery && filteredTools.length === 0 && (
                        <MenuItem disabled>
                            <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', width: '100%' }}>
                                No tools found matching "{searchQuery}"
                            </Typography>
                        </MenuItem>
                    )}

                    {/*{agentTools.map((technicalName) => (*/}
                    {/*    <MenuItem key={technicalName} value={technicalName}>*/}
                    {/*        {string.toolNameMapping[technicalName] || technicalName}*/}
                    {/*    </MenuItem>*/}
                    {/*))}*/}

                    {!isLoadingTools && toolsError === null && agentTools.length === 0 && !searchQuery && (
                        <MenuItem disabled>
                            <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', width: '100%' }}>
                                {string.leftPanel.noToolAvailableMessage}
                            </Typography>
                        </MenuItem>
                    )}
                </Select>
            </FormControl>
            {/*{!isLoadingTools && !toolsError && (*/}
            {/*    <Typography variant="body2" color="text.secondary" sx={{mt: 1}}>*/}
            {/*        {string.leftPanel.noToolAvailableMessage}*/}
            {/*    </Typography>*/}
            {/*)}*/}
        </Box>
    )
}

export default AgentToolSelector;