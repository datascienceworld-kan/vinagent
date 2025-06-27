import React, { useState, useRef } from 'react';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';

import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Box from '@mui/material/Box';

import string from "../../../config/string.ts";
import ModelsTab, {type ModelsTabRef} from "./ModelTab/ModelsTab.tsx";
import SettingTab, {type SettingTabRef} from "./SettingTab/SettingTab.tsx";
import {Alert, CircularProgress, Snackbar} from "@mui/material";
import {saveConfiguration} from "../../../services/apiService.ts";
import {useModelContext} from "./ModelTab/ModelContext.tsx";


interface SettingsPopupProps {
    onClose: () => void;
    // onSaveSuccess?: () => void;
    // onSaveError?: (error: string) => void;
}


const SettingsPopup: React.FC<SettingsPopupProps> = ({ onClose}) => {
    const [activeTab, setActiveTab] = useState(0);
    const [isSaving, setIsSaving] = useState(false);
    const [saveStatus, setSaveStatus] = useState<{ open: boolean, message: string, severity: 'success' | 'error' | 'info' }>({
        open: false,
        message: '',
        severity: 'info',
    });

    const {agentDescription, agentSkills, toolsPath, isResetTool ,refetch } = useModelContext();

    const modelsTabRef = useRef<ModelsTabRef>(null);
    const settingTabRef = useRef<SettingTabRef>(null);
    // const [models, setModels] = useState<Model[]>([]);
    // const [loading, setLoading] = useState(true);
    // const [error, setError] = useState<string | null>(null);


    const handleTabChange = (_event: React.ChangeEvent<object>, newValue: number) => {
        setActiveTab(newValue);
    };

    const handleSave = async () => {
        if (!modelsTabRef.current || !settingTabRef.current) {
            console.error("Refs to tab components are not available.");
            setSaveStatus({ open: true, message: 'Internal error: Configuration components not ready.', severity: 'error' });
            return;
        }

        const modelsConfig = modelsTabRef.current.getConfig();
        const settingConfig = settingTabRef.current.getConfig();
        const dataToSave = {
            model_id: modelsConfig.selectedModelId,
            description: settingConfig.description,
            skills: settingConfig.selectedSkills,
        };
        console.log("Data to save:", dataToSave);


        setIsSaving(true);

        try {
            await saveConfiguration(dataToSave);

            console.log("Configuration saved successfully!");
            setSaveStatus({ open: true, message: 'Configuration saved successfully!', severity: 'success' });
            refetch();
        } catch (err: unknown) {
            let errorMessage = "Failed to save configuration.";
            if (err instanceof Error) {
                errorMessage = err.message;
            } else if (typeof err === 'string') {
                errorMessage = err;
            }
            console.error("Error saving configuration:", err);
            setSaveStatus({ open: true, message: errorMessage, severity: 'error' });

            // }
        } finally {
            setIsSaving(false);
        }
    };

    const handleCloseSnackbar = (_event?: React.SyntheticEvent | Event, reason?: string) => {
        if (reason === 'clickaway') {
            return;
        }
        setSaveStatus({ ...saveStatus, open: false });
    };

    return (
        <Dialog open={true} onClose={onClose} fullWidth maxWidth="md">
            <DialogTitle>
                {string.header.configuration.title}
                <IconButton
                    aria-label="close"
                    onClick={onClose}
                    sx={{
                        position: 'absolute',
                        right: 8,
                        top: 8,
                        color: (theme) => theme.palette.grey[500],
                    }}
                >
                    <CloseIcon />
                </IconButton>
            </DialogTitle>

            <DialogContent dividers sx={{ p: 0 }}>
                <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                    <Tabs value={activeTab} onChange={handleTabChange} aria-label="settings tabs">
                        <Tab label={string.header.configuration.tab1Label} />
                        <Tab label={string.header.configuration.tab2Label} />
                    </Tabs>
                </Box>

                <Box sx={{ display: activeTab === 0 ? 'block' : 'none' }}>
                    <ModelsTab ref={modelsTabRef} />
                </Box>

                <Box sx={{ display: activeTab === 1 ? 'block' : 'none' }}>
                    <SettingTab ref={settingTabRef}
                                initialDescription={agentDescription}
                                initialSkills={agentSkills}
                                initialToolPath={toolsPath}
                                initialResetTool={isResetTool}
                                />
                </Box>

            </DialogContent>

            <DialogActions>
                <Button
                    onClick={handleSave}
                    color="primary"
                    disabled={isSaving}
                    startIcon={isSaving ? <CircularProgress size={20} /> : null}
                >
                    {isSaving ? 'Saving...' : 'Save'}
                </Button>
                <Button onClick={onClose} color="secondary">
                    {string.header.configuration.closeButtonLabel}
                </Button>
            </DialogActions>

            <Snackbar open={saveStatus.open} autoHideDuration={6000} onClose={handleCloseSnackbar}>
                <Alert onClose={handleCloseSnackbar} severity={saveStatus.severity} sx={{ width: '100%' }}>
                    {saveStatus.message}
                </Alert>
            </Snackbar>
        </Dialog>
    );
};

export default SettingsPopup;