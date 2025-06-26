import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {Box, Typography, Button, Menu, MenuItem} from '@mui/material';
import type {MarkdownArtifactData} from "../../../types/types.ts";
import string from "../../../config/string.ts";
import { downloadMarkdownConversion } from '../../../services/apiService';

interface MarkdownDisplayProps {
    data: MarkdownArtifactData;
}

const MarkdownDisplay: React.FC<MarkdownDisplayProps> = ({data}) => {

    const title = string.rightPanel.markdownViewer.title

    const content = data.content


    const filename = string.rightPanel.markdownViewer.fileName;

    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

    const open = Boolean(anchorEl);

    // Handlers for the download menu
    const handleClickDownload = (event: React.MouseEvent<HTMLButtonElement>) => {
        setAnchorEl(event.currentTarget);
    };

    const handleCloseDownloadMenu = () => {
        setAnchorEl(null);
    };

    const handleDownloadMD = () => {
        if (!content) return;

        const blob = new Blob([content], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${filename}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        handleCloseDownloadMenu(); // Close menu after initiating download
    };

    const handleDownloadPDF = async () => {
        if (!content) return;
        const filename = 'downloaded_document';
        try {
            // Call the service function for PDF
            const blob = await downloadMarkdownConversion(content, 'pdf');

            // Use the returned blob to create a download link
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${filename}.pdf`; // Set the correct file extension
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

        } catch (error) {
            console.error('Error during PDF download:', error);
            alert('An error occurred while trying to download the PDF file.');
        } finally {
            handleCloseDownloadMenu(); // Close menu after initiating download (success or fail)
        }
    };

    const handleDownloadDOCX = async () => {
        if (!content) return;
        const filename = 'downloaded_document';
        try {
            // Call the service function for DOCX
            const blob = await downloadMarkdownConversion(content, 'docx');

            // Use the returned blob to create a download link
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${filename}.docx`; // Set the correct file extension
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

        } catch (error) {
            console.error('Error during DOCX download:', error);
            alert('An error occurred while trying to download the DOCX file.');
        } finally {
            handleCloseDownloadMenu(); // Close menu after initiating download (success or fail)
        }
    };


    return (
        <Box>
            <Typography variant="h6" gutterBottom sx={{padding: '0 20px', mt: 1, flexShrink: 0}}>
                {title}
            </Typography>

            <Box sx={{ padding: '0 20px', mb: 2 }}>
                <Button
                    variant="outlined"
                    onClick={handleClickDownload}
                    aria-controls={open ? 'download-menu' : undefined}
                    aria-haspopup="true"
                    aria-expanded={open ? 'true' : undefined}
                >{string.rightPanel.markdownViewer.downloadButton}</Button>

                <Menu
                    id="download-menu"
                    anchorEl={anchorEl}
                    open={open}
                    onClose={handleCloseDownloadMenu}
                    MenuListProps={{
                        'aria-labelledby': 'basic-button', // Make sure this matches the button's aria-controls if needed
                    }}
                >
                    {/* Update MenuItems to reflect Markdown formats */}
                    <MenuItem onClick={handleDownloadMD}>
                        {string.rightPanel.markdownViewer.downloadAsMd}
                    </MenuItem>
                    <MenuItem onClick={handleDownloadPDF}>
                        {string.rightPanel.markdownViewer.downloadAsPdf}
                    </MenuItem>
                    <MenuItem onClick={handleDownloadDOCX}>
                        {/* Use string config or a default */}
                        {string.rightPanel.markdownViewer.downloadAsDocx}
                    </MenuItem>
                </Menu>
            </Box>
            <Box sx={{
                width: '100%',
                height: '100%',
                display: 'flex',
                overflow: 'auto',
                flexDirection: 'column',
                maxHeight: 625
            }}>

                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {content}
                </ReactMarkdown>
            </Box>
        </Box>
    );
};

export default MarkdownDisplay;