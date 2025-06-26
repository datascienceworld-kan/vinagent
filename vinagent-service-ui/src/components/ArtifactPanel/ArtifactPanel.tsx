import React from 'react';
import './ArtifactPanel.css';
import type {ArtifactDataPayload, PlotlyArtifactData, TableArtifactData, MarkdownArtifactData} from "../../types/types.ts";
import {Box, Typography} from "@mui/material";
import string from "../../config/string.ts";
import DataObjectIcon from '@mui/icons-material/DataObject';
import DataTable from "./DataTable/DataTable.tsx";
import PlotlyChart from "./Plot/PlotlyChart.tsx";
import MarkdownDisplay from "./Document/MarkdownRender.tsx";


interface ArtifactPanelProps {
    data: ArtifactDataPayload | null;
}

const ArtifactPanel: React.FC<ArtifactPanelProps> = ({data}) => {

    const isDataEmpty = data === null || data === undefined;

    const renderArtifactItem = (item: ArtifactDataPayload) => {

        if (item.type === "table") {
            const tableData = item as TableArtifactData;
            return <DataTable key="artifact-table" data={tableData} />
        } else if (item.type === "plotly") { // Add case for plotly
            const plotlyData = item as PlotlyArtifactData;
            return <PlotlyChart key="artifact-plotly" data={plotlyData} />
        } else if (item.type === "markdown") { // Add case for plotly
            const mdData = item as MarkdownArtifactData;
            return <MarkdownDisplay key="artifact-plotly" data={mdData} />
        }
        return null;
    };

    return (
        <Box className="artifact-panel" >
            <Typography variant="h2" gutterBottom>{string.rightPanel.title}</Typography>


            { isDataEmpty ? (<Box sx={{ textAlign: 'center', color: 'text.secondary', flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                <DataObjectIcon sx={{ fontSize: 60, mb: 2 }} />
                <Typography variant="body1">
                    {string.rightPanel.emptyArtifactMessage}
                </Typography>
            </Box>):(
                <Box sx={{ width: '100%', flexGrow: 1 }}>
                    {renderArtifactItem(data)}
                </Box>
            )}
        </Box>
    );
};

export default ArtifactPanel;