import React, { useRef } from 'react';
import { Box, Typography, Button } from '@mui/material';
import type { PlotlyArtifactData } from "../../../types/types.ts";
import Plot from "react-plotly.js";
import string from "../../../config/string.ts";
import Plotly from 'plotly.js';

interface PlotlyChartProps {
    data: PlotlyArtifactData;
}

const PlotlyChart: React.FC<PlotlyChartProps> = ({ data }) => {
    const plotData = JSON.parse(data.content);
    const title = string.rightPanel.plotlyChart.title;

    const plotRef = useRef<Plotly.PlotlyHTMLElement | null>(null);

    const handleDownloadPNG = () => {
        if (plotRef.current) {
            Plotly.downloadImage(plotRef.current, {
                format: 'png',
                filename: 'chart',
                width: 800,
                height: 600
            });
        }
    };

    return (
        <Box sx={{
            width: '100%',
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
        }}>
            <Typography variant="h6" gutterBottom sx={{ padding: '0 20px', mt: 1, flexShrink: 0 }}>
                {title}
            </Typography>

            <Box sx={{ padding: '0 20px', mb: 2, flexShrink: 0 }}>
                <Button variant="outlined" onClick={handleDownloadPNG}>
                    Download Chart as PNG
                </Button>
            </Box>

            <Box sx={{
                flexGrow: 1,
                overflowY: 'auto',
                width: '100%',
            }}>
                <Plot
                    data={plotData.data}
                    layout={plotData.layout}
                    frames={plotData.frames}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: '100%', height: 'auto' ,maxHeight: 625}}
                    onInitialized={(_, graphDiv) => {
                        plotRef.current = graphDiv as Plotly.PlotlyHTMLElement;
                    }}
                    onUpdate={(_, graphDiv) => {
                        plotRef.current = graphDiv as Plotly.PlotlyHTMLElement;
                    }}
                />
            </Box>
        </Box>
    );
};

export default PlotlyChart;
