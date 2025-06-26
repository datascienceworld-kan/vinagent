// src/config/string.ts

interface AppStrings {
    header: {
        title: string;
        configuration: {
            title: string;
            tab1Label: string;
            tab2Label: string;
            loading: string;
            select: string;
            modelName: string;
            noModelLabel: string;
            organLabel: string;
            typeLabel: string;
            pricingLabel: string;
            findByDisplayNameLabel: string;
            closeButtonLabel: string;
            skillLabel: string;
            descriptionLabel: string;
        }
    };
    leftPanel: {
        title: string;
        toolSelectLabel: string;
        chatInputPlaceholder: string;
        sendMessageButton: string;
        noToolAvailableMessage: string;
        setToolsError: string
        // initialMessages: {
        //     welcome: string;
        //     dataAnalysisRequest: string;
        //     analysisResultIntro: string;
        //     chartRequestSuggestion: string;
        // };
        // agentResponseTemplate: string;
        senderNames: {
            user: string;
            agent: string;
        };
        artifactLinkDefaultText: string;

        connectionStatus: {
            disconnected: string;
            disconnecting: string;
            connecting: string;
            connected: string;
            connectionError: string;
            closed: string;
            reconnecting: string;
            statusLabel: string;
        };

        alertMessages: {
            connectionError: string;
            connectionClosed: string;
            disconnecting: string;
            maxReconnectAttemptsReached: string;
            connectionNotOpen: string;
        };
    };
    rightPanel: {
        title: string;
        emptyArtifactMessage: string;
        dataTable: {
            noDataAvailableMessage: string;
            linePerPageLabel: string;
            title: string;
        };
        plotlyChart: {
            title: string;
        };
        markdownViewer: {
            title: string;
            fileName: string;
            downloadButton: string;
            downloadAsMd: string;
            downloadAsPdf: string;
            downloadAsDocx: string;
        };
        downloadButton: string;
        downloadAsCSV: string;
        downloadAsExcel: string;
        downloadAsPDF: string;
        downloadAsDoc: string;
        downloadAsPNG: string;
    };

    toolNameMapping: { [key: string]: string }
}

const appStrings: AppStrings = {
    header: {
        title: 'Agent Service - Finance Assistant',
        configuration: {
            title: 'Configuration',
            tab1Label: 'Models',
            tab2Label: 'Setting',
            loading: 'Model is loading...',
            select: 'Select',
            modelName: 'Display Name',
            noModelLabel: 'No model found',
            organLabel: 'Organization',
            typeLabel: 'Type',
            pricingLabel: 'Pricing (per 1M tokens)',
            findByDisplayNameLabel: 'Find by Display Name',
            closeButtonLabel: 'Close',
            skillLabel: 'Select skills',
            descriptionLabel: 'Description'
        }
    },
    leftPanel: {
        title: 'Chat Panel',
        toolSelectLabel: 'Select Agent Tool',
        chatInputPlaceholder: 'Enter your message...',
        sendMessageButton: 'Send',
        noToolAvailableMessage: 'No tools available.',
        setToolsError: 'Failed to load agent tools. Please try again later.',
        artifactLinkDefaultText: "View Data",
        senderNames: {
            user: "You",
            agent: "Agent",
        },
        connectionStatus: {
            disconnected: "Disconnected",
            disconnecting: 'Disconnecting...',
            connecting: "Connecting...",
            connected: "Connected",
            connectionError: "Connection Error",
            closed: "Closed",
            reconnecting: "Reconnecting in {0}s...",
            statusLabel: "Connection Status:",
        },
        alertMessages: {
            connectionError: "WebSocket connection error. Please check the server or network.",
            connectionClosed: "WebSocket connection closed.",
            disconnecting: "WebSocket is disconnecting.",
            maxReconnectAttemptsReached: "Max reconnect attempts reached. Cannot establish connection.",
            connectionNotOpen: "Cannot send message. Connection not open.",
        },
    },
    rightPanel: {
        title: 'Artifact Panel',
        emptyArtifactMessage: "Analysis results or charts will appear here.",
        dataTable: {
            noDataAvailableMessage: "No data available for this table artifact.",
            linePerPageLabel: "Line per page:",
            title: "Data Table"
        },
        plotlyChart: {
            title: "Plotly Chart"
        },
        markdownViewer: {
            title: "Markdown Viewer",
            fileName: "downloaded_file",
            downloadButton: "Download Document",
            downloadAsMd: "Download .md",
            downloadAsPdf: "Download .pdf",
            downloadAsDocx: "Download .docx"

        },
        downloadButton: "Download Data",
        downloadAsCSV: "Download as CSV",
        downloadAsExcel: "Download as Excel",
        downloadAsPDF: "Download as PDF",
        downloadAsDoc: "Download as Word",
        downloadAsPNG: "Download as PNG",
    },

    toolNameMapping: {
        "agentools.tools.deepsearch_opt": "Deep Search Tool",
        "agentools.tools.yfinance_tools": "Financial Data (Yahoo Finance)",
        "agentools.tools.trending_news": "Trending News",
        "agentools.tools.computer_tools": "Computer Tools",
        "agentools.tools.websearch_tools": "Web Search",
        "agentools.tools.terminal_tools": "Terminal Access",
        "agentools.tools.alpha_vantage_tools": "Financial Data (Alpha Vantage)",
    },


}

export default appStrings;