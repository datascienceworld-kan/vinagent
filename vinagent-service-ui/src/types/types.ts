
export interface TableArtifactData {
    type: 'table';
    columns: string[];
    rows: (string | number | boolean | null)[][];
}

export interface PlotlyArtifactData {
    type: "plotly";
    content: string;
}

export interface MarkdownArtifactData {
    type: "markdown";
    content: string;
}

export type ArtifactDataPayload = TableArtifactData | PlotlyArtifactData | MarkdownArtifactData;


export interface ArtifactLinkData {
    displayText: string;
    data: ArtifactDataPayload | null;
}


export interface ChatMessage {
    id: string;
    sender: 'user' | 'agent';
    text: string;
    timestamp: Date;
    artifactLink?: {
        displayText: string;
        data: ArtifactDataPayload | null;
    };
    isTypingIndicator?: boolean;
}


export interface OutgoingQueryMessage {
    query: string;
    query_id: string;
    selected_tools: string[];
}

export type OutgoingWebSocketMessage = OutgoingQueryMessage;


export interface IncomingWebSocketMessagePayload {
    query_id: string;
    chat_message?: {
        from: 'user' | 'agent';
        text: string;
    };
    artifact_data?: ArtifactDataPayload | null;
}

export type IncomingWebSocketMessage = IncomingWebSocketMessagePayload;
