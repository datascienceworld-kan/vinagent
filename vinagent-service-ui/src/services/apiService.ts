const API_BASE_URL = 'http://localhost:8888/api';

interface ConfigurationData {
    model_id: string | null;
    description: string;
    skills: string[];
}

interface ModelsApiResponse {
    models: Array<{
        id: string;
        type: string;
        display_name: string;
        organization: string;
        pricing: {
            input?: number | string;
            output?: number | string;
            base?: number | string;
        };
        running?: boolean;
    }>;
    agent_description?: string | null;
    agent_skills?: string[] | null;
    tools_path?: string | null;
    is_reset_tools: boolean;
}

export const fetchAgentTools = async (): Promise<string[]> => {
    try {
        console.log(`Workspaceing agent tools from ${API_BASE_URL}/tools`);
        const response = await fetch(`${API_BASE_URL}/tools`);

        if (!response.ok) {
            console.error(`API Error fetching tools: ${response.status} ${response.statusText}`);
            return [];
        }

        const tools: string[] = await response.json();
        console.log("Successfully fetched agent tools:", tools);
        return tools;
    } catch (error) {
        console.error('Error fetching agent tools:', error);
        return [];
    }
};

export const downloadMarkdownConversion = async (
    markdownContent: string,
    format: 'pdf' | 'docx'
): Promise<Blob> => {
    try {
        console.log(`Requesting markdown conversion to ${format} from ${API_BASE_URL}/convert-markdown`);
        const response = await fetch(`${API_BASE_URL}/convert-markdown`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ markdown: markdownContent, format }),
        });

        if (!response.ok) {
            // Try to read error details from the response
            const errorDetails = await response.json().catch(() => null); // Attempt to parse JSON, fallback to null
            const errorMsg = errorDetails ? errorDetails.detail : response.statusText;
            console.error(`API Error converting markdown: ${response.status} ${errorMsg}`);
            throw new Error(`Server responded with status ${response.status}: ${errorMsg}`);
        }

        // The response body is the file content as a blob
        const blob = await response.blob();
        return blob;

    } catch (error) {
        console.error('Error calling markdown conversion API:', error);
        throw error; // Re-throw the error to be caught by the caller
    }
};

export const saveConfiguration = async (config: ConfigurationData): Promise<void> => {
    const saveEndpoint = `${API_BASE_URL}/save-config`;

    console.log(`Saving configuration to ${saveEndpoint}`, config);

    try {
        const response = await fetch(saveEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(config),
        });

        if (!response.ok) {
            const errorDetails = await response.json().catch(() => null);
            const errorMsg = errorDetails ? errorDetails.detail : response.statusText;
            console.error(`API Error saving configuration: ${response.status} ${errorMsg}`);
            throw new Error(`Server responded with status ${response.status}: ${errorMsg}`);
        }

        console.log("Configuration saved successfully!");
    } catch (error) {
        console.error('Error calling save configuration API:', error);
        throw error;
    }
};

export const fetchModelsData = async (): Promise<ModelsApiResponse> => {
    const endpoint = `${API_BASE_URL}/models`;
    console.log(`Fetching models data from ${endpoint}`);

    try {
        const response = await fetch(endpoint);

        if (!response.ok) {
            const errorDetails = await response.json().catch(() => null);
            const errorMsg = errorDetails ? errorDetails.detail : response.statusText;
            console.error(`API Error fetching models data: ${response.status} ${errorMsg}`);
            throw new Error(`Server responded with status ${response.status}: ${errorMsg}`);
        }

        const data: ModelsApiResponse = await response.json();
        console.log("Successfully fetched models data:", data);
        return data;

    } catch (error) {
        console.error('Error calling fetch models data API:', error);
        throw error;
    }
};