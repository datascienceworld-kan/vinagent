import React, { createContext, useContext, useEffect, useState } from 'react';
import {fetchModelsData} from "../../../../services/apiService.ts";

export interface Model {
    id: string;
    type: string;
    display_name: string;
    organization: string;
    pricing: {
        input?: number | string;
        output?: number | string;
        base?: number | string;
    };
    running?: boolean | undefined;
}

interface ModelContextType {
    models: Model[];
    agentDescription: string | null;
    agentSkills: string[] | null;
    toolsPath: string | null;
    isResetTool: boolean | false;
    loading: boolean;
    error: string | null;
    refetch: () => void;
}

const ModelContext = createContext<ModelContextType>({
    models: [],
    agentDescription: null,
    agentSkills: null,
    toolsPath: null,
    isResetTool: false,
    loading: false,
    error: null,
    refetch: () => {},
});

export const useModelContext = () => useContext(ModelContext);

export const ModelProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [models, setModels] = useState<Model[]>([]);
    const [agentDescription, setAgentDescription] = useState<string | null>(null);
    const [toolsPath, setToolsPath] = useState<string | null>(null);
    const [isResetTool, setIsResetTool] = useState<boolean>(false)
    const [agentSkills, setAgentSkills] = useState<string[] | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchModels = async () => {
        setLoading(true);
        setAgentDescription(null);
        setAgentSkills(null);
        setToolsPath(null);
        setIsResetTool(false);
        try {
            const data = await fetchModelsData();
            if (Array.isArray(data.models)) {
                setModels(data.models);
                setAgentDescription(data.agent_description ?? null);
                setAgentSkills(data.agent_skills ?? null);
                setToolsPath(data.tools_path ?? null);
                setIsResetTool(data.is_reset_tools)
                setError(null);
            } else {
                // Log the unexpected data structure if 'models' is not an array
                console.error("API returned unexpected data structure for models array:", data);
                throw new Error('Invalid models array structure from API');
            }
        } catch (err: unknown) {
            let errorMessage = 'Unknown error fetching models';
            if (err instanceof Error) {
                errorMessage = err.message;
            }
            setError(errorMessage);
            setModels([]);
            setAgentDescription(null);
            setAgentSkills(null);
            setToolsPath(null)
            setIsResetTool(false)
            console.error("Error fetching models:", err);
        } finally {
            setLoading(false);
        }

    };

    useEffect(() => {
        fetchModels();
    }, []);

    return (
        <ModelContext.Provider value={{ models, agentDescription, agentSkills, toolsPath, isResetTool ,loading, error, refetch: fetchModels }}>
            {children}
        </ModelContext.Provider>
    );
};
