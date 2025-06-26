// src/App.tsx
import React, {useState} from 'react';
import './App.css';
import Header from "./components/Header/Header.tsx";
import ChatPanel from "./components/ChatPanel/ChatPanel.tsx";
import ArtifactPanel from "./components/ArtifactPanel/ArtifactPanel.tsx";
import Footer from "./components/Footer/Footer.tsx";
import {ThemeProvider} from '@mui/material/styles';
import financeTheme from "./themes/theme.ts";
import type {ArtifactDataPayload} from "./types/types.ts";
import {ModelProvider} from "./components/Header/SettingPopup/ModelTab/ModelContext.tsx";

const App: React.FC = () => {

    const [displayedArtifactData, setDisplayedArtifactData] = useState<ArtifactDataPayload | null>(null);

    return (
        <ThemeProvider theme={financeTheme}>
            <ModelProvider>
                <div className="App">
                    <Header/>
                    <ChatPanel setDisplayedArtifactData={setDisplayedArtifactData}/>
                    <ArtifactPanel data={displayedArtifactData}/>
                    <Footer/>
                </div>
            </ModelProvider>
        </ThemeProvider>
    );
};

export default App;