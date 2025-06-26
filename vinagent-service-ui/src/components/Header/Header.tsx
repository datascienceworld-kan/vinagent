// src/components/Header.tsx
import React, {useState} from 'react';
import './Header.css';
import string from "../../config/string.ts";
import { FiSettings } from 'react-icons/fi';
import SettingsPopup from "./SettingPopup/SettingsPopup.tsx";

const Header: React.FC = () => {

    const [showSettingsPopup, setShowSettingsPopup] = useState(false);

    const toggleSettingsPopup = () => {
        setShowSettingsPopup(!showSettingsPopup);
    };

    return (
        <header className="header">
            <div className="header-content">
                <h1>{string.header.title}</h1>
            </div>
            <div className="settings-icon" onClick={toggleSettingsPopup}>
                <FiSettings size={24} color="white" />
            </div>
            {showSettingsPopup && (
                <SettingsPopup onClose={toggleSettingsPopup} />
            )}
        </header>
    );
};

export default Header;