import React, {useState, forwardRef, useImperativeHandle, useEffect} from 'react';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select, {type SelectChangeEvent } from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';

import Chip from '@mui/material/Chip';
import CancelIcon from '@mui/icons-material/Cancel';
import string from "../../../../config/string.ts";

export interface SettingConfig {
    description: string;
    selectedSkills: string[];
}

export interface SettingTabRef {
    getConfig: () => SettingConfig;
}

interface SettingTabProps {
    initialDescription?: string | null;
    initialSkills?: string[] | null;
}

const SettingTab= forwardRef<SettingTabRef, SettingTabProps>(({ initialDescription, initialSkills }, ref)  => {
    const [description, setDescription] = useState('');
    const [selectedSkill, setSelectedSkill] = useState<string[]>([]);

    useEffect(() => {
        setDescription(initialDescription || '');
    }, [initialDescription]);

    useEffect(() => {
        setSelectedSkill(initialSkills ? initialSkills.filter((skill): skill is string => typeof skill === 'string' && skill !== null) : []);
    }, [initialSkills]);

    useImperativeHandle(ref, () => ({
        getConfig: () => ({
            description: description,
            selectedSkills: selectedSkill,
        }),
    }));

    const handleDescriptionChange = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        const value = event.target.value;
        setDescription(value);
        // if (onDescriptionChange) {
        //    onDescriptionChange(value);
        // }
    };


    const handleSkill = (event: SelectChangeEvent<string[]>) => {
        const {
            target: { value },
        } = event;
        const valueArray = typeof value === 'string' ? value.split(',') : value;

        const validSkills = valueArray ? valueArray.filter((skill): skill is string => typeof skill === 'string' && skill !== null) : [];

        setSelectedSkill(validSkills);
        // if (onFunctionChange) {
        //    onFunctionChange(valueArray);
        // }
    };

    const handleDeleteSkill = (functionToRemove: string) => {
        const newSelectedFunctions = selectedSkill.filter(
            (func) => func !== functionToRemove
        );
        setSelectedSkill(newSelectedFunctions);
        // if (onFunctionChange) {
        //    onFunctionChange(newSelectedFunctions);
        // }
    };


    const renderSelectedValues = (selected: string[]) => {
        return (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {selected.map((value) => (
                    <Chip key={value} label={value}
                          size="small"
                          onDelete={(event) => {
                              event.stopPropagation();
                              handleDeleteSkill(value)
                          }}
                          deleteIcon={
                              <CancelIcon
                                  onMouseDown={(event) => {
                                      event.stopPropagation();
                                  }}
                              />
                          } />
                ))}
            </Box>
        );
    };

    const skillOptions = initialSkills ? initialSkills.filter((skill): skill is string => typeof skill === 'string' && skill !== null) : [];

    return (
        <Box sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
                Setting
            </Typography>

            <Stack spacing={2}>
                <TextField
                    label={string.header.configuration.descriptionLabel}
                    multiline
                    rows={4}
                    variant="outlined"
                    fullWidth
                    value={description}
                    onChange={handleDescriptionChange}
                />

                <FormControl fullWidth variant="outlined">
                    <InputLabel id="function-select-label">Select Function</InputLabel>
                    <Select
                        labelId="function-select-label"
                        id="function-select"
                        multiple
                        value={selectedSkill}
                        label={string.header.configuration.skillLabel}
                        onChange={handleSkill}
                        renderValue={(selected) => renderSelectedValues(selected as string[])}
                    >
                        {skillOptions.map((option) => (
                            <MenuItem key={option} value={option}>
                                {option}
                                {/* Optional: Thêm Checkbox bên cạnh MenuItem để hiển thị trạng thái check */}
                                {/* <Checkbox checked={selectedFunctions.indexOf(option) > -1} /> */}
                                {/* <ListItemText primary={option} /> */}
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>
            </Stack>
        </Box>
    );
});

export default SettingTab;