import string from "../../../../config/string.ts";
import React, {useImperativeHandle, useMemo, useState, forwardRef, useEffect} from "react";
import {useModelContext} from "./ModelContext.tsx";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import TableContainer from "@mui/material/TableContainer";
import Paper from "@mui/material/Paper";
import Table from "@mui/material/Table";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import TableCell from "@mui/material/TableCell";
import TableBody from "@mui/material/TableBody";
import Radio from "@mui/material/Radio";
import Box from "@mui/material/Box";
import Pagination from "@mui/material/Pagination";

export interface ModelsConfig {
    selectedModelId: string | null;
}

export interface ModelsTabRef {
    getConfig: () => ModelsConfig;
}

interface Model {
    id: string;
    type: string;
    display_name: string;
    organization: string;
    pricing: {
        input?: number | string;
        output?: number | string;
        base?: number | string;
    };
    running: boolean

}

const ModelsTab= forwardRef<ModelsTabRef>((_props, ref) => {

    const [searchTerm, setSearchTerm] = useState('');
    const [currentPage, setCurrentPage] = useState(1);
    const [modelsPerPage] = useState(10);
    const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

    const { models, loading, error } = useModelContext();

    useImperativeHandle(ref, () => ({
        getConfig: () => ({
            selectedModelId: selectedModelId,
        }),
    }));

    useEffect(() => {
        if (!loading && !error && models.length > 0) {
            const runningModel = models.find(model => model.running);
            if (runningModel) {
                setSelectedModelId(runningModel.id);
                console.log("Auto-selected running model:", runningModel.display_name, runningModel.id);
            } else {
                setSelectedModelId(null);
            }
        }
    }, [models, loading, error]);

    const filteredModels = useMemo(() => {
        if (!searchTerm) return models;
        return models.filter(model =>
            model.display_name.toLowerCase().includes(searchTerm.toLowerCase())
        );
    }, [models, searchTerm]);

    const indexOfLastModel = currentPage * modelsPerPage;
    const indexOfFirstModel = indexOfLastModel - modelsPerPage;
    const currentModels = filteredModels.slice(indexOfFirstModel, indexOfLastModel);

    const totalPages = Math.ceil(filteredModels.length / modelsPerPage);

    const handlePageChange = (_event: React.ChangeEvent<unknown>, page: number) => {
        setCurrentPage(page);
    };

    const handleRadioChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setSelectedModelId(event.target.value);
        // TODO: Lưu model đã chọn vào state hoặc context/redux nếu cần sử dụng ở nơi khác
        console.log("Selected Model ID:", event.target.value);
    };


    const formatNumber = (value: number | string | undefined): string => {
        if (value === undefined || value === null) return '';
        const num = typeof value === 'string' ? parseFloat(value) : value;
        return parseFloat(num.toFixed(2)).toString();
    };

    const renderPricing = (pricing: Model['pricing']) => {
        if (!pricing) return 'N/A';

        const input = pricing.input !== undefined ? parseFloat(pricing.input.toString()) : null;
        const output = pricing.output !== undefined ? parseFloat(pricing.output.toString()) : null;
        const base = pricing.base !== undefined ? parseFloat(pricing.base.toString()) : null;

        const isValidInput = input !== null && !isNaN(input);
        const isValidOutput = output !== null && !isNaN(output);
        const isValidBase = base !== null && !isNaN(base);


        if (isValidInput && isValidOutput) {
            if (input === 0 && output === 0) return 'Free';
            if (output === 0 && input > 0) return formatNumber(input);
            if (input === 0 && output > 0) return formatNumber(output);
            return `${formatNumber(input)} / ${formatNumber(output)}`;
        }

        if (isValidInput) return formatNumber(input);
        if (isValidOutput) return formatNumber(output);
        if (isValidBase && base !== 0) return formatNumber(base);


        return 'N/A';
    };

    return (
        <Box sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
                {string.header.configuration.tab1Label}
            </Typography>

            <TextField
                label={string.header.configuration.findByDisplayNameLabel}
                variant="outlined"
                size="small"
                fullWidth
                value={searchTerm}
                onChange={(e) => {
                    setSearchTerm(e.target.value);
                    setCurrentPage(1);
                }}
                sx={{ mb: 2 }}
            />

            {loading && <Typography align="center">{string.header.configuration.loading}</Typography>}
            {error && <Typography color="error" align="center">{error}</Typography>}

            {!loading && !error && (
                <>
                    <TableContainer component={Paper} sx={{ maxHeight: 350, mb: 2 }}>
                        <Table stickyHeader size="small" aria-label="models table">
                            <TableHead>
                                <TableRow>
                                    <TableCell sx={{ width: '60px', minWidth: '60px', py: 1, px: 2 }}>
                                        {string.header.configuration.select}
                                    </TableCell>
                                    <TableCell sx={{ minWidth: '150px', py: 1, px: 2 }}>
                                        {string.header.configuration.modelName}
                                    </TableCell>
                                    <TableCell sx={{ width: '120px', minWidth: '120px', py: 1, px: 2 }}>
                                        {string.header.configuration.organLabel}
                                    </TableCell>
                                    <TableCell sx={{ width: '80px', minWidth: '80px', py: 1, px: 2 }}>
                                        {string.header.configuration.typeLabel}
                                    </TableCell>
                                    <TableCell sx={{ width: '150px', minWidth: '150px', py: 1, px: 2 }}>
                                        {string.header.configuration.pricingLabel}
                                    </TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {currentModels.length > 0 ? (
                                    currentModels.map(model => (
                                        <TableRow key={model.id} hover>
                                            <TableCell sx={{ width: '60px', minWidth: '60px', py: 1, px: 2 }}>
                                                <Radio
                                                    value={model.id}
                                                    checked={selectedModelId === model.id}
                                                    onChange={handleRadioChange}
                                                    size="small"
                                                />
                                            </TableCell>
                                            <TableCell sx={{ minWidth: '150px', py: 1, px: 2 }}>
                                                {model.display_name}
                                            </TableCell>
                                            <TableCell sx={{ width: '120px', minWidth: '120px', py: 1, px: 2 }}>
                                                {model.organization}
                                            </TableCell>
                                            <TableCell sx={{ width: '80px', minWidth: '80px', py: 1, px: 2 }}>
                                                {model.type}
                                            </TableCell>
                                            <TableCell sx={{ width: '150px', minWidth: '150px', py: 1, px: 2 }}>
                                                {renderPricing(model.pricing)}
                                            </TableCell>
                                        </TableRow>
                                    ))
                                ) : (
                                    <TableRow>
                                        <TableCell colSpan={5} align="center" sx={{ py: 1, px: 2 }}>
                                            {string.header.configuration.noModelLabel}
                                        </TableCell>
                                    </TableRow>
                                )}
                            </TableBody>
                        </Table>
                    </TableContainer>



                    {totalPages > 1 && (
                        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
                            <Pagination
                                count={totalPages}
                                page={currentPage}
                                onChange={handlePageChange}
                                color="primary"
                                showFirstButton
                                showLastButton
                            />
                        </Box>
                    )}
                </>
            )}
        </Box>
    );
});

export default ModelsTab;