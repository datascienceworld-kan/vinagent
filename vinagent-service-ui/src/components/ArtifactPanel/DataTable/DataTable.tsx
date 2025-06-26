import React, {useState} from 'react';
import type {TableArtifactData} from "../../../types/types.ts";
import {
    Box,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TablePagination,
    TableRow,
    Typography,
    Button,
    Menu,
    MenuItem
} from '@mui/material';
import string from "../../../config/string.ts";
import * as XLSX from 'xlsx';
import { saveAs } from 'file-saver';

interface DataTableProps {
    data: TableArtifactData;
}

const DataTable: React.FC<DataTableProps> = ({data}) => {

    // State for pagination
    const [page, setPage] = useState(0);

    const [rowsPerPage, setRowsPerPage] = useState(10);
    const title = string.rightPanel.dataTable.title;

    // State for download menu
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
    const open = Boolean(anchorEl);

    const handleClickDownload = (event: React.MouseEvent<HTMLButtonElement>) => {
        setAnchorEl(event.currentTarget);
    };

    const handleCloseDownloadMenu = () => {
        setAnchorEl(null);
    };

    if (!data || !data.columns || !data.rows || data.rows.length === 0) return (
        <Box sx={{textAlign: 'center', mt: 2}}>
            {title && (<Typography variant="body2" color="text.secondary">
                    {string.rightPanel.dataTable.noDataAvailableMessage}
                </Typography>
            )}

        </Box>
    );

    const paginatedRows = data.rows.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);

    const handleChangePage = (_event: unknown, newPage: number) => {
        setPage(newPage);
    };

    const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
        setRowsPerPage(parseInt(event.target.value, 10));
        setPage(0); // Reset page to 0 when rows per page changes
    };

    const handleDownloadCSV = () => {
        if (!data || !data.columns || !data.rows) return;
        const header = data.columns.join(',');
        const rows = data.rows.map(row =>
            row.map(cell => {
                if (cell === null || cell === undefined) return '';
                const cellString = cell.toString();
                if (cellString.includes(',') || cellString.includes('"') || cellString.includes('\n')) {
                    return `"${cellString.replace(/"/g, '""')}"`;
                }
                return cellString;
            }).join(',')
        ).join('\n');

        const csvContent = `${header}\n${rows}`;
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);

        link.setAttribute('href', url);
        link.setAttribute('download', 'data.csv');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        handleCloseDownloadMenu();
    }

    const handleDownloadExcel = () => {
        if (!data || !data.columns || !data.rows) return;
        const worksheet = XLSX.utils.aoa_to_sheet([data.columns, ...data.rows]);
        const workbook = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(workbook, worksheet, 'Sheet1');


        // Save the Excel file
        const excelBuffer = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' });
        const blob = new Blob([excelBuffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;charset=UTF-8' });
        saveAs(blob, 'data.xlsx');
        handleCloseDownloadMenu();

    }

    return (

        <Paper sx={{width: '100%' ,overflow: 'hidden', mt: 2}}>

            {title && (
                <Typography variant="h6" gutterBottom sx={{padding: '0 20px', mt: 1}}>
                    {title}
                </Typography>
            )}

            <Box sx={{ padding: '0 20px', mb: 2 }}>
                <Button
                    variant="outlined"
                    onClick={handleClickDownload}
                    aria-controls={open ? 'download-menu' : undefined}
                    aria-haspopup="true"
                    aria-expanded={open ? 'true' : undefined}
                >{string.rightPanel.downloadButton}</Button>

                <Menu
                    id="download-menu"
                    anchorEl={anchorEl}
                    open={open}
                    onClose={handleCloseDownloadMenu}
                    MenuListProps={{
                        'aria-labelledby': 'basic-button',
                    }}
                >
                    <MenuItem onClick={handleDownloadCSV}>{string.rightPanel.downloadAsCSV}</MenuItem>
                    <MenuItem onClick={handleDownloadExcel}>{string.rightPanel.downloadAsExcel}</MenuItem>
                </Menu>
            </Box>

            <TableContainer sx={{maxHeight: 625, overflowY: 'auto', mt: 2, width: '100%'}}>
                <Table stickyHeader aria-label="financial data table">
                    <TableHead>
                        <TableRow>
                            {data.columns.map((column, index) => (
                                <TableCell key={index} sx={{
                                    fontWeight: 'bold',
                                    backgroundColor: '#f5f5f5',
                                    borderBottom: '1px solid #e0e0e0'
                                }}>
                                    {column}
                                </TableCell>
                            ))}
                        </TableRow>
                    </TableHead>

                    <TableBody>
                        {paginatedRows.map((row, rowIndex) => (
                            <TableRow key={rowIndex} sx={{
                                '&:last-child td, &:last-child th': {border: 0},
                                '&:hover': {
                                    backgroundColor: '#f9f9f9',
                                },
                            }}>
                                {row.map((cellData, cellIndex) => (
                                    <TableCell key={cellIndex} sx={{
                                        textAlign: typeof cellData === 'number' ? 'right' : 'left',
                                        borderBottom: '1px solid #eeeeee',
                                    }}>
                                        {cellData !== null && cellData !== undefined ? cellData.toString() : ''}
                                    </TableCell>
                                ))}
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>

            <TablePagination
                rowsPerPageOptions={[5, 10, 25]}
                component="div"
                count={data.rows.length}
                rowsPerPage={rowsPerPage}
                page={page}
                onPageChange={handleChangePage}
                onRowsPerPageChange={handleChangeRowsPerPage}
                labelRowsPerPage={string.rightPanel.dataTable.linePerPageLabel}
                labelDisplayedRows={({from, to, count}) => `${from}-${to} / ${count}`}
            />
        </Paper>
    )
};

export default DataTable;