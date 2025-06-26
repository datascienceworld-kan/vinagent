import { createTheme } from '@mui/material/styles';

const financeTheme = createTheme({
    palette: {
        primary: {
            main: '#004d40',
            light: '#39796b',
            dark: '#00251a',
            contrastText: '#ffffff'
        },

        secondary: {
            main: '#4db6ac',
            light: '#82e9de',
            dark: '#00867d',
            contrastText: '#ffffff'
        },

        error: {
            main: '#d32f2f'
        },

        warning: {
            main: '#ed6c02'
        },

        info: {
            main: '#0288d1'
        },

        success: {
            main: '#2e7d32'
        },

        text: {
            primary: '#212121',
            secondary: '#757575',
            disabled: '#bdbdbd'
        },

        background: {
            default: '#f5f5f5',
            paper: '#ffffff'
        },

        divider: '#e0e0e0',
    },
    typography: {
        fontFamily: 'Roboto, "Helvetica Neue", Arial, sans-serif',

        h1: {
            fontSize: '2.5rem',
            fontWeight: 700
        },

        h2: {
            fontSize: '2rem',
            fontWeight: 600
        },

        body1: {
            fontSize: '1rem',
            lineHeight: 1.6,
        },

        button: {
            textTransform: 'none',
            fontWeight: 500,
        },
    },

    components: {
        MuiButton: {
            defaultProps: {
                disableElevation: true
            },
            styleOverrides: {
                root: {
                    borderRadius: 4
                },
            },
        },

        MuiPaper: {
            defaultProps: {
                elevation: 2,
            },
            styleOverrides: {
                root: {
                    borderRadius: 0
                },
            },
        },

        MuiTextField: {
            styleOverrides: {
                root: {
                    borderRadius: 4
                },
            },
        },

        MuiSelect: {
            styleOverrides: {
                root: {
                    borderRadius: 4
                },
            },
        },

        MuiList: {
            styleOverrides: {
                root: {
                    padding: 0
                },
            },
        },

        MuiMenuItem: {
            styleOverrides: {
                root: {
                    padding: '10px 16px'
                },
            },
        },

        MuiTableCell: {
            styleOverrides: {
                root: {
                    padding: '10px 16px',
                    borderBottom: '1px solid #e0e0e0',
                },
            },
        },

        MuiTableHead: {
            styleOverrides: {
                root: {
                    backgroundColor: '#e0e0e0'
                },
            },
        }
    }
});

export default financeTheme;