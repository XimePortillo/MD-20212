#!/usr/bin/python3

import PySimpleGUI as sg
import os.path
import pandas as pd               # Para la manipulación de datos y análisis
import numpy as np                # Para crear vectores de datos, matrices de n dimensiones
import matplotlib.pyplot as plt   # Para generar gráficos
import seaborn as sns             # Para visualización de los datos
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#header_list=["Data"]
types, data, nulos, corr_layout=[], [], [], []

def plot_relacion(df):
    fig = plt.figure(figsize=(7,4))
    sns.heatmap(df.corr(), cmap='RdBu_r', annot=True)
    return fig
    
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def new_window():
    #structure_Col = [
                        
    #                ]

    corr_layout =   [
                        [
                            sg.Text("Correlación de los Datos", font=('Helvetica', 15), text_color='#F6CEF5'),
                        ],
                        [
                            sg.Table(values=corr,
                            headings=headers,
                            display_row_numbers=False,
                            auto_size_columns=False,
                            num_rows=min(20, len(headers)),
                            background_color = "#A9E2F3",
                            text_color = "black",
                            header_background_color ="#8181F7",
                            key="-Corr-"),
                            sg.Button("Mapa de calor")
                        ],
                        [
                            sg.Canvas(key='-CORR-', 
                            #size=(100,70),
                            pad=(10,10))
                        ]
                    ]

    tab2_layout =   [
                        [
                            sg.Text("Datos", font=('Helvetica', 15), text_color='#F6CEF5'),
                        ],
                        [
                            sg.Table(values=data,
                            headings=header_list,
                            display_row_numbers=False,
                            auto_size_columns=False,
                            num_rows=min(25, len(data)),
                            alternating_row_color='lightblue',
                            key="-TABLE-")
                        ],
                        [
                            sg.Text("Estructura de los Datos", font=('Helvetica', 15), size=(30,1),text_color='#F6CEF5')
                        ],
                        [
                            sg.Table(values=types,
                            headings=["Dato", "Tipo Dato", "Valores Nulos"],
                            display_row_numbers=False,
                            auto_size_columns=False,
                            num_rows=min(30, len(types)),
                            background_color = "white",
                            text_color = "black",
                            header_background_color = "#F6CEF5",
                            key="-Datos-")
                        ] #,
                        #[sg.Column(structure_Col),
                        #sg.Column(corr_layout)]
                    ]

    tab3_layout =   [
                        #[
                        #    sg.Canvas(key='AtipicData', 
                        #    #size=(100,70),
                        #    pad=(10,10))
                        #],
                        [
                            sg.Text("Valores Atípicos", font=('Helvetica', 15), size=(30,1),text_color='#F6CEF5')
                        ],
                        [
                            sg.Text("Selecciona el Datos", font=(None, 15)),
                            sg.In(size=(30, 1), enable_events=True, key="DatoAtipico"),
                            sg.Button("Graficar")
                        ],
                        [
                            sg.Canvas(key='Atipic', 
                            #size=(100,70),
                            pad=(10,10))
                        ]
                    ]

    tab4_layout =   [
                        [
                            sg.Text("Selecciona un archivo"),
                        ],
                    ]

    tab5_layout =   [
                        [
                            sg.Text("Selecciona un archivo"),
                        ],
                    ]

    statistic_layout =  [
                            [
                            sg.Text("Estadisticas de los Datos", font=('Helvetica', 15), size=(30,1),text_color='#F6CEF5')
                            ],
                            [
                                sg.Table(values=data,
                                headings=header_list,
                                display_row_numbers=False,
                                auto_size_columns=False,
                                num_rows=min(30, len(data)),
                                background_color = "white",
                                text_color = "black",
                                header_background_color = "#F6CEF5",
                                key="-Statistic-")
                            #    sg.Table(values=statistic,
                            #    headings=header_stat,
                            #    font='Helvetica',
                            #    pad=(10,10),
                            #    auto_size_columns=True,
                            #    num_rows=min(25, len(data)))
                            ]
                        ]

    tab1_Col = [[sg.Column(tab2_layout, scrollable=True, vertical_scroll_only=True, size=(1600, 700))]]

    tab2_Col = [[sg.Column(statistic_layout, scrollable=True, vertical_scroll_only=True, size=(1600, 700))]]

    tabGroup1 = [
                    [sg.TabGroup([[sg.Tab('Estructura', tab1_Col), sg.Tab('Valores Atípicos', tab3_layout), sg.Tab('Estadisticas', tab2_Col), sg.Tab('Correlación', corr_layout)]])]
                ]

    layout = [
        [sg.TabGroup([[sg.Tab('EDA', tabGroup1), sg.Tab('Componentes principales', tab4_layout), sg.Tab('Clustering', tab5_layout)]])]
    ]

    # create the form and show it without the plot
    window = sg.Window('Analisis '+filename, 
                       layout,
                       finalize=True,  
                       font='Helvetica 12',
                       grab_anywhere=False)

    #fig = plt.figure(figsize=(5,3)) 
    df.hist(figsize=(14,14), xrot=45)
    plt.show()
    #fig_canvas_agg = draw_figure(window['AtipicData'].TKCanvas, fig)

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Mapa de calor":
            graph = plot_relacion(df)
            fig_canvas_agg = draw_figure(window['-CORR-'].TKCanvas, graph)
        if event == "Graficar":
            atipic = values["DatoAtipico"]
            if(atipic != None and atipic != "" and atipic in header_list):
                fig = plt.figure(figsize=(5,3)) 
                sns.boxplot(x=df[atipic], data=df)
                fig_canvas_agg = draw_figure(window['Atipic'].TKCanvas, fig)
            else:
                sg.popup_error('Error al intentar graficar, revise la cadena ingresada')
                #plt.show()

    window.close()

if __name__ == "__main__":

    layout =   [
                    [
                        sg.Image(filename="MD.png", key="-MD-"),
                        sg.Text("Minería de Datos GUI", font=("Helvetica", 25), text_color='#2ECCFA'),
                        sg.Image(filename="UNAM3.png", key="-UNAM-")
                    ],
                    [
                        sg.Text("Selecciona un archivo", font=(None, 15)),
                        sg.In(size=(30, 1), enable_events=True, key="-FILE-"),
                        sg.FileBrowse(file_types=(("CSV", "*.csv"),("ALL", "*"))),
                    ]
                ]

    ventana = sg.Window("Proyecto Mineria", layout)

    while True:
        event, values = ventana.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "-FILE-":
            filename = values["-FILE-"]
            #try:
            df = pd.read_csv(filename)
            data = df.values.tolist()
            header_list = df.columns.tolist()
            tipos = df.dtypes.tolist()
            nulls = df.isnull().sum().tolist()
            for i in range (0, len(tipos)):
                types.append([header_list[i], tipos[i], nulls[i]])
            correlaciones = df.corr(method='pearson')
            headers = correlaciones.columns.tolist()
            stats = df.describe()
            statistic = stats.values.tolist()
            for i,d in enumerate(statistic):
                d.insert(0,list(stats.index)[i])
            header_stat=['Feature']+header_list
            corr = correlaciones.values.tolist()
            new_window()
            break
            #except:
            #    sg.popup_error('El archivo seleccionado no puede abrirse')
            #    pass

    ventana.close()
