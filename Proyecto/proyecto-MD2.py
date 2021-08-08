#!/usr/bin/python3

import PySimpleGUI as sg
import re
#import os.path
import pandas as pd                                         # Para la manipulación de datos y análisis
import numpy as np                                          # Para crear vectores de datos, matrices de n dimensiones
import matplotlib.pyplot as plt                             # Para generar gráficos
import seaborn as sns                                       # Para visualización de los datos
from sklearn.preprocessing import StandardScaler            # Para estandarización de datos
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans                          # Para clustering
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

types, data, nulos, tempList = [], [], [], []
headVal = ["Eigenvalues"]

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
    mapCorr = 0
    NormDF = []
    header_num = [ i for i in range (0, len(df.columns.tolist()))]

    norm_layout =   [
                        [
                            sg.Text("Normalización de los Datos", font=('Helvetica', 15)), # text_color='#F6CEF5'
                        ],
                        [
                            sg.Text("Variables dependientes a eliminar", font=('Helvetica', 15)),
                            sg.In(size=(30, 1), enable_events=True, key="-VarDependiente-"),
                            sg.Button("Eliminar")
                        ],
                        [
                            sg.Table(values=NormDF,
                            headings=header_list,
                            display_row_numbers=False,
                            auto_size_columns=False,
                            num_rows=min(20, len(NormDF)),
                            alternating_row_color='lightblue',
                            key="-Norm-")
                        ]
                    ]

    componentes_layout =[
                            [
                                sg.Text("Calculo de Componentes y varianza", font=('Helvetica', 15))
                            ],
                            [
                                sg.Text("Número de Componentes", font=('Helvetica', 15)),
                                sg.In(size=(30, 1), enable_events=True, key="-CompNum-"),
                                sg.Button("Calcular", key="CompNum")
                            ],
                            [
                                sg.Text("Componentes Normalizado", font=('Helvetica', 15))
                            ],
                            [
                                sg.Table(values=NormDF,
                                headings=header_num,
                                display_row_numbers=False,
                                auto_size_columns=False,
                                num_rows=min(20, len(NormDF)),
                                alternating_row_color='lightblue',
                                key="-CompNorm-")
                            ],
                            [
                                sg.Text("Componentes", font=('Helvetica', 15))
                            ],
                            [
                                sg.Table(values=NormDF,
                                headings=header_num,
                                display_row_numbers=False,
                                auto_size_columns=False,
                                num_rows=min(20, len(NormDF)),
                                alternating_row_color='lightblue',
                                key="-Comp-")
                            ]
                        ]

    numComp_layout =    [
                            [
                                sg.Text("Elección Número de Componentes", font=('Helvetica', 15))
                            ],
                            [
                                sg.Text("Porcentaje de relevancia", font=('Helvetica', 15))
                            ],
                            [
                                sg.Text("Límite inferior", font=('Helvetica', 12)),
                                sg.In(size=(10, 1), enable_events=True, key="-NumInf-"),
                                sg.Text("Límite superior", font=('Helvetica', 12)),
                                sg.In(size=(10, 1), enable_events=True, key="-NumSup-")
                            ],
                            [
                                sg.Button("Calculo", key="varianzaCalc")
                            ],
                            [
                                sg.Table(values=NormDF,
                                headings=headVal,
                                display_row_numbers=False,
                                auto_size_columns=False,
                                num_rows=min(30, len(NormDF)),
                                alternating_row_color='lightblue',
                                key="-EigenVal-")
                            ],
                            [
                                sg.Text(size=(35,2), font=('Helvetica', 12), key="-Varianza-")
                            ],
                            [
                                sg.Canvas(key='-VarAc-', 
                                pad=(10,10))
                            ]
                        ]

    corr_layout =   [
                        [
                            sg.Text("Correlación de los Datos", font=('Helvetica', 15)),
                        ],
                        [
                            sg.Table(values=corr,
                            headings=headers,
                            display_row_numbers=False,
                            auto_size_columns=False,
                            num_rows=min(20, len(headers)),
                            alternating_row_color='lightblue',
                            key="-Corr-")
                        ],
                        [
                            sg.Canvas(key='-CORR-', 
                            pad=(10,10)),
                            sg.Button("Mapa de calor")
                        ]
                    ]

    tab2_layout =   [
                        [
                            sg.Text("Datos", font=('Helvetica', 15)),
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
                            sg.Text("Estructura de los Datos", font=('Helvetica', 15), size=(30,1))
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
                        ]
                    ]

    tab3_layout =   [
                        [
                            sg.Text("Valores Atípicos", font=('Helvetica', 15), size=(30,1))
                        ],
                        [
                            sg.Text("Histograma", font=('Helvetica', 15), size=(30,1)),
                            sg.Button("Graficar", key="Histograma")
                        ],
                        [
                            sg.Text("Selecciona el Dato", font=(None, 15)),
                            sg.In(size=(30, 1), enable_events=True, key="DatoAtipico"),
                            sg.Button("Graficar")
                        ]
                    ]

    if flag_obj == 1:
        statistic_layout =  [
                                [
                                    sg.Text("Estadisticas de los Datos", font=('Helvetica', 15), size=(30,1))
                                ],
                                [ 
                                    sg.Text("Datos Numéricos", font=('Helvetica', 15), size=(30,1))
                                ],
                                [
                                    sg.Table(values=statistic,
                                    headings=header_stat,
                                    display_row_numbers=False,
                                    auto_size_columns=True,
                                    num_rows=min(30, len(statistic)),
                                    background_color = "white",
                                    text_color = "black",
                                    header_background_color = "#F6CEF5",
                                    key="-StatisticNum-")
                                ],
                                [ 
                                    sg.Text("Datos Nominales", font=('Helvetica', 15), size=(30,1))
                                ],
                                [
                                    sg.Table(values=statisticNom,
                                    headings=header_statNom,
                                    display_row_numbers=False,
                                    auto_size_columns=True,
                                    num_rows=min(30, len(statisticNom)),
                                    background_color = "white",
                                    text_color = "black",
                                    header_background_color = "#F6CEF5",
                                    key="-StatisticNom-")
                                ]
                            ]
    else:
        statistic_layout =  [
                            [
                                sg.Text("Estadisticas de los Datos", font=('Helvetica', 15), size=(30,1))
                            ],
                            [ 
                                sg.Text("Datos Numéricos", font=('Helvetica', 15), size=(30,1))
                            ],
                            [
                                sg.Table(values=statistic,
                                headings=header_stat,
                                display_row_numbers=False,
                                auto_size_columns=True,
                                num_rows=min(30, len(statistic)),
                                background_color = "white",
                                text_color = "black",
                                header_background_color = "#F6CEF5",
                                key="-StatisticNum-")
                            ]
                        ]
    CP_layout = [
                    [
                        sg.Text("Clustering Particional", font=('Helvetica', 15), size=(30,1))
                    ],
                    [
                        sg.Text("Cantidad Clusters", font=('Helvetica', 15), size=(30,1))
                    ],
                    [
                        sg.Text("Límite inferior", font=('Helvetica', 12)),
                        sg.In(size=(10, 1), enable_events=True, key="-ClustInf-"),
                        sg.Text("Límite superior", font=('Helvetica', 12)),
                        sg.In(size=(10, 1), enable_events=True, key="-ClustSup-")
                    ],
                    [
                        sg.Button("Cálculo", key="ClustCalc"),
                        sg.Text(size=(35,1), font=('Helvetica', 12), key="-Elbow-")
                    ],
                    [
                        sg.Canvas(key='-ClustPart-', 
                        pad=(10,10))
                    ]
                ]
    Particional_layout =[
                            [
                                sg.Text("Creación de Etiquetas", font=('Helvetica', 15), size=(30,1))
                            ],
                            [
                                sg.Table(values=NormDF,
                                headings=header_cluster,
                                display_row_numbers=False,
                                auto_size_columns=False,
                                num_rows=min(30, len(NormDF)),
                                alternating_row_color='lightblue',
                                key="-ClustData-")
                            ],
                            [
                                sg.Text("Cantidad de Datos por Cluster", font=('Helvetica', 15), size=(30,1))
                            ],
                            [
                                sg.Table(values=NormDF,
                                headings=["No. Cluster", "Cantidad de Datos"],
                                display_row_numbers=False,
                                auto_size_columns=False,
                                num_rows=min(5, len(NormDF)),
                                alternating_row_color='lightblue',
                                key="-ClustData1-")
                            ],
                            [
                                sg.Canvas(key='-ClustPartGraph-', 
                                pad=(10,10))
                            ]
                        ]
    Centroides_layout = [
                            [
                                sg.Text("Centroides", font=('Helvetica', 15), size=(30,1))
                            ],
                            [
                                sg.Table(values=NormDF,
                                headings=header_list,
                                display_row_numbers=False,
                                auto_size_columns=False,
                                num_rows=min(5, len(NormDF)),
                                alternating_row_color='lightblue',
                                key="-ClustCenter-")
                            ],
                            [
                                sg.Button("Graficar Clusters", key="GraphClust"),
                            ],
                            [
                                sg.Canvas(key='-ClustCenterGraph-', 
                                pad=(10,10))
                            ]
                        ]
    variable_layout =   [
                            [
                                sg.Text("Definición de variables", font=('Helvetica', 15), size=(30,1))
                            ]
                        ]

    tab1_Col = [[sg.Column(tab2_layout, scrollable=True, vertical_scroll_only=True, size=(1200, 700))]]

    tab2_Col = [[sg.Column(statistic_layout, scrollable=True, vertical_scroll_only=True, size=(1200, 700))]]

    tab3_Col = [[sg.Column(componentes_layout, scrollable=True, vertical_scroll_only=True, size=(1200, 700))]]

    tabGroupEDA = [
                    [sg.TabGroup([[sg.Tab('Estructura', tab1_Col), sg.Tab('Valores Atípicos', tab3_layout), sg.Tab('Estadisticas', tab2_Col), sg.Tab('Correlación', corr_layout)]])]
                ]
    tabGroupPCA = [
                    [sg.TabGroup([[sg.Tab('Normalización', norm_layout), sg.Tab('Componentes', tab3_Col), sg.Tab('Elección Componentes', numComp_layout)]])]
                ]

    tabGroupClustering = [
                    [sg.TabGroup([[sg.Tab('Clustering Particional', CP_layout), sg.Tab('Etiquetado', Particional_layout), sg.Tab('Centroides', Centroides_layout)]])]
                ]
    tabGroupClasificacion = [
                    [sg.TabGroup([[sg.Tab('Definición de variables', variable_layout)]])]
                ]

    layout = [
        [sg.TabGroup([[sg.Tab('EDA', tabGroupEDA), sg.Tab('Componentes principales', tabGroupPCA), sg.Tab('Clustering', tabGroupClustering), sg.Tab('Regresión', tabGroupClasificacion)]])]
    ]

    fn = filename.split('/')[-1]
    window = sg.Window('Analisis '+fn, 
                       layout,
                       finalize=True,  
                       font='Helvetica 12',
                       grab_anywhere=False,
                       size=(1200, 700))

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Mapa de calor":
            if mapCorr == 0:
                graph = plot_relacion(df)
                fig_canvas_agg = draw_figure(window['-CORR-'].TKCanvas, graph)
                mapCorr = 1
        if event == "Histograma":
            df.hist(figsize=(14,14), xrot=45)
            plt.show()
        if event == "Graficar":
            atipic = values["DatoAtipico"]
            if(atipic != None and atipic != "" and atipic in header_list):
                fig = plt.figure(figsize=(5,3)) 
                sns.boxplot(x=df[atipic], data=df)
                plt.show()
            else:
                sg.popup_error('Error al intentar graficar, revise la cadena ingresada')
        if event == "Eliminar":
            varDep = values["-VarDependiente-"]
            normalizar = StandardScaler()
            colValues = []
            Mdf = df
            try:
                for var in varDep.split(' '):
                    colValues.append(df.columns.get_loc(var))
                    Mdf = Mdf.drop([var], axis=1)
                try:
                    normalizar.fit(Mdf)
                    NormDF = normalizar.transform(Mdf)
                    dfTemp = []    
                    for norm in NormDF.tolist():
                        for i in colValues:
                            norm.insert(i, "-")
                        dfTemp.append(norm)
                    tempDataFrame = pd.DataFrame(dfTemp, columns=df.columns)
                    window.FindElement("-Norm-").Update(values=tempDataFrame.values.tolist())
                    window.FindElement("-Norm-").Update(num_rows=min(30, len(tempDataFrame.values.tolist())))
                    # Suburb Address Type Method SellerG Date CouncilArea Regionname
                except:
                    sg.popup_error('Problema para normalizar datos, verifique que no existan valores tipo string')
            except:
                normalizar.fit(Mdf)
                NormDF = normalizar.transform(Mdf)
                tempDataFrame = pd.DataFrame(NormDF, columns=df.columns)
                window.FindElement("-Norm-").Update(values=tempDataFrame.values.tolist())
                window.FindElement("-Norm-").Update(num_rows=min(30, len(tempDataFrame.values.tolist())))
        if event == "CompNum":
            numComp = values["-CompNum-"]
            if numComp == None or numComp == "" or numComp == " ":
                Componentes = PCA(n_components=None)
                Componentes.fit(NormDF)
                X_Comp = Componentes.transform(NormDF)
                tempDataFrame = pd.DataFrame(X_Comp)
                window.FindElement("-CompNorm-").Update(values=tempDataFrame.values.tolist())
                window.FindElement("-CompNorm-").Update(num_rows=min(10, len(tempDataFrame.values.tolist())))
            elif re.match(r'[0-9]*', numComp):
                numComp = int(numComp)
                if numComp < len(df.columns.tolist()):
                    Componentes = PCA(n_components=numComp)
                    Componentes.fit(NormDF)
                    X_Comp = Componentes.transform(NormDF)
                    tempDataFrame = pd.DataFrame(X_Comp)
                    window.FindElement("-CompNorm-").Update(values=tempDataFrame.values.tolist())
                    window.FindElement("-CompNorm-").Update(num_rows=min(10, len(tempDataFrame.values.tolist())))
            tempDataFrame = pd.DataFrame(Componentes.components_)
            window.FindElement("-Comp-").Update(values=tempDataFrame.values.tolist())
            window.FindElement("-Comp-").Update(num_rows=min(10, len(tempDataFrame.values.tolist())))
        if event == "varianzaCalc":
            numInf = int(values["-NumInf-"]) / 100
            numSup = int(values["-NumSup-"]) / 100
            Varianza = Componentes.explained_variance_ratio_
            temp = []
            maxVal = 0
            cont = 0
            for var in Varianza:
                temp.append([var])
                tempVal = sum(Varianza[0:cont+1])
                if tempVal < numSup:
                    cont = cont + 1
                    maxVal = tempVal
            window.FindElement("-EigenVal-").Update(values=temp)
            window.FindElement("-EigenVal-").Update(num_rows=min(10, len(temp)))
            cadenaVar = "Varianza acumulada: " + str(maxVal) + "\nNúmero de Componentes: " + str(cont)
            window.FindElement("-Varianza-").Update(cadenaVar)
            fig = plt.figure(figsize=(5,3))
            plt.plot(np.cumsum(Componentes.explained_variance_ratio_))
            plt.xlabel('Número de componentes')
            plt.ylabel('Varianza acumulada')
            plt.grid()
            fig_canvas_agg = draw_figure(window['-VarAc-'].TKCanvas, fig)
        if event == "ClustCalc":
            numInf = int(values["-ClustInf-"])
            numSup = int(values["-ClustSup-"])
            SSE =[]
            for i in range(numInf, numSup):
              km = KMeans(n_clusters=i, random_state=0)
              km.fit(Mdf)
              SSE.append(km.inertia_)
            kl = KneeLocator(range(numInf,numSup), SSE, curve="convex", direction="decreasing")
            elbowVal = kl.elbow
            window.FindElement("-Elbow-").Update("Número K (ELbow Method): " + str(elbowVal))
            plt.style.use("ggplot")
            kl.plot_knee()
            plt.show()
            fig = plt.figure(figsize=(6, 3))
            plt.plot(range(numInf, numSup), SSE, marker="o")
            plt.xlabel("Cantidad de clusters *k*")
            plt.ylabel("SSE")
            plt.title("Elbow Method")
            fig_canvas_agg = draw_figure(window['-ClustPart-'].TKCanvas, fig)
            MParticional = KMeans(n_clusters=elbowVal, random_state=0).fit(Mdf)
            MParticional.predict(Mdf)
            MParticional.labels_
            tempDataFrame = df
            tempDataFrame["clusterP"] = MParticional.labels_
            dataC = tempDataFrame.values.tolist()
            window.FindElement("-ClustData-").Update(values=dataC)
            window.FindElement("-ClustData-").Update(num_rows=min(10, len(dataC)))
            temp=[]
            for i in range(0,elbowVal):
                temp.append([i, tempDataFrame.groupby(["clusterP"])["clusterP"].count().tolist()[i]])
            window.FindElement("-ClustData1-").Update(values=temp)
            window.FindElement("-ClustData1-").Update(num_rows=min(5, elbowVal))
            fig = plt.figure(figsize=(10, 7))
            plt.scatter(Mdf.iloc[:, 0], Mdf.iloc[:, 1], c=MParticional.labels_, cmap="rainbow")
            fig_canvas_agg = draw_figure(window['-ClustPartGraph-'].TKCanvas, fig)
            CentroidesP = MParticional.cluster_centers_
            tempDF = pd.DataFrame(CentroidesP.round(4),columns=Mdf.columns.tolist())
            dfTemp = []    
            for val in tempDF.values.tolist():
                for i in colValues:
                    val.insert(i, "-")
                dfTemp.append(val)
            tempDataFrame = pd.DataFrame(dfTemp)
            window.FindElement("-ClustCenter-").Update(values=tempDataFrame.values.tolist())
            window.FindElement("-ClustCenter-").Update(num_rows=min(20, len(tempDataFrame.values.tolist())))
        if event == "GraphClust":
            plt.rcParams["figure.figsize"] = (10,7)
            plt.style.use("ggplot")
            colores=["red","blue","green","yellow","purple","cyan","pink","olive"]
            asignar, colorList = [], []
            for row in MParticional.labels_:
              asignar.append(colores[row])
            for row in range(0,elbowVal):
              colorList.append(colores[row])

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(Mdf.iloc[:,0],Mdf.iloc[:,1],Mdf.iloc[:,2], marker="o", c=asignar,s=60)
            ax.scatter(CentroidesP[:,0],CentroidesP[:,1],CentroidesP[:,2], marker="*", c=colorList,s=1000)
            fig_canvas_agg = draw_figure(window['-ClustCenterGraph-'].TKCanvas, fig)
            #plt.show()
            


    window.close()

if __name__ == "__main__":
    sg.theme('Purple')
    layout =   [
                    [
                        sg.Image(filename="MD.png", key="-MD-"),
                        sg.Text("Minería de Datos GUI", font=("Helvetica", 25), text_color='#0174DF'),
                        sg.Image(filename="UNAM.png", key="-UNAM-")
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
            stats = df.describe().T
            statistic = stats.values.tolist()
            for i,d in enumerate(statistic):
                d.insert(0,list(stats.index)[i])
            header_stat = list(stats.columns)
            header_stat=['Feature']+header_stat
            header_cluster = header_list + ["clusterP"]
            try:
                statsNom = df.describe( include="object").T
                statisticNom = statsNom.values.tolist()
                for i,d in enumerate(statisticNom):
                    d.insert(0,list(statsNom.index)[i])
                header_statNom = list(statsNom.columns)
                header_statNom =['Feature']+header_statNom
                flag_obj = 1
            except:
                flag_obj = 0
                pass
            corr = correlaciones.values.tolist()
            new_window()
            break
            #except:
            #    sg.popup_error('El archivo seleccionado no puede abrirse')
            #    pass

    ventana.close()
