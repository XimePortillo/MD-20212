from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd               # Para la manipulación de datos y análisis
import numpy as np                # Para crear vectores de datos, matrices de n dimensiones
import matplotlib.pyplot as plt   # Para generar gráficos
import seaborn as sns             # Para visualización de los datos
import platform

def Carga():
	style = ttk.Style()
	style.configure("TLabelframe", background="#A9E2F3")
	labelCarga.grid(column =0, row =1, padx=10,pady=5)

	buttonCarga = ttk.Button(labelCarga, text="Upload", command=fileDialog)
	buttonCarga.grid(column =1, row =1)

def fileDialog():
	global labelArch, filename, datosAnalizar
	if (platform.system() != "Windows"):
		filename = filedialog.askopenfilename(initialdir=".",title="Selecciona un archivo") 
	else:
		filename = filedialog.askopenfilename(initialdir=".",title="Selecciona un archivo",filetype=(("csv","*.csv"),("All files","*")))
	labelArch=ttk.Label(labelCarga, text="Se seleccionó archivo:")
	labelArch.grid(column =2, row =1)
	label=ttk.Label(labelCarga, text="")
	label.grid(column =3, row =1)
	label.configure(text = filename)
	try:
		datosAnalizar = pd.read_csv(filename)
	except:
		messagebox.showerror('Error','Archivo no válido, seleccione uno nuevo')
	EtapaAnalisis()

def EtapaAnalisis():
	labelEDA.configure(text="Analizando archivo "+filename)
	labelAnalisis = ttk.LabelFrame(labelEDA, text="Datos Disponibles")
	labelAnalisis.grid(column =1, row =1)
	tabla=ttk.Treeview(labelAnalisis,columns=tuple(datosAnalizar.columns))
	for i in range(len(datosAnalizar.index.values)-1,len(datosAnalizar.index.values)-101,-1):
		tabla.insert("",0,text=i,value=tuple(datosAnalizar.values[i]))
	for i in range(100,0,-1):
		tabla.insert("",0,text=i,value=tuple(datosAnalizar.values[i]))
	tabla.pack()	
	labelData = ttk.LabelFrame(labelEDA, text="Descripción de Datos")
	labelData.grid(column =1, row =3)

def EDA():
	style = ttk.Style()
	style.configure("TLabelframe", background="#A9E2F3")
	labelEDA.grid(column =0, row =1, padx=10,pady=5)
	

def Window():
	global labelEDA, labelCarga, tabla
	ventana=Tk()
	ventana.title("Mineria de Datos")
	if (platform.system() == "Windows"):
		ventana.iconbitmap("./MD.ico")
	ventana.geometry("700x450")
	ventana.config(bg="#cfeafa")
	notebook = ttk.Notebook()
	labelCarga=ttk.LabelFrame(text="Ingresa el archivo para analizar")
	labelEDA=ttk.LabelFrame(text="Analizando Archivo")
	Carga()
	EDA()
	notebook.add(labelCarga, text="Carga Datos", padding=20)
	notebook.add(labelEDA, text="AnalisisDatos", padding=20)
	notebook.pack(padx=10, pady=10)
	ventana.mainloop()

if __name__=="__main__":
	wind = Window()
