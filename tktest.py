from tkinter import *
from tkinter import ttk

root = Tk()
frm = ttk.Frame(root, padding = 10)
frm.grid()
ttk.Label(frm, text="Hello World").grid(row=0,column=0)
ttk.Button(frm, text="Quit", command=root.destroy).grid(row=1,column=0)
root.mainloop()