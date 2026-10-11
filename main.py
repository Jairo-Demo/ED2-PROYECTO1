import tkinter as tk

# Importamos nuestras 3 partes del patrón MVC que construimos en los días anteriores
from model.game_model import TresEnRayaModel
from view.gui_view import TresEnRayaView
from controller.game_controller import GameController

"""
Día 5: Archivo principal de Integración.
Aquí es donde funciona el "Restaurante" completo.
Juntamos al Cocinero (Modelo), el Plato (Vista) y el Mesero (Controlador).
"""

if __name__ == "__main__":
   
    root = tk.Tk()

    modelo = TresEnRayaModel()

    vista = TresEnRayaView(root)

    controlador = GameController(modelo, vista)

    
    root.mainloop()
