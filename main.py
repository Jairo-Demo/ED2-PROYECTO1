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
    # 1. Creamos la ventana principal del sistema
    root = tk.Tk()

    # 2. Creamos el cerebro del juego (las reglas matemáticas)
    modelo = TresEnRayaModel()

    # 3. Creamos la interfaz gráfica pasándole la ventana
    vista = TresEnRayaView(root)

    # 4. Creamos el controlador y le entregamos el modelo y la vista 
    # para que los conecte y los haga hablar entre sí
    controlador = GameController(modelo, vista)

    # 5. ¡Arrancamos el juego y dejamos la ventana abierta esperando clics!
    root.mainloop()
