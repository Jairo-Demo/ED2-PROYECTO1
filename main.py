"""
Punto de Entrada Principal del Proyecto.
Integra los componentes del Patrón MVC:
- Modelo: TresEnRayaModel, MinimaxAgent, MLAgent (DecisionTreeClassifier).
- Vista: TresEnRayaView (Interfaz gráfica en Tkinter).
- Controlador: GameController (Coordinación de eventos y métricas en tiempo real).
"""

import os
import tkinter as tk

from model.game_model import TresEnRayaModel
from model.minimax_agent import MinimaxAgent
from model.ml_agent import MLAgent
from view.gui_view import TresEnRayaView
from controller.game_controller import GameController


if __name__ == "__main__":
    # 1. Creamos la ventana principal
    root = tk.Tk()

    # 2. Instanciamos el modelo del juego (reglas matemáticas 3x3)
    modelo = TresEnRayaModel()

    # 3. Instanciamos el Agente Minimax (Semana 2: IA Simbólica)
    agente_minimax = MinimaxAgent(ai_player='O', human_player='X')

    # 4. Instanciamos y entrenamos el Agente de Machine Learning (Semana 3: Árbol de Decisión)
    agente_ml = MLAgent(ai_player='O', rival_player='X')
    ruta_dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_tres_en_raya.csv")

    if not os.path.exists(ruta_dataset):
        # Generar datos sintéticos si fuera primera ejecución en un entorno limpio
        agente_ml.generar_dataset_sintetico(num_partidas=1000, ruta_salida_csv=ruta_dataset)

    agente_ml.cargar_y_limpiar_dataset(ruta_dataset)
    agente_ml.entrenar_arbol(criterion='entropy')

    # 5. Instanciamos la vista (Interfaz de usuario)
    vista = TresEnRayaView(root)

    # 6. Instanciamos el controlador que conecta la vista con ambos agentes de IA
    controlador = GameController(
        modelo=modelo,
        vista=vista,
        agente_ia=agente_minimax,
        agente_ml=agente_ml
    )

    # 7. Lanzamos la aplicación
    root.mainloop()
