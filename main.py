"""Tres en Raya con Minimax, Backtracking y Árbol de Decisión (MVC).

Proyecto educativo de 3 semanas. Se puede jugar en tres modos:
  1. Humano vs Humano.
  2. Humano vs IA Minimax: busca en el árbol de juego con backtracking y
     muestra los nodos explorados y el tiempo de respuesta.
  3. Humano vs IA Machine Learning: un DecisionTreeClassifier entrenado con
     jugadas de Minimax predice la jugada sin explorar el árbol.

Arquitectura (patrón MVC):
  model/       game_model.py, minimax_agent.py, ml_agent.py
  view/        gui_view.py (interfaz Tkinter)
  controller/  game_controller.py

Requisitos: Python 3, pandas, scikit-learn y matplotlib.
Ejecución, desde esta carpeta:  python main.py
Benchmark Minimax vs Árbol ML:  python comparativa_rendimiento.py
"""

import os
import tkinter as tk

from controller.game_controller import GameController
from model.game_model import TresEnRayaModel
from model.minimax_agent import MinimaxAgent
from model.ml_agent import MLAgent
from view.gui_view import TresEnRayaView

RUTA_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dataset_tres_en_raya.csv")


def preparar_agente_ml():
    """Entrena el árbol de decisión (genera el dataset si no existe)."""
    agente_ml = MLAgent(ai_player='O', human_player='X')
    if not os.path.exists(RUTA_DATASET):
        agente_ml.generar_dataset_sintetico(
            num_partidas=1000, ruta_salida_csv=RUTA_DATASET)
    agente_ml.cargar_y_limpiar_dataset(RUTA_DATASET)
    agente_ml.entrenar_arbol(criterion='entropy')
    return agente_ml


def main():
    """Crea Modelo, Vista y Controlador y abre la ventana."""
    ventana = tk.Tk()
    modelo = TresEnRayaModel()
    vista = TresEnRayaView(ventana)
    GameController(
        modelo=modelo,
        vista=vista,
        agente_minimax=MinimaxAgent(ai_player='O', human_player='X'),
        agente_ml=preparar_agente_ml(),
    )
    ventana.mainloop()


if __name__ == "__main__":
    main()
