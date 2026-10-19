"""Benchmark: Minimax (búsqueda) contra el árbol de decisión (ML).

Mide el tiempo de respuesta, los nodos explorados y con qué frecuencia el
árbol elige la misma jugada que Minimax.
Ejecución, desde esta carpeta:  python comparativa_rendimiento.py
"""

import os
import random
import time

from model.game_model import TresEnRayaModel
from model.minimax_agent import MinimaxAgent
from model.ml_agent import MLAgent

RUTA_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dataset_tres_en_raya.csv")


def generar_tableros_prueba(cantidad):
    """Crea tableros legales, no terminales, donde le toca jugar a O.

    X empieza, por eso O mueve cuando hay 1, 3 o 5 fichas colocadas.
    """
    tableros = []
    while len(tableros) < cantidad:
        juego = TresEnRayaModel()
        ficha = 'X'
        for _ in range(random.choice((1, 3, 5))):
            juego.make_move(random.choice(juego.get_available_moves()), ficha)
            ficha = 'O' if ficha == 'X' else 'X'
        if not juego.check_winner('X') and not juego.check_winner('O'):
            tableros.append(juego)
    return tableros


def imprimir_fila(metrica, valor_minimax, valor_ml):
    """Imprime una fila de la tabla de resultados."""
    print(f"{metrica:<30} | {valor_minimax:<20} | {valor_ml:<20}")


def ejecutar_comparativa_rendimiento(num_pruebas=100):
    """Compara ambos agentes en num_pruebas tableros y muestra la tabla."""
    agente_minimax = MinimaxAgent(ai_player='O', human_player='X')
    agente_ml = MLAgent(ai_player='O', human_player='X')
    if not os.path.exists(RUTA_DATASET):
        agente_ml.generar_dataset_sintetico(1000, RUTA_DATASET)
    agente_ml.cargar_y_limpiar_dataset(RUTA_DATASET)
    agente_ml.entrenar_arbol(criterion='entropy')

    tiempos_minimax = []
    nodos_minimax = []
    tiempos_ml = []
    coincidencias = 0

    for juego in generar_tableros_prueba(num_pruebas):
        inicio = time.perf_counter()
        jugada_minimax = agente_minimax.get_best_move(juego)
        tiempos_minimax.append(time.perf_counter() - inicio)
        nodos_minimax.append(agente_minimax.nodes_evaluated)

        inicio = time.perf_counter()
        jugada_ml = agente_ml.predecir_jugada(juego.tablero)
        tiempos_ml.append(time.perf_counter() - inicio)

        if jugada_ml == jugada_minimax:
            coincidencias += 1

    promedio_minimax = sum(tiempos_minimax) / num_pruebas
    promedio_ml = sum(tiempos_ml) / num_pruebas
    aceleracion = promedio_minimax / promedio_ml if promedio_ml > 0 else 0.0

    print("\nRESULTADOS (Minimax vs Árbol de Decisión)")
    print("-" * 76)
    imprimir_fila("Métrica", "Minimax", "Árbol ML")
    print("-" * 76)
    imprimir_fila("Tiempo promedio por jugada",
                  f"{promedio_minimax * 1000:.4f} ms",
                  f"{promedio_ml * 1000:.4f} ms")
    imprimir_fila(f"Tiempo total ({num_pruebas} jugadas)",
                  f"{sum(tiempos_minimax):.4f} s",
                  f"{sum(tiempos_ml):.4f} s")
    imprimir_fila("Nodos explorados (promedio)",
                  f"{sum(nodos_minimax) / num_pruebas:,.1f}", "0")
    imprimir_fila("Nodos explorados (total)",
                  f"{sum(nodos_minimax):,}", "0")
    imprimir_fila("Coincidencia con Minimax", "100% (referencia)",
                  f"{coincidencias / num_pruebas:.2%}")
    imprimir_fila("Velocidad", "1.0x", f"{aceleracion:.1f}x más rápido")

    return {
        "tiempo_minimax_ms": promedio_minimax * 1000,
        "tiempo_ml_ms": promedio_ml * 1000,
        "nodos_minimax_promedio": sum(nodos_minimax) / num_pruebas,
        "aceleracion": aceleracion,
        "coincidencia_ml": coincidencias / num_pruebas * 100,
    }


if __name__ == '__main__':
    ejecutar_comparativa_rendimiento(num_pruebas=100)
