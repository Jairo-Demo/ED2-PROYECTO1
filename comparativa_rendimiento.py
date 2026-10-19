"""
Semana 3 - Día 5: Script de Pruebas Comparativas de Rendimiento.
Compara empíricamente la IA Simbólica (Minimax con Backtracking)
contra el Machine Learning Supervisado (DecisionTreeClassifier)
en términos de latencia (tiempo de respuesta), nodos explorados y precisión.
"""

import os
import sys
import time
import random
from typing import List, Dict

# Aseguramos inclusión de la raíz del proyecto para importaciones
DIRECTORIO_RAIZ = os.path.dirname(os.path.abspath(__file__))
if DIRECTORIO_RAIZ not in sys.path:
    sys.path.insert(0, DIRECTORIO_RAIZ)

from model.game_model import TresEnRayaModel
from model.minimax_agent import MinimaxAgent
from model.ml_agent import MLAgent


def generar_estados_prueba(num_estados: int = 100) -> List[TresEnRayaModel]:
    """
    Genera una colección diversa de tableros de prueba válidos
    con diferente número de turnos jugados (de 1 a 6 fichas ya colocadas).
    """
    estados = []
    print(f"[*] Generando {num_estados} tableros de prueba con jugadas aleatorias y legales...")

    while len(estados) < num_estados:
        juego = TresEnRayaModel()
        # Decidimos un número aleatorio de turnos previos (entre 1 y 6 jugadas)
        num_jugadas_previas = random.randint(1, 6)
        jugador_turno = 'X'

        partida_valida = True
        for _ in range(num_jugadas_previas):
            movimientos_libres = juego.get_available_moves()
            if not movimientos_libres or juego.check_winner('X') or juego.check_winner('O'):
                partida_valida = False
                break

            jugada = random.choice(movimientos_libres)
            juego.make_move(jugada, jugador_turno)
            jugador_turno = 'O' if jugador_turno == 'X' else 'X'

        # Solo guardamos si el tablero no es terminal y le toca jugar a la IA ('O')
        if partida_valida and len(juego.get_available_moves()) > 0 and not juego.check_winner('X') and not juego.check_winner('O'):
            estados.append(juego)

    return estados


def ejecutar_comparativa_rendimiento(num_pruebas: int = 100) -> Dict:
    """
    Ejecuta el experimento formal comparativo entre Minimax y el Árbol de Decisión.
    """
    print("=" * 72)
    print("  SEMANA 3 - DÍA 5: BENCHMARK COMPARATIVO (MINIMAX vs ÁRBOL ML)")
    print("=" * 72)

    # 1. Inicialización de los agentes
    print("\n[1] Inicializando y preparando los agentes inteligentes...")
    agente_minimax = MinimaxAgent(ai_player='O', human_player='X')

    agente_ml = MLAgent(ai_player='O', rival_player='X')
    ruta_dataset = os.path.join(DIRECTORIO_RAIZ, "dataset_tres_en_raya.csv")

    if not os.path.exists(ruta_dataset):
        print("    -> Generando dataset sintético inicial...")
        agente_ml.generar_dataset_sintetico(num_partidas=1000, ruta_salida_csv=ruta_dataset)

    agente_ml.cargar_y_limpiar_dataset(ruta_dataset)
    agente_ml.entrenar_arbol(criterion='entropy')

    # 2. Generación de tableros de prueba
    tableros = generar_estados_prueba(num_pruebas)

    # 3. Métricas acumuladas
    tiempos_minimax = []
    nodos_minimax = []
    tiempos_ml = []
    jugadas_coincidentes = 0

    print(f"\n[2] Ejecutando pruebas comparativas sobre {len(tableros)} posiciones de juego...")

    for i, tablero in enumerate(tableros, 1):
        # --- A. Evaluación con Minimax (Búsqueda Exhaustiva) ---
        t0_minimax = time.perf_counter()
        jugada_minimax = agente_minimax.get_best_move(tablero)
        t1_minimax = time.perf_counter()

        tiempo_minimax = t1_minimax - t0_minimax
        tiempos_minimax.append(tiempo_minimax)
        nodos_minimax.append(agente_minimax.nodes_evaluated)

        # --- B. Evaluación con Machine Learning (Inferencia por Árbol) ---
        t0_ml = time.perf_counter()
        jugada_ml = agente_ml.predecir_jugada(tablero.tablero)
        t1_ml = time.perf_counter()

        tiempo_ml = t1_ml - t0_ml
        tiempos_ml.append(tiempo_ml)

        # --- C. Verificación de coincidencia con la jugada óptima ---
        if jugada_ml == jugada_minimax:
            jugadas_coincidentes += 1

    # 4. Cálculo de promedios y estadísticas
    promedio_tiempo_minimax = sum(tiempos_minimax) / len(tiempos_minimax)
    promedio_tiempo_ml = sum(tiempos_ml) / len(tiempos_ml)
    promedio_nodos_minimax = sum(nodos_minimax) / len(nodos_minimax)
    total_nodos_minimax = sum(nodos_minimax)

    aceleracion = promedio_tiempo_minimax / promedio_tiempo_ml if promedio_tiempo_ml > 0 else 0.0
    porcentaje_precision = (jugadas_coincidentes / num_pruebas) * 100.0

    # 5. Visualización de la tabla comparativa
    print("\n" + "=" * 72)
    print("                    RESULTADOS EXPERIMENTALES FINALES")
    print("=" * 72)
    print(f"{'Métrica':<35} | {'Minimax (IA Simbólica)':<20} | {'Árbol ML (Scikit-Learn)':<20}")
    print("-" * 80)
    print(f"{'Tiempo promedio por jugada':<35} | {promedio_tiempo_minimax * 1000:10.4f} ms      | {promedio_tiempo_ml * 1000:10.4f} ms")
    print(f"{'Tiempo total (100 jugadas)':<35} | {sum(tiempos_minimax):10.4f} s       | {sum(tiempos_ml):10.4f} s")
    print(f"{'Nodos explorados promedio':<35} | {promedio_nodos_minimax:10.1f}          | {'0 (Inferencia)':<20}")
    print(f"{'Nodos totales explorados':<35} | {total_nodos_minimax:10,d}          | {'0 (Inferencia)':<20}")
    print(f"{'Precisión vs Jugada Óptima':<35} | {'100.00% (Referencia)':<20} | {porcentaje_precision:10.2f}%")
    print(f"{'Factor de Aceleración (Velocidad)':<35} | {'1.0x':<20} | {aceleracion:10.1f}x más rápido")
    print("=" * 72)

    return {
        "tiempo_minimax_ms": promedio_tiempo_minimax * 1000,
        "tiempo_ml_ms": promedio_tiempo_ml * 1000,
        "nodos_minimax_promedio": promedio_nodos_minimax,
        "aceleracion": aceleracion,
        "precision_ml": porcentaje_precision
    }


if __name__ == '__main__':
    ejecutar_comparativa_rendimiento(num_pruebas=100)
