"""
Módulo del Agente de Machine Learning y Generador de Datos Sintéticos (Semana 3).
Permite la simulación automatizada de partidas entre agentes, la recolección
de estados de juego y la construcción del dataset para el entrenamiento del Árbol de Decisión.
"""

import os
import random
import csv
from typing import List, Tuple, Dict, Optional

# Importamos las clases del Modelo existentes
from model.game_model import TresEnRayaModel
from model.minimax_agent import MinimaxAgent


class MLAgent:
    """
    Agente basado en Machine Learning para el juego Tres en Raya.
    En el Día 1 se enfoca en la simulación automatizada de partidas y recolección
    de datos sintéticos supervisados por Minimax.
    """

    # Mapeo numérico estándar para las casillas del tablero:
    # 0 = Vacía, 1 = 'X' (Humano / Rival), -1 = 'O' (IA)
    MAPEO_FICHAS = {
        ' ': 0,
        'X': 1,
        'O': -1
    }

    def __init__(self, ai_player: str = 'O', rival_player: str = 'X'):
        """
        Inicializa el agente de Machine Learning.
        
        Args:
            ai_player (str): Ficha asignada a la IA ('O' por defecto).
            rival_player (str): Ficha asignada al rival ('X' por defecto).
        """
        self.ai_player = ai_player
        self.rival_player = rival_player
        
        # Referencia al agente Minimax para la supervisión y etiquetado de jugadas
        self.supervisor_minimax = MinimaxAgent(ai_player=self.ai_player, human_player=self.rival_player)
        
        # Almacén de registros sintéticos en memoria
        self.dataset_records: List[Dict] = []
        
        # Atributos reservados para los Días 3 y 4 (Modelo de Scikit-Learn)
        self.model = None
        self.is_trained = False
        self.accuracy = 0.0

    @classmethod
    def vectorizar_tablero(cls, tablero: List[List[str]]) -> List[int]:
        """
        Convierte una matriz 3x3 de caracteres en un vector plano de 9 enteros.
        
        Args:
            tablero (List[List[str]]): Matriz 3x3 del estado del juego.
            
        Returns:
            List[int]: Vector unidimensional de longitud 9 con valores (-1, 0, 1).
        """
        vector = []
        for fila in range(3):
            for col in range(3):
                casilla = tablero[fila][col]
                vector.append(cls.MAPEO_FICHAS.get(casilla, 0))
        return vector

    @staticmethod
    def coordenadas_a_indice(fila: int, col: int) -> int:
        """
        Convierte una coordenada (fila, col) al índice plano correspondiente (0 a 8).
        """
        return fila * 3 + col

    @staticmethod
    def indice_a_coordenadas(indice: int) -> Tuple[int, int]:
        """
        Convierte un índice plano (0 a 8) a coordenadas matriciales (fila, col).
        """
        return divmod(indice, 3)

    def simular_partida(
        self,
        prob_aleatorio_rival: float = 0.5,
        prob_aleatorio_ia: float = 0.2
    ) -> Dict:
        """
        Ejecuta una partida automatizada entre la IA y un rival simulado,
        recolectando en cada turno de la IA el par (estado_tablero, mejor_movimiento).
        
        Para garantizar la máxima diversidad del dataset (exploración del espacio de estados),
        los agentes aplican una política epsilon-greedy: con cierta probabilidad eligen
        una jugada aleatoria válida en vez de la óptima de Minimax.
        
        Args:
            prob_aleatorio_rival (float): Probabilidad de que el rival juegue aleatorio (0.0 a 1.0).
            prob_aleatorio_ia (float): Probabilidad de que la IA juegue aleatorio para variar partidas.
            
        Returns:
            Dict: Resumen de la partida (ganador, número de turnos, ejemplos recolectados).
        """
        juego = TresEnRayaModel()
        turno = 'X'  # Convención tradicional: 'X' inicia la partida
        ejemplos_partida = []
        
        while True:
            movimientos_validos = juego.get_available_moves()
            if not movimientos_validos:
                # Tablero lleno -> Empate
                ganador = 'Empate'
                break

            # -----------------------------------------------------------
            # Turno de la IA ('O'): Se recolecta la etiqueta supervisada
            # -----------------------------------------------------------
            if turno == self.ai_player:
                # 1. Minimax determina SIEMPRE la jugada óptima teórica para este estado
                mejor_movimiento_minimax = self.supervisor_minimax.get_best_move(juego)
                
                if mejor_movimiento_minimax is not None:
                    # 2. Vectorizamos el tablero antes de aplicar la jugada
                    vector_estado = self.vectorizar_tablero(juego.tablero)
                    indice_objetivo = self.coordenadas_a_indice(*mejor_movimiento_minimax)
                    
                    # 3. Guardamos la muestra sintética: (c0..c8 -> jugada_optima)
                    ejemplos_partida.append({
                        'c0': vector_estado[0],
                        'c1': vector_estado[1],
                        'c2': vector_estado[2],
                        'c3': vector_estado[3],
                        'c4': vector_estado[4],
                        'c5': vector_estado[5],
                        'c6': vector_estado[6],
                        'c7': vector_estado[7],
                        'c8': vector_estado[8],
                        'jugada_optima': indice_objetivo
                    })

                # 4. Decidir qué jugada aplicar físicamente en la simulación
                if random.random() < prob_aleatorio_ia:
                    movimiento_a_jugar = random.choice(movimientos_validos)
                else:
                    movimiento_a_jugar = mejor_movimiento_minimax or random.choice(movimientos_validos)

                juego.make_move(movimiento_a_jugar, self.ai_player)

                # Verificar si ganó la IA
                if juego.check_winner(self.ai_player):
                    ganador = self.ai_player
                    break

                turno = self.rival_player

            # -----------------------------------------------------------
            # Turno del Rival ('X'): Simula humano (mezcla Minimax y aleatorio)
            # -----------------------------------------------------------
            else:
                if random.random() < prob_aleatorio_rival:
                    movimiento_rival = random.choice(movimientos_validos)
                else:
                    # El rival también puede calcular una buena jugada
                    supervisor_rival = MinimaxAgent(ai_player=self.rival_player, human_player=self.ai_player)
                    movimiento_rival = supervisor_rival.get_best_move(juego) or random.choice(movimientos_validos)

                juego.make_move(movimiento_rival, self.rival_player)

                # Verificar si ganó el rival
                if juego.check_winner(self.rival_player):
                    ganador = self.rival_player
                    break

                turno = self.ai_player

        return {
            'ganador': ganador,
            'ejemplos': ejemplos_partida
        }

    def generar_dataset_sintetico(
        self,
        num_partidas: int = 1000,
        ruta_salida_csv: Optional[str] = None,
        eliminar_duplicados: bool = True
    ) -> List[Dict]:
        """
        Simula múltiples partidas automatizadas y consolida el dataset de entrenamiento.
        
        Args:
            num_partidas (int): Cantidad de partidas a simular (por defecto 1,000+ según la guía).
            ruta_salida_csv (Optional[str]): Ruta del archivo donde exportar el CSV.
            eliminar_duplicados (bool): Si es True, preserva estados únicos con su mejor decisión.
            
        Returns:
            List[Dict]: Lista con todos los registros recolectados.
        """
        print(f"[*] Iniciando simulación de {num_partidas} partidas automatizadas...")
        
        todos_los_ejemplos = []
        estadisticas = {'O': 0, 'X': 0, 'Empate': 0}

        for i in range(1, num_partidas + 1):
            # Alternamos perfiles de rival para enriquecer la variedad de tableros:
            # - Partidas contra rival 100% aleatorio (descubre errores y tableros caóticos)
            # - Partidas contra rival 50% inteligente (juego semi-competitivo)
            # - Partidas contra rival Minimax perfecto (juego de alta competitividad)
            if i % 3 == 0:
                prob_rival = 0.8  # Mayormente aleatorio
            elif i % 3 == 1:
                prob_rival = 0.4  # Balanceado
            else:
                prob_rival = 0.0  # Rival Minimax óptimo

            resultado = self.simular_partida(prob_aleatorio_rival=prob_rival)
            todos_los_ejemplos.extend(resultado['ejemplos'])
            estadisticas[resultado['ganador']] = estadisticas.get(resultado['ganador'], 0) + 1

            if i % 250 == 0 or i == num_partidas:
                print(f"    -> {i}/{num_partidas} partidas completadas. Ejemplos acumulados: {len(todos_los_ejemplos)}")

        # Filtrado opcional de duplicados para optimizar el dataset
        if eliminar_duplicados:
            registros_unicos = {}
            for ejemplo in todos_los_ejemplos:
                clave_estado = tuple(ejemplo[f'c{idx}'] for idx in range(9))
                # Guardamos el mapeo único del estado a su jugada óptima
                registros_unicos[clave_estado] = ejemplo['jugada_optima']
            
            datos_finales = []
            for estado, jugada in registros_unicos.items():
                fila_dict = {f'c{idx}': estado[idx] for idx in range(9)}
                fila_dict['jugada_optima'] = jugada
                datos_finales.append(fila_dict)
        else:
            datos_finales = todos_los_ejemplos

        self.dataset_records = datos_finales

        print("\n[+] Resumen de la generación de datos:")
        print(f"    - Total partidas simuladas: {num_partidas}")
        print(f"    - Victorias IA ('O'): {estadisticas.get('O', 0)} ({estadisticas.get('O', 0)/num_partidas*100:.1f}%)")
        print(f"    - Victorias Rival ('X'): {estadisticas.get('X', 0)} ({estadisticas.get('X', 0)/num_partidas*100:.1f}%)")
        print(f"    - Empates: {estadisticas.get('Empate', 0)} ({estadisticas.get('Empate', 0)/num_partidas*100:.1f}%)")
        print(f"    - Muestras brutas generadas: {len(todos_los_ejemplos)}")
        print(f"    - Muestras únicas consolidadas: {len(datos_finales)}")

        # Exportación a CSV si se especificó ruta
        if ruta_salida_csv:
            self.guardar_csv(ruta_salida_csv)

        return datos_finales

    def guardar_csv(self, ruta_archivo: str) -> None:
        """
        Guarda los registros recolectados en formato CSV con cabeceras estándar:
        c0, c1, c2, c3, c4, c5, c6, c7, c8, jugada_optima
        """
        if not self.dataset_records:
            print("[!] Advertencia: No hay registros para guardar. Ejecuta generar_dataset_sintetico primero.")
            return

        # Asegurar creación del directorio padre si no existe
        directorio = os.path.dirname(ruta_archivo)
        if directorio and not os.path.exists(directorio):
            os.makedirs(directorio, exist_ok=True)

        columnas = [f'c{i}' for i in range(9)] + ['jugada_optima']

        with open(ruta_archivo, mode='w', newline='', encoding='utf-8') as archivo_csv:
            escritor = csv.DictWriter(archivo_csv, fieldnames=columnas)
            escritor.writeheader()
            escritor.writerows(self.dataset_records)

        print(f"[+] Dataset exportado exitosamente a: {ruta_archivo} ({len(self.dataset_records)} filas)")


# ------------------------------------------------------------------------------
# Script ejecutable directo para pruebas del Día 1
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("  SEMANA 3 - DÍA 1: GENERADOR DE DATOS SINTÉTICOS Y SIMULADOR")
    print("=" * 65)
    
    agente_ml = MLAgent()
    
    # Ruta estándar para el dataset en la raíz del proyecto
    ruta_dataset = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset_tres_en_raya.csv")
    
    # Generar 1,000 partidas automatizadas según la guía del curso
    agente_ml.generar_dataset_sintetico(
        num_partidas=1000,
        ruta_salida_csv=ruta_dataset,
        eliminar_duplicados=True
    )
    
    print("\n[OK] Dia 1 completado exitosamente.")
