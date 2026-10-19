"""
Módulo del Agente de Machine Learning y Generador de Datos Sintéticos (Semana 3).
Permite la simulación automatizada de partidas, recolección de estados de juego,
limpieza y estructuración del dataset con Pandas, y preparación de matrices (X, y)
para el entrenamiento del Árbol de Decisión con Scikit-Learn.
"""

import os
import sys
import random
import csv
from typing import List, Tuple, Dict, Optional

# Aseguramos que la raíz del proyecto esté en sys.path para poder ejecutar directamente este archivo
DIRECTORIO_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DIRECTORIO_RAIZ not in sys.path:
    sys.path.insert(0, DIRECTORIO_RAIZ)

# Librerías esenciales de Ciencia de Datos y Machine Learning
import pandas as pd
from sklearn.model_selection import train_test_split

# Importamos las clases del Modelo existentes
from model.game_model import TresEnRayaModel
from model.minimax_agent import MinimaxAgent


class MLAgent:
    """
    Agente basado en Machine Learning para el juego Tres en Raya.
    
    Semana 3:
    - Día 1: Simulación automatizada y recolección de datos sintéticos supervisados por Minimax.
    - Día 2: Limpieza, estructuración en DataFrame (Pandas) y partición en X (entradas) e y (objetivo).
    - Días posteriores: Entrenamiento e inferencia del Árbol de Decisión.
    """

    # Mapeo numérico estándar para las casillas del tablero:
    # 0 = Vacía, 1 = 'X' (Humano / Rival), -1 = 'O' (IA)
    MAPEO_FICHAS = {
        ' ': 0,
        'X': 1,
        'O': -1
    }

    # Nombres de las 9 características (casillas del tablero 3x3)
    COLUMNAS_CARACTERISTICAS = [f'c{i}' for i in range(9)]
    COLUMNA_OBJETIVO = 'jugada_optima'

    def __init__(self, ai_player: str = 'O', rival_player: str = 'X'):
        """
        Inicializa el agente de Machine Learning y sus estructuras de datos.
        
        Args:
            ai_player (str): Ficha asignada a la IA ('O' por defecto).
            rival_player (str): Ficha asignada al rival ('X' por defecto).
        """
        self.ai_player = ai_player
        self.rival_player = rival_player
        
        # Referencia al agente Minimax para la supervisión y etiquetado de jugadas
        self.supervisor_minimax = MinimaxAgent(ai_player=self.ai_player, human_player=self.rival_player)
        
        # Almacén de registros sintéticos en memoria (Día 1)
        self.dataset_records: List[Dict] = []
        
        # Estructuras de datos con Pandas y Scikit-Learn (Día 2)
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Atributos reservados para los Días 3 y 4 (Modelo de Scikit-Learn)
        self.model = None
        self.is_trained = False
        self.accuracy = 0.0

    # --------------------------------------------------------------------------
    # MÉTODOS DEL DÍA 1: VECTORIZACIÓN, SIMULACIÓN Y GENERACIÓN SINTÉTICA
    # --------------------------------------------------------------------------

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
        """Convierte una coordenada (fila, col) al índice plano correspondiente (0 a 8)."""
        return fila * 3 + col

    @staticmethod
    def indice_a_coordenadas(indice: int) -> Tuple[int, int]:
        """Convierte un índice plano (0 a 8) a coordenadas matriciales (fila, col)."""
        return divmod(indice, 3)

    def simular_partida(
        self,
        prob_aleatorio_rival: float = 0.5,
        prob_aleatorio_ia: float = 0.2
    ) -> Dict:
        """
        Ejecuta una partida automatizada entre la IA y un rival simulado,
        recolectando en cada turno de la IA el par (estado_tablero, mejor_movimiento).
        """
        juego = TresEnRayaModel()
        turno = 'X'  # Convención tradicional: 'X' inicia la partida
        ejemplos_partida = []
        
        while True:
            movimientos_validos = juego.get_available_moves()
            if not movimientos_validos:
                ganador = 'Empate'
                break

            # Turno de la IA ('O'): Se recolecta la etiqueta supervisada
            if turno == self.ai_player:
                mejor_movimiento_minimax = self.supervisor_minimax.get_best_move(juego)
                
                if mejor_movimiento_minimax is not None:
                    vector_estado = self.vectorizar_tablero(juego.tablero)
                    indice_objetivo = self.coordenadas_a_indice(*mejor_movimiento_minimax)
                    
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

                if random.random() < prob_aleatorio_ia:
                    movimiento_a_jugar = random.choice(movimientos_validos)
                else:
                    movimiento_a_jugar = mejor_movimiento_minimax or random.choice(movimientos_validos)

                juego.make_move(movimiento_a_jugar, self.ai_player)

                if juego.check_winner(self.ai_player):
                    ganador = self.ai_player
                    break

                turno = self.rival_player

            # Turno del Rival ('X'): Simula humano (mezcla Minimax y aleatorio)
            else:
                if random.random() < prob_aleatorio_rival:
                    movimiento_rival = random.choice(movimientos_validos)
                else:
                    supervisor_rival = MinimaxAgent(ai_player=self.rival_player, human_player=self.ai_player)
                    movimiento_rival = supervisor_rival.get_best_move(juego) or random.choice(movimientos_validos)

                juego.make_move(movimiento_rival, self.rival_player)

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
        """
        print(f"[*] Iniciando simulación de {num_partidas} partidas automatizadas...")
        
        todos_los_ejemplos = []
        estadisticas = {'O': 0, 'X': 0, 'Empate': 0}

        for i in range(1, num_partidas + 1):
            if i % 3 == 0:
                prob_rival = 0.8
            elif i % 3 == 1:
                prob_rival = 0.4
            else:
                prob_rival = 0.0

            resultado = self.simular_partida(prob_aleatorio_rival=prob_rival)
            todos_los_ejemplos.extend(resultado['ejemplos'])
            estadisticas[resultado['ganador']] = estadisticas.get(resultado['ganador'], 0) + 1

            if i % 250 == 0 or i == num_partidas:
                print(f"    -> {i}/{num_partidas} partidas completadas. Ejemplos acumulados: {len(todos_los_ejemplos)}")

        if eliminar_duplicados:
            registros_unicos = {}
            for ejemplo in todos_los_ejemplos:
                clave_estado = tuple(ejemplo[f'c{idx}'] for idx in range(9))
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

        if ruta_salida_csv:
            self.guardar_csv(ruta_salida_csv)

        return datos_finales

    def guardar_csv(self, ruta_archivo: str) -> None:
        """Guarda los registros recolectados en formato CSV."""
        if not self.dataset_records:
            print("[!] Advertencia: No hay registros para guardar.")
            return

        directorio = os.path.dirname(ruta_archivo)
        if directorio and not os.path.exists(directorio):
            os.makedirs(directorio, exist_ok=True)

        columnas = self.COLUMNAS_CARACTERISTICAS + [self.COLUMNA_OBJETIVO]

        with open(ruta_archivo, mode='w', newline='', encoding='utf-8') as archivo_csv:
            escritor = csv.DictWriter(archivo_csv, fieldnames=columnas)
            escritor.writeheader()
            escritor.writerows(self.dataset_records)

        print(f"[+] Dataset exportado exitosamente a: {ruta_archivo} ({len(self.dataset_records)} filas)")

    # --------------------------------------------------------------------------
    # MÉTODOS DEL DÍA 2: LIMPIEZA, ESTRUCTURACIÓN CON PANDAS Y PREPARACIÓN (X, y)
    # --------------------------------------------------------------------------

    def cargar_y_limpiar_dataset(self, ruta_csv: str) -> pd.DataFrame:
        """
        Carga el archivo CSV en un DataFrame de Pandas y ejecuta la limpieza de datos:
        1. Verifica la existencia del archivo.
        2. Carga los datos en un DataFrame.
        3. Comprueba y elimina filas con valores nulos (NaN).
        4. Valida los tipos de datos enteros.
        5. Filtra duplicados asegurando coherencia del tablero.
        
        Args:
            ruta_csv (str): Ruta al archivo dataset_tres_en_raya.csv.
            
        Returns:
            pd.DataFrame: DataFrame limpio y validado.
        """
        if not os.path.exists(ruta_csv):
            raise FileNotFoundError(f"No se encontró el archivo de datos en: {ruta_csv}")

        # 1. Cargar archivo CSV usando Pandas
        df = pd.read_csv(ruta_csv)

        # 2. Validar que contenga las 10 columnas obligatorias (c0 a c8 y jugada_optima)
        columnas_esperadas = self.COLUMNAS_CARACTERISTICAS + [self.COLUMNA_OBJETIVO]
        columnas_faltantes = [col for col in columnas_esperadas if col not in df.columns]
        if columnas_faltantes:
            raise ValueError(f"El dataset no contiene las columnas esperadas: {columnas_faltantes}")

        filas_iniciales = len(df)

        # 3. Limpieza: Eliminar posibles valores nulos o faltantes
        df = df.dropna()

        # 4. Asegurar que todas las columnas sean de tipo entero (int)
        for col in columnas_esperadas:
            df[col] = df[col].astype(int)

        # 5. Validar rangos numéricos permitidos:
        #    - Casillas: únicamente -1, 0, o 1
        #    - Jugada óptima: índices de 0 a 8
        filtro_casillas = df[self.COLUMNAS_CARACTERISTICAS].isin([-1, 0, 1]).all(axis=1)
        filtro_objetivo = df[self.COLUMNA_OBJETIVO].between(0, 8)
        df = df[filtro_casillas & filtro_objetivo]

        # 6. Eliminar registros duplicados idénticos si existiesen
        df = df.drop_duplicates()

        self.df = df
        filas_limpias = len(self.df)

        print(f"[+] Dataset cargado y limpiado exitosamente:")
        print(f"    - Filas originales: {filas_iniciales}")
        print(f"    - Filas tras limpieza: {filas_limpias}")
        print(f"    - Valores nulos detectados: {filas_iniciales - filas_limpias}")

        return self.df

    def estructurar_caracteristicas_y_objetivo(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Formatea y separa el dataset en dos estructuras clave para Machine Learning:
        - X (Matriz de Características): Las 9 casillas de entrada (c0 a c8).
        - y (Vector Objetivo): La jugada óptima a predecir (0 a 8).
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Tupla con (X, y).
        """
        if self.df is None:
            raise ValueError("Primero debes cargar el dataset con 'cargar_y_limpiar_dataset'.")

        # X: Seleccionamos exclusivamente las 9 columnas de las casillas
        self.X = self.df[self.COLUMNAS_CARACTERISTICAS].copy()

        # y: Seleccionamos exclusivamente la columna con la decisión óptima
        self.y = self.df[self.COLUMNA_OBJETIVO].copy()

        print(f"[+] Estructuración de matrices completada:")
        print(f"    - Matriz de Características (X): {self.X.shape} (muestras x casillas)")
        print(f"    - Vector Objetivo (y):           {self.y.shape} (etiquetas)")

        return self.X, self.y

    def dividir_datos_entrenamiento(
        self,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide las matrices (X, y) en conjuntos de Entrenamiento y Prueba:
        - 80% Entrenamiento: Para que el Árbol de Decisión aprenda las reglas (Día 3).
        - 20% Prueba: Para verificar imparcialmente la precisión del modelo aprendido.
        
        Args:
            test_size (float): Porcentaje para el conjunto de prueba (por defecto 0.2 = 20%).
            random_state (int): Semilla para reproducibilidad de los resultados.
            
        Returns:
            Tuple con (X_train, X_test, y_train, y_test).
        """
        if self.X is None or self.y is None:
            self.estructurar_caracteristicas_y_objetivo()

        # Partición formal usando Scikit-Learn
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state
        )

        print(f"[+] Partición Train / Test realizada:")
        print(f"    - Muestras de Entrenamiento (Train): {len(self.X_train)} ({(1-test_size)*100:.0f}%)")
        print(f"    - Muestras de Evaluación (Test):     {len(self.X_test)} ({test_size*100:.0f}%)")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def obtener_resumen_dataset(self) -> Dict:
        """
        Retorna un resumen estadístico del dataset para auditoría y visualización.
        """
        if self.df is None:
            return {"estado": "Sin cargar"}

        distribucion_jugadas = self.df[self.COLUMNA_OBJETIVO].value_counts().sort_index().to_dict()

        return {
            "total_muestras": len(self.df),
            "total_columnas": len(self.df.columns),
            "caracteristicas": self.COLUMNAS_CARACTERISTICAS,
            "columna_objetivo": self.COLUMNA_OBJETIVO,
            "distribucion_jugadas": distribucion_jugadas,
            "muestras_train": len(self.X_train) if self.X_train is not None else 0,
            "muestras_test": len(self.X_test) if self.X_test is not None else 0
        }


# ------------------------------------------------------------------------------
# Script ejecutable directo para pruebas del Día 2
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 70)
    print("  SEMANA 3 - DÍA 2: LIMPIEZA Y ESTRUCTURACIÓN DEL DATASET (PANDAS & ML)")
    print("=" * 70)

    agente_ml = MLAgent()

    # Ruta estándar al archivo CSV generado en el Día 1
    ruta_dataset = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset_tres_en_raya.csv"
    )

    # Si no existiera el archivo, generarlo preventivamente
    if not os.path.exists(ruta_dataset):
        print("[!] El archivo CSV no existía. Generando automáticamente...")
        agente_ml.generar_dataset_sintetico(num_partidas=1000, ruta_salida_csv=ruta_dataset)

    # 1. Cargar y limpiar los datos con Pandas
    print("\n--- 1. Carga y Limpieza con Pandas ---")
    df_limpio = agente_ml.cargar_y_limpiar_dataset(ruta_dataset)

    print("\nPrimeras 5 filas del DataFrame:")
    print(df_limpio.head())

    # 2. Formatear y separar en X (Características) e y (Objetivo)
    print("\n--- 2. Estructuración de Matrices (X, y) ---")
    X, y = agente_ml.estructurar_caracteristicas_y_objetivo()
    print("\nEjemplo de vector de entrada X (Fila 0):")
    print(X.iloc[0].values)
    print(f"Jugada óptima supervisada y (Fila 0): Casilla {y.iloc[0]}")

    # 3. Dividir en conjuntos de Entrenamiento y Prueba
    print("\n--- 3. Partición de Datos (Train / Test) ---")
    agente_ml.dividir_datos_entrenamiento(test_size=0.2, random_state=42)

    # 4. Resumen final de auditoría
    print("\n--- 4. Resumen Estadístico de Distribución ---")
    resumen = agente_ml.obtener_resumen_dataset()
    print("Distribución de jugadas óptimas recomendadas por casilla (0 a 8):")
    for casilla, cantidad in resumen["distribucion_jugadas"].items():
        print(f"   Casilla {casilla}: {cantidad:3d} veces recomendada")

    print("\n[OK] Semana 3 - Día 2 completado exitosamente y listo para el Árbol de Decisión.")
