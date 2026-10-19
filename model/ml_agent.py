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
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

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


    # --------------------------------------------------------------------------
    # MÉTODOS DEL DÍA 3: ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN, REGLAS Y GRÁFICO
    # --------------------------------------------------------------------------

    def entrenar_arbol(
        self,
        criterion: str = 'entropy',
        max_depth: Optional[int] = None,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Entrena el modelo DecisionTreeClassifier con los datos de entrenamiento (X_train, y_train).
        Evalúa el porcentaje de acierto (Accuracy) en datos de entrenamiento y prueba.
        
        Args:
            criterion (str): Criterio de división ('entropy' para ganancia de información o 'gini').
            max_depth (Optional[int]): Profundidad máxima del árbol (None para sin límite).
            random_state (int): Semilla para garantizar reproducibilidad.
            
        Returns:
            Dict[str, float]: Diccionario con la precisión de entrenamiento y prueba.
        """
        if self.X_train is None or self.y_train is None:
            self.dividir_datos_entrenamiento()

        print("\n[*] Entrenando Árbol de Decisión (DecisionTreeClassifier)...")
        # Instanciamos el clasificador de árbol de decisión de Scikit-Learn
        self.model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            random_state=random_state
        )

        # Entrenamos el árbol pasándole las entradas (X_train) y las respuestas correctas (y_train)
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True

        # Evaluamos qué tan bien aprendió
        precision_train = float(self.model.score(self.X_train, self.y_train))
        precision_test = float(self.model.score(self.X_test, self.y_test))
        self.accuracy = precision_test

        print(f"[+] Entrenamiento completado con éxito:")
        print(f"    - Criterio de división:           {criterion}")
        print(f"    - Profundidad máxima alcanzada:   {self.model.get_depth()}")
        print(f"    - Cantidad total de hojas (nodos):{self.model.get_n_leaves()}")
        print(f"    - Precisión en Entrenamiento:     {precision_train * 100:.2f}%")
        print(f"    - Precisión en Prueba (Test):     {precision_test * 100:.2f}%")

        return {
            "precision_train": precision_train,
            "precision_test": precision_test,
            "profundidad": self.model.get_depth(),
            "hojas": self.model.get_n_leaves()
        }

    def predecir_jugada(self, tablero: List[List[str]]) -> Optional[Tuple[int, int]]:
        """
        Recibe el tablero 3x3 del juego actual y utiliza el Árbol de Decisión entrenado
        para inferir de forma instantánea la mejor coordenada (fila, col) a jugar.
        
        Args:
            tablero (List[List[str]]): Matriz 3x3 del estado actual del juego.
            
        Returns:
            Optional[Tuple[int, int]]: Coordenada recomendada por el árbol de decisión.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("El modelo aún no ha sido entrenado. Ejecuta 'entrenar_arbol()' primero.")

        # 1. Convertimos la matriz del tablero en DataFrame con las 9 características
        vector_tablero = pd.DataFrame(
            [self.vectorizar_tablero(tablero)],
            columns=self.COLUMNAS_CARACTERISTICAS
        )

        # 2. El árbol infiere la casilla óptima (0 a 8) de manera instantánea
        indice_predicho = int(self.model.predict(vector_tablero)[0])
        fila, col = self.indice_a_coordenadas(indice_predicho)

        # 3. Validación de seguridad: aseguramos que la casilla predicha esté vacía
        if tablero[fila][col] == ' ':
            return (fila, col)

        # Si por alguna eventualidad predice una casilla ocupada, elegimos la mejor disponible
        # según las probabilidades que calculó el árbol
        probabilidades = self.model.predict_proba(vector_tablero)[0]
        clases = list(self.model.classes_)
        indices_ordenados = sorted(range(len(probabilidades)), key=lambda k: probabilidades[k], reverse=True)

        for idx in indices_ordenados:
            casilla_candidata = clases[idx]
            f_cand, c_cand = self.indice_a_coordenadas(casilla_candidata)
            if tablero[f_cand][c_cand] == ' ':
                return (f_cand, c_cand)

        # En caso extremo, selecciona la primera casilla vacía
        for r in range(3):
            for c in range(3):
                if tablero[r][c] == ' ':
                    return (r, c)

        return None

    def exportar_reglas_texto(self, ruta_archivo: Optional[str] = None) -> str:
        """
        Extrae y muestra las reglas lógicas 'Si... Entonces...' aprendidas por el árbol
        mediante sklearn.tree.export_text (tal como solicita la guía del docente).
        
        Args:
            ruta_archivo (Optional[str]): Si se especifica, guarda el texto en un archivo .txt.
            
        Returns:
            str: Texto formateado con las reglas lógicas del árbol.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Debes entrenar el modelo antes de exportar sus reglas.")

        reglas = export_text(self.model, feature_names=self.COLUMNAS_CARACTERISTICAS)

        if ruta_archivo:
            with open(ruta_archivo, mode='w', encoding='utf-8') as f:
                f.write(reglas)
            print(f"[+] Reglas lógicas del Árbol de Decisión guardadas en: {ruta_archivo}")

        return reglas

    def exportar_arbol_grafico(self, ruta_imagen: str, max_depth: int = 3) -> None:
        """
        Genera y exporta una imagen gráfica (.png) de la estructura del árbol de decisión
        utilizando sklearn.tree.plot_tree y matplotlib.
        
        Args:
            ruta_imagen (str): Ruta completa donde se guardará la imagen (ej: arbol_decision.png).
            max_depth (int): Nivel de profundidad visible para mantener el diagrama legible.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Debes entrenar el modelo antes de graficar el árbol.")

        print(f"[*] Generando visualización gráfica del Árbol de Decisión...")
        plt.figure(figsize=(18, 10), dpi=200)

        nombres_clases = [f"Casilla {i}" for i in range(9)]
        plot_tree(
            self.model,
            max_depth=max_depth,
            feature_names=self.COLUMNAS_CARACTERISTICAS,
            class_names=[nombres_clases[c] for c in self.model.classes_],
            filled=True,
            rounded=True,
            fontsize=8
        )
        plt.title(
            f"Árbol de Decisión para Tres en Raya (Niveles iniciales 0 a {max_depth})",
            fontsize=14,
            weight='bold'
        )

        plt.savefig(ruta_imagen, bbox_inches='tight')
        plt.close()
        print(f"[+] Diagrama del Árbol de Decisión exportado exitosamente a: {ruta_imagen}")


# ------------------------------------------------------------------------------
# Script ejecutable directo para pruebas del Día 3
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 70)
    print("  SEMANA 3 - DÍA 3: ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN Y VISUALIZACIÓN")
    print("=" * 70)

    agente_ml = MLAgent()

    # Directorio base del proyecto
    dir_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_dataset = os.path.join(dir_proyecto, "dataset_tres_en_raya.csv")
    ruta_reglas = os.path.join(dir_proyecto, "reglas_arbol_decision.txt")
    ruta_grafico = os.path.join(dir_proyecto, "arbol_decision.png")

    # 1. Cargar y preparar datos (Día 2)
    agente_ml.cargar_y_limpiar_dataset(ruta_dataset)
    agente_ml.dividir_datos_entrenamiento(test_size=0.2, random_state=42)

    # 2. Entrenar el Árbol de Decisión (Día 3)
    metricas = agente_ml.entrenar_arbol(criterion='entropy')

    # 3. Exportar las reglas lógicas en texto legible
    print("\n--- Exportación de Reglas Lógicas (export_text) ---")
    reglas_texto = agente_ml.exportar_reglas_texto(ruta_archivo=ruta_reglas)
    lineas_reglas = reglas_texto.strip().split("\n")
    print(f"Mostrando las primeras 12 líneas de reglas aprendidas:")
    for linea in lineas_reglas[:12]:
        print(f"   {linea}")
    print("   ... (ver el archivo completo en reglas_arbol_decision.txt)")

    # 4. Exportar el árbol en formato gráfico (plot_tree)
    print("\n--- Exportación Gráfica del Árbol (plot_tree) ---")
    agente_ml.exportar_arbol_grafico(ruta_imagen=ruta_grafico, max_depth=3)

    # 5. Prueba de inferencia en tiempo real
    print("\n--- Prueba de Inferencia Rápida ---")
    tablero_prueba = [
        ['X', ' ', ' '],
        [' ', 'O', ' '],
        [' ', ' ', 'X']
    ]
    jugada_sugerida = agente_ml.predecir_jugada(tablero_prueba)
    print("Tablero de prueba:")
    for fila in tablero_prueba:
        print(f"   {fila}")
    print(f"-> El Árbol de Decisión predijo instantáneamente la jugada en: {jugada_sugerida}")

    print("\n[OK] Semana 3 - Día 3 completado exitosamente.")
