"""Agente de Machine Learning para el Tres en Raya (Semana 3).

Flujo: Minimax juega partidas simuladas y etiqueta la mejor jugada de cada
tablero -> se guarda un dataset CSV -> se entrena un DecisionTreeClassifier
que luego predice la jugada sin recorrer el árbol de juego.

Para regenerar reglas y gráfico del árbol, desde la raíz del proyecto:
    python -m model.ml_agent
"""

import csv
import os
import random

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from model.game_model import TresEnRayaModel
from model.minimax_agent import MinimaxAgent


class MLAgent:
    """IA que aprende de Minimax mediante un árbol de decisión."""

    # Valor numérico de cada ficha en las características del modelo.
    VALOR_FICHA = {' ': 0, 'X': 1, 'O': -1}
    # Las 9 casillas del tablero (entrada) y la jugada a predecir (salida).
    COLUMNAS_CASILLAS = [f'c{i}' for i in range(9)]
    COLUMNA_JUGADA = 'jugada_optima'

    def __init__(self, ai_player='O', human_player='X'):
        self.ai_player = ai_player
        self.human_player = human_player
        # Minimax que etiqueta jugadas: uno para la IA y otro para el rival.
        self.minimax_ia = MinimaxAgent(ai_player, human_player)
        self.minimax_rival = MinimaxAgent(human_player, ai_player)

        self.registros = []        # ejemplos generados por la simulación
        self.datos = None          # DataFrame limpio
        self.x_entrenamiento = None
        self.x_prueba = None
        self.y_entrenamiento = None
        self.y_prueba = None
        self.modelo = None         # DecisionTreeClassifier
        self.is_trained = False

    # ------------------------------------------------------------------
    # Conversión tablero <-> datos
    # ------------------------------------------------------------------

    @classmethod
    def vectorizar_tablero(cls, tablero):
        """Convierte el tablero 3x3 en una lista de 9 números."""
        return [cls.VALOR_FICHA[casilla] for fila in tablero
                for casilla in fila]

    @staticmethod
    def coordenadas_a_indice(fila, col):
        """(fila, col) -> índice 0..8."""
        return fila * 3 + col

    @staticmethod
    def indice_a_coordenadas(indice):
        """Índice 0..8 -> (fila, col)."""
        return divmod(indice, 3)

    # ------------------------------------------------------------------
    # Generación del dataset (simulación)
    # ------------------------------------------------------------------

    def simular_partida(self, prob_rival_aleatorio=0.5, prob_ia_aleatoria=0.2):
        """Simula una partida y devuelve (ganador, ejemplos).

        En cada turno de la IA guarda (tablero, mejor jugada según Minimax).
        Parte de jugadas aleatorias para visitar tableros variados.
        """
        juego = TresEnRayaModel()
        turno = self.human_player  # X empieza
        ejemplos = []

        while True:
            libres = juego.get_available_moves()
            if not libres:
                return 'Empate', ejemplos

            if turno == self.ai_player:
                mejor = self.minimax_ia.get_best_move(juego)
                estado = self.vectorizar_tablero(juego.tablero)
                ejemplo = dict(zip(self.COLUMNAS_CASILLAS, estado))
                ejemplo[self.COLUMNA_JUGADA] = (
                    self.coordenadas_a_indice(*mejor))
                ejemplos.append(ejemplo)

                if random.random() < prob_ia_aleatoria:
                    mejor = random.choice(libres)
                juego.make_move(mejor, self.ai_player)
                if juego.check_winner(self.ai_player):
                    return self.ai_player, ejemplos
                turno = self.human_player
            else:
                if random.random() < prob_rival_aleatorio:
                    jugada = random.choice(libres)
                else:
                    jugada = self.minimax_rival.get_best_move(juego)
                juego.make_move(jugada, self.human_player)
                if juego.check_winner(self.human_player):
                    return self.human_player, ejemplos
                turno = self.ai_player

    def generar_dataset_sintetico(self, num_partidas=1000,
                                  ruta_salida_csv=None):
        """Simula num_partidas, quita tableros repetidos y guarda el CSV."""
        print(f"Simulando {num_partidas} partidas...")
        todos = []
        resultados = {self.ai_player: 0, self.human_player: 0, 'Empate': 0}
        # Probabilidad de que el rival juegue al azar, según la partida:
        # muy aleatorio, medio aleatorio y óptimo (se repite cada 3).
        prob_por_ciclo = (0.8, 0.4, 0.0)

        for numero in range(1, num_partidas + 1):
            prob_rival = prob_por_ciclo[numero % 3]
            ganador, ejemplos = self.simular_partida(prob_rival)
            todos.extend(ejemplos)
            resultados[ganador] += 1
            if numero % 250 == 0:
                print(f"  {numero}/{num_partidas} partidas...")

        # Un tablero repetido se queda una sola vez con su jugada óptima.
        unicos = {}
        for ejemplo in todos:
            estado = tuple(ejemplo[c] for c in self.COLUMNAS_CASILLAS)
            unicos[estado] = ejemplo
        self.registros = list(unicos.values())

        print(f"Resultados (IA/Humano/Empate): {resultados}")
        print(f"Ejemplos brutos: {len(todos)} | únicos: {len(self.registros)}")

        if ruta_salida_csv:
            self.guardar_csv(ruta_salida_csv)
        return self.registros

    def guardar_csv(self, ruta_archivo):
        """Guarda los registros generados en un archivo CSV."""
        columnas = self.COLUMNAS_CASILLAS + [self.COLUMNA_JUGADA]
        with open(ruta_archivo, 'w', newline='', encoding='utf-8') as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=columnas)
            escritor.writeheader()
            escritor.writerows(self.registros)
        print(f"Dataset guardado en {ruta_archivo} "
              f"({len(self.registros)} filas)")

    # ------------------------------------------------------------------
    # Preparación de datos
    # ------------------------------------------------------------------

    def cargar_y_limpiar_dataset(self, ruta_csv):
        """Lee el CSV con Pandas y descarta filas inválidas o repetidas."""
        if not os.path.exists(ruta_csv):
            raise FileNotFoundError(f"No existe el dataset: {ruta_csv}")

        datos = pd.read_csv(ruta_csv)
        columnas = self.COLUMNAS_CASILLAS + [self.COLUMNA_JUGADA]
        faltantes = [c for c in columnas if c not in datos.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas en el dataset: {faltantes}")

        filas_iniciales = len(datos)
        datos = datos.dropna()
        datos[columnas] = datos[columnas].astype(int)

        casillas_validas = datos[self.COLUMNAS_CASILLAS].isin(
            [-1, 0, 1]).all(axis=1)
        jugada_valida = datos[self.COLUMNA_JUGADA].between(0, 8)
        self.datos = datos[casillas_validas & jugada_valida].drop_duplicates()

        print(f"Dataset cargado: {filas_iniciales} filas -> "
              f"{len(self.datos)} tras la limpieza")
        return self.datos

    def dividir_datos_entrenamiento(self, test_size=0.2, random_state=42):
        """Separa características (9 casillas) y objetivo (jugada).

        Reparte 80% entrenamiento y 20% prueba.
        """
        caracteristicas = self.datos[self.COLUMNAS_CASILLAS]
        jugadas = self.datos[self.COLUMNA_JUGADA]
        (self.x_entrenamiento, self.x_prueba,
         self.y_entrenamiento, self.y_prueba) = train_test_split(
            caracteristicas, jugadas,
            test_size=test_size, random_state=random_state)
        print(f"Entrenamiento: {len(self.x_entrenamiento)} | "
              f"Prueba: {len(self.x_prueba)}")

    # ------------------------------------------------------------------
    # Entrenamiento e inferencia
    # ------------------------------------------------------------------

    def entrenar_arbol(self, criterion='entropy', max_depth=None,
                       random_state=42):
        """Entrena el DecisionTreeClassifier y devuelve su precisión."""
        if self.x_entrenamiento is None:
            self.dividir_datos_entrenamiento()

        self.modelo = DecisionTreeClassifier(
            criterion=criterion, max_depth=max_depth,
            random_state=random_state)
        self.modelo.fit(self.x_entrenamiento, self.y_entrenamiento)
        self.is_trained = True

        resultado = {
            'precision_train': self.modelo.score(
                self.x_entrenamiento, self.y_entrenamiento),
            'precision_test': self.modelo.score(
                self.x_prueba, self.y_prueba),
            'profundidad': self.modelo.get_depth(),
            'hojas': self.modelo.get_n_leaves(),
        }
        print(f"Árbol entrenado: profundidad {resultado['profundidad']}, "
              f"precisión train {resultado['precision_train']:.1%}, "
              f"test {resultado['precision_test']:.1%}")
        return resultado

    def predecir_jugada(self, tablero):
        """Predice la jugada (fila, col) para el tablero dado.

        Si el árbol sugiere una casilla ocupada, usa la libre con mayor
        probabilidad. Devuelve None si el tablero está lleno.
        """
        if not self.is_trained:
            raise ValueError("Primero ejecuta entrenar_arbol().")

        entrada = pd.DataFrame([self.vectorizar_tablero(tablero)],
                               columns=self.COLUMNAS_CASILLAS)
        probabilidades = self.modelo.predict_proba(entrada)[0]
        # Casillas de la más a la menos probable (empate: la menor).
        candidatas = sorted(zip(probabilidades, self.modelo.classes_),
                            key=lambda par: (-par[0], par[1]))
        for _, indice in candidatas:
            fila, col = self.indice_a_coordenadas(int(indice))
            if tablero[fila][col] == ' ':
                return fila, col

        # Ninguna clase aprendida sirve: primera casilla libre.
        for fila in range(3):
            for col in range(3):
                if tablero[fila][col] == ' ':
                    return fila, col
        return None

    # ------------------------------------------------------------------
    # Visualización del árbol aprendido
    # ------------------------------------------------------------------

    def exportar_reglas_texto(self, ruta_archivo=None):
        """Devuelve (y opcionalmente guarda) las reglas del árbol."""
        reglas = export_text(self.modelo,
                             feature_names=self.COLUMNAS_CASILLAS)
        if ruta_archivo:
            with open(ruta_archivo, 'w', encoding='utf-8') as archivo:
                archivo.write(reglas)
        return reglas

    def exportar_arbol_grafico(self, ruta_imagen, max_depth=3):
        """Guarda una imagen PNG con los primeros niveles del árbol."""
        plt.figure(figsize=(18, 10), dpi=200)
        plot_tree(
            self.modelo,
            max_depth=max_depth,
            feature_names=self.COLUMNAS_CASILLAS,
            class_names=[f"Casilla {c}" for c in self.modelo.classes_],
            filled=True, rounded=True, fontsize=8)
        plt.title(f"Árbol de Decisión para Tres en Raya "
                  f"(Niveles iniciales 0 a {max_depth})",
                  fontsize=14, weight='bold')
        plt.savefig(ruta_imagen, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    # Entrena con el dataset existente y exporta reglas y gráfico.
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agente = MLAgent()
    agente.cargar_y_limpiar_dataset(
        os.path.join(raiz, 'dataset_tres_en_raya.csv'))
    agente.entrenar_arbol()
    agente.exportar_reglas_texto(
        os.path.join(raiz, 'reglas_arbol_decision.txt'))
    agente.exportar_arbol_grafico(os.path.join(raiz, 'arbol_decision.png'))
