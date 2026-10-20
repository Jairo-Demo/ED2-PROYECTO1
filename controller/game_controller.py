"""Controlador: conecta la vista con el modelo y los agentes de IA.

Recibe los eventos de la vista, aplica las jugadas en el modelo, pide su
jugada a la IA según el modo elegido y actualiza la pantalla.
"""

import time

MODO_HUMANO = "Humano vs Humano"
MODO_MINIMAX = "Humano vs IA Minimax"
MODO_ML = "Humano vs IA Machine Learning"

# X siempre empieza; O es el segundo jugador o la IA.
FICHA_X = 'X'
FICHA_O = 'O'


class GameController:
    """Coordina el flujo de turnos y las métricas de cada IA."""

    def __init__(self, modelo, vista, agente_minimax, agente_ml):
        self.modelo = modelo
        self.vista = vista
        self.agente_minimax = agente_minimax
        self.agente_ml = agente_ml

        self.turno_actual = FICHA_X
        self.modo_juego = MODO_HUMANO
        self.dificultad = "Difícil"
        self.juego_terminado = False

        # La vista avisa al controlador a través de estas funciones.
        self.vista.set_click_callback(self.manejar_clic)
        self.vista.set_modo_callback(self.cambiar_modo)
        self.vista.set_dificultad_callback(self.cambiar_dificultad)
        self.vista.set_reiniciar_callback(self.reiniciar_juego)

    def manejar_clic(self, fila, col):
        """Procesa el clic en una casilla y, si toca, el turno de la IA."""
        if self.juego_terminado:
            return

        if not self.modelo.make_move((fila, col), self.turno_actual):
            return  # casilla ocupada
        self.vista.actualizar_tablero(self.modelo.tablero)

        if self._terminar_si_corresponde(
                f"¡GANÓ EL JUGADOR {self.turno_actual}!"):
            return

        if self.modo_juego == MODO_HUMANO:
            siguiente = FICHA_X
            if self.turno_actual == FICHA_X:
                siguiente = FICHA_O
            self._pasar_turno(siguiente)
        else:
            self._jugar_turno_ia()

    def _jugar_turno_ia(self):
        """La IA del modo actual elige y aplica su jugada."""
        if self.modo_juego == MODO_MINIMAX:
            nombre, mensaje_victoria = "IA Minimax", "¡GANÓ LA IA (O)!"
        else:
            nombre, mensaje_victoria = "IA ML", "¡GANÓ LA IA ML (O)!"

        self.turno_actual = FICHA_O
        self.vista.actualizar_mensaje(f"{nombre} pensando...")
        self.vista.refrescar()

        inicio = time.perf_counter()
        if self.modo_juego == MODO_MINIMAX:
            jugada = self.agente_minimax.get_best_move(
                self.modelo, dificultad=self.dificultad)
            nodos_explorados = self.agente_minimax.nodes_evaluated
        else:
            # El árbol de decisión no explora ramas futuras: 0 nodos.
            jugada = self.agente_ml.predecir_jugada(self.modelo.tablero)
            nodos_explorados = 0
        tiempo_transcurrido = time.perf_counter() - inicio

        self.vista.actualizar_metricas(nodos_explorados, tiempo_transcurrido)
        self.modelo.make_move(jugada, FICHA_O)
        self.vista.actualizar_tablero(self.modelo.tablero)

        if not self._terminar_si_corresponde(mensaje_victoria):
            self._pasar_turno(FICHA_X)

    def _terminar_si_corresponde(self, mensaje_victoria):
        """Si el turno actual ganó o hay empate, termina la partida."""
        if self.modelo.check_winner(self.turno_actual):
            mensaje = mensaje_victoria
        elif self.modelo.es_empate():
            mensaje = "¡EMPATE! Tablero lleno."
        else:
            return False
        self.vista.actualizar_mensaje(mensaje)
        self.juego_terminado = True
        return True

    def _pasar_turno(self, ficha):
        """Cede el turno a la ficha indicada y lo muestra."""
        self.turno_actual = ficha
        self.vista.actualizar_mensaje(f"Turno del Jugador: {ficha}")

    def cambiar_modo(self, nuevo_modo):
        """Cambia el modo de juego y reinicia la partida."""
        self.modo_juego = nuevo_modo
        self.reiniciar_juego()

    def cambiar_dificultad(self, nueva_dificultad):
        """Cambia el nivel de dificultad de la IA y reinicia la partida."""
        self.dificultad = nueva_dificultad
        self.reiniciar_juego()

    def reiniciar_juego(self):
        """Vacía el tablero y deja las métricas en cero."""
        self.modelo.reiniciar()
        self.turno_actual = FICHA_X
        self.juego_terminado = False
        self.vista.actualizar_tablero(self.modelo.tablero)
        self.vista.actualizar_mensaje(f"Turno del Jugador: {FICHA_X}")
        self.vista.actualizar_metricas(0, 0.0)
