import time
from model.minimax_agent import MinimaxAgent


class GameController:
    """
    Controlador del juego (Patrón MVC).
    Coordina los eventos del usuario, las respuestas de la IA Minimax,
    el cálculo de métricas de rendimiento y la sincronización con la vista.
    """

    def __init__(self, modelo, vista, agente_ia=None):
        # Guardamos el modelo (reglas) y la vista (interfaz)
        self.modelo = modelo
        self.vista = vista
        
        # Agente Minimax (juega con 'O' contra el humano 'X')
        self.agente_ia = agente_ia if agente_ia else MinimaxAgent(ai_player='O', human_player='X')
        
        # Variables de estado del juego
        self.turno_actual = 'X'
        self.modo_juego = "Humano vs Humano"
        self.juego_terminado = False

        # Conectamos las funciones del controlador con los eventos de la vista
        self.vista.set_click_callback(self.manejar_clic)
        self.vista.set_modo_callback(self.cambiar_modo)
        self.vista.set_reiniciar_callback(self.reiniciar_juego)

    def manejar_clic(self, fila: int, col: int):
        """Maneja el clic del usuario en una casilla del tablero."""
        if self.juego_terminado:
            return

        # 1. Movimiento del jugador humano
        movimiento_exitoso = self.modelo.hacer_movimiento(fila, col, self.turno_actual)
        if not movimiento_exitoso:
            return

        # Actualizamos el tablero en pantalla
        self.vista.actualizar_tablero(self.modelo.tablero)

        # 2. Verificamos si este movimiento termina el juego
        if self.modelo.verificar_victoria(self.turno_actual):
            self.vista.actualizar_mensaje(f"¡GANÓ EL JUGADOR {self.turno_actual}!")
            self.juego_terminado = True
            return

        elif self.modelo.es_empate():
            self.vista.actualizar_mensaje("¡EMPATE! Tablero lleno.")
            self.juego_terminado = True
            return

        # 3. Si no ha terminado el juego, gestionamos el turno
        if self.modo_juego == "Humano vs Humano":
            self.turno_actual = 'O' if self.turno_actual == 'X' else 'X'
            self.vista.actualizar_mensaje(f"Turno del Jugador: {self.turno_actual}")

        elif self.modo_juego == "Humano vs IA Minimax":
            # Turno de la IA Minimax
            self.turno_actual = 'O'
            self.vista.actualizar_mensaje("IA Minimax pensando...")
            self.vista.root.update()  # Forzamos a la ventana a actualizar el texto antes del cálculo

            # Semana 2 - Día 5: Medición de rendimiento (tiempo y nodos explorados)
            tiempo_inicio = time.time()
            mejor_movimiento = self.agente_ia.get_best_move(self.modelo)
            tiempo_transcurrido = time.time() - tiempo_inicio
            
            nodos_explorados = self.agente_ia.nodes_evaluated

            # Enviamos las métricas calculadas a la vista
            self.vista.actualizar_metricas(nodos_explorados, tiempo_transcurrido)

            if mejor_movimiento is not None:
                self.modelo.make_move(mejor_movimiento, self.agente_ia.ai_player)
                self.vista.actualizar_tablero(self.modelo.tablero)

                # Verificamos si la IA ganó o empató
                if self.modelo.check_winner(self.agente_ia.ai_player):
                    self.vista.actualizar_mensaje("¡GANÓ LA IA (O)!")
                    self.juego_terminado = True
                elif self.modelo.es_empate():
                    self.vista.actualizar_mensaje("¡EMPATE! Tablero lleno.")
                    self.juego_terminado = True
                else:
                    self.turno_actual = 'X'
                    self.vista.actualizar_mensaje(f"Turno del Jugador: {self.turno_actual}")

    def cambiar_modo(self, nuevo_modo: str):
        """Cambia el modo de juego y reinicia la partida."""
        self.modo_juego = nuevo_modo
        self.reiniciar_juego()

    def reiniciar_juego(self):
        """Reinicia el tablero y limpia las métricas para una nueva partida."""
        self.modelo.reiniciar()
        self.turno_actual = 'X'
        self.juego_terminado = False
        self.vista.actualizar_tablero(self.modelo.tablero)
        self.vista.actualizar_mensaje(f"Turno del Jugador: {self.turno_actual}")
        self.vista.actualizar_metricas(0, 0.0)
