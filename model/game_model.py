"""Modelo del Tres en Raya: tablero 3x3 y reglas del juego.

Los nombres make_move, undo_move, get_available_moves y check_winner
se mantienen en inglés porque el agente Minimax del PDF los utiliza.
"""


class TresEnRayaModel:
    """Estado del tablero y reglas. No conoce la interfaz gráfica."""

    def __init__(self):
        self.tablero = []
        self.reiniciar()

    def reiniciar(self):
        """Deja el tablero vacío (' ' = casilla libre)."""
        self.tablero = [[' '] * 3 for _ in range(3)]

    def get_available_moves(self):
        """Devuelve las casillas libres como lista de (fila, col)."""
        return [
            (fila, col)
            for fila in range(3)
            for col in range(3)
            if self.tablero[fila][col] == ' '
        ]

    def make_move(self, movimiento, jugador):
        """Pone la ficha del jugador en (fila, col) si está libre.

        Devuelve True si la jugada fue válida, False si estaba ocupada.
        """
        fila, col = movimiento
        if self.tablero[fila][col] != ' ':
            return False
        self.tablero[fila][col] = jugador
        return True

    def undo_move(self, movimiento):
        """Backtracking: vacía la casilla para deshacer una jugada."""
        fila, col = movimiento
        self.tablero[fila][col] = ' '

    def check_winner(self, jugador):
        """True si el jugador completó una fila, columna o diagonal."""
        t = self.tablero
        lineas = (
            [t[i] for i in range(3)]                          # filas
            + [[t[f][i] for f in range(3)] for i in range(3)]  # columnas
            + [[t[i][i] for i in range(3)]]                   # diagonal \
            + [[t[i][2 - i] for i in range(3)]]               # diagonal /
        )
        return any(linea == [jugador] * 3 for linea in lineas)

    def es_empate(self):
        """True si el tablero está lleno."""
        return not self.get_available_moves()
