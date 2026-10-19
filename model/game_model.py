"""
Módulo que contiene la lógica principal del juego Tres en Raya (Modelo).
Sigue el patrón de arquitectura MVC.
"""


class TresEnRayaModel:
    """
    Representa el estado y las reglas del juego Tres en Raya.
    El tablero se define como una matriz de 3x3.
    """

    def __init__(self):
        """Inicializa un tablero vacío de 3x3."""
        # Se usa ' ' para casilla vacía, 'X' o 'O' para las jugadas.
        self.tablero = [
            [' ', ' ', ' '],
            [' ', ' ', ' '],
            [' ', ' ', ' ']
        ]

    def reiniciar(self) -> None:
        """Limpia el tablero para iniciar una nueva partida."""
        self.tablero = [
            [' ', ' ', ' '],
            [' ', ' ', ' '],
            [' ', ' ', ' ']
        ]

    def obtener_movimientos_validos(self) -> list:
        """
        Devuelve una lista con las coordenadas de las casillas vacías.
        
        Returns:
            list: Lista de tuplas (fila, columna) que representan posiciones válidas.
        """
        movimientos = []
        for fila in range(3):
            for col in range(3):
                if self.tablero[fila][col] == ' ':
                    movimientos.append((fila, col))
        return movimientos

    def hacer_movimiento(self, fila: int, col: int, jugador: str) -> bool:
        """
        Registra la ficha del jugador en la posición indicada si está vacía.
        
        Args:
            fila (int): Índice de la fila (0, 1 o 2).
            col (int): Índice de la columna (0, 1 o 2).
            jugador (str): 'X' o 'O'.
            
        Returns:
            bool: True si el movimiento fue exitoso, False si la casilla estaba ocupada.
        """
        if self.tablero[fila][col] == ' ':
            self.tablero[fila][col] = jugador
            return True
        return False

    def deshacer_movimiento(self, fila: int, col: int) -> None:
        """
        Deshace un movimiento en el tablero (vuelve a dejar la casilla vacía).
        Mecanismo clave de Backtracking: permite a la IA retroceder en el árbol.
        
        Args:
            fila (int): Índice de la fila.
            col (int): Índice de la columna.
        """
        self.tablero[fila][col] = ' '

    def verificar_victoria(self, jugador: str) -> bool:
        """
        Comprueba si el jugador especificado ha ganado la partida.
        
        Args:
            jugador (str): 'X' o 'O'.
            
        Returns:
            bool: True si el jugador tiene 3 en raya, False en caso contrario.
        """
        # 1. Comprobar las 3 filas y las 3 columnas
        for i in range(3):
            # Filas
            if (self.tablero[i][0] == jugador and 
                self.tablero[i][1] == jugador and 
                self.tablero[i][2] == jugador):
                return True
            
            # Columnas
            if (self.tablero[0][i] == jugador and 
                self.tablero[1][i] == jugador and 
                self.tablero[2][i] == jugador):
                return True

        # 2. Comprobar la diagonal principal (\)
        if (self.tablero[0][0] == jugador and 
            self.tablero[1][1] == jugador and 
            self.tablero[2][2] == jugador):
            return True

        # 3. Comprobar la diagonal secundaria (/)
        if (self.tablero[0][2] == jugador and 
            self.tablero[1][1] == jugador and 
            self.tablero[2][0] == jugador):
            return True

        return False

    def es_empate(self) -> bool:
        """
        Verifica si el juego terminó en empate.
        
        Returns:
            bool: True si el tablero está lleno, False si aún hay espacio.
        """
        return len(self.obtener_movimientos_validos()) == 0

    # ------------------------------------------------------------------
    # Métodos de compatibilidad para el Agente Minimax (Semana 2)
    # Permiten pasar 'move' como tupla (fila, col) exactamente como en el PDF
    # ------------------------------------------------------------------

    def make_move(self, move: tuple, player: str) -> bool:
        """Aplica un movimiento en el tablero (recibe tupla de coordenadas)."""
        fila, col = move
        return self.hacer_movimiento(fila, col, player)

    def undo_move(self, move: tuple) -> None:
        """
        Mecanismo de Backtracking: deshace un movimiento aplicado.
        Restaura el estado previo del tablero para explorar otras ramas.
        """
        fila, col = move
        self.deshacer_movimiento(fila, col)

    def get_available_moves(self) -> list:
        """Retorna las casillas disponibles para explorar jugadas."""
        return self.obtener_movimientos_validos()

    def check_winner(self, player: str) -> bool:
        """Verifica si el jugador ganó (para evaluar estados terminales)."""
        return self.verificar_victoria(player)
