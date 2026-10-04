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
        Deshace un movimiento en el tablero (lo vuelve a dejar vacío).
        Nota pedagógica: Fundamental para implementar el algoritmo de Backtracking en la Semana 2.
        
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
