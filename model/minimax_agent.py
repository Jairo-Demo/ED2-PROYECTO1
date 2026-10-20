"""Agente Minimax con Backtracking para el Tres en Raya.

Explora el árbol de jugadas (implícito, mediante recursión) y elige la
mejor jugada para la IA según el nivel de dificultad seleccionado.
"""

import random


class MinimaxAgent:
    """IA simbólica: busca exhaustivamente en el árbol de juego."""

    def __init__(self, ai_player='O', human_player='X'):
        self.ai_player = ai_player
        self.human_player = human_player
        # Cuántos nodos del árbol visitó la última búsqueda.
        self.nodes_evaluated = 0

    def evaluate(self, board):
        """Valor de un estado final: +10 gana IA, -10 gana humano, 0 resto."""
        if board.check_winner(self.ai_player):
            return 10
        if board.check_winner(self.human_player):
            return -10
        return 0

    def minimax(self, board, depth, is_maximizing, max_depth=None):
        """Devuelve el mejor puntaje alcanzable desde el tablero actual.

        depth: jugadas simuladas hasta aquí (premia ganar rápido).
        is_maximizing: True si juega la IA (MAX), False si el humano (MIN).
        max_depth: límite de niveles a explorar en el árbol (None = sin límite).
        """
        self.nodes_evaluated += 1
        score = self.evaluate(board)

        # Casos base: hojas del árbol (victoria o empate).
        if score == 10:
            return score - depth
        if score == -10:
            return score + depth
        if not board.get_available_moves():
            return 0

        # Límite de profundidad (poda de árbol)
        if max_depth is not None and depth >= max_depth:
            return 0

        if is_maximizing:
            best_score = -float('inf')
            for move in board.get_available_moves():
                board.make_move(move, self.ai_player)
                current_score = self.minimax(board, depth + 1, False, max_depth)
                board.undo_move(move)  # BACKTRACKING
                best_score = max(current_score, best_score)
            return best_score

        best_score = float('inf')
        for move in board.get_available_moves():
            board.make_move(move, self.human_player)
            current_score = self.minimax(board, depth + 1, True, max_depth)
            board.undo_move(move)  # BACKTRACKING
            best_score = min(current_score, best_score)
        return best_score

    def get_best_move(self, board, dificultad="Difícil"):
        """Devuelve la jugada (fila, col) según el nivel de dificultad.

        Niveles:
        - Fácil: 60% probabilidad de jugada al azar (comete fallos evidentes).
        - Medio: 20% probabilidad de fallo (juega bien pero se le puede ganar).
        - Difícil: Minimax completo al 100% (imbatible).
        """
        self.nodes_evaluated = 0
        movimientos_validos = board.get_available_moves()
        if not movimientos_validos:
            return None

        # Nivel Fácil: se equivoca frecuentemente jugando al azar
        if "Fácil" in dificultad and random.random() < 0.60:
            self.nodes_evaluated = 1
            return random.choice(movimientos_validos)

        # Nivel Medio: juega con buena táctica pero comete errores ocasionales
        if "Medio" in dificultad and random.random() < 0.20:
            self.nodes_evaluated = 1
            return random.choice(movimientos_validos)

        # Nivel Difícil / Imbatible: Minimax exhaustivo óptimo
        best_score = -float('inf')
        best_move = None

        for move in movimientos_validos:
            board.make_move(move, self.ai_player)
            score = self.minimax(board, 0, False)
            board.undo_move(move)  # BACKTRACKING

            if score > best_score:
                best_score = score
                best_move = move

        return best_move
