"""Agente Minimax con Backtracking para el Tres en Raya.

Explora el árbol de jugadas (implícito, mediante recursión) y elige la
mejor jugada para la IA suponiendo que el rival también juega óptimo.
"""


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

    def minimax(self, board, depth, is_maximizing):
        """Devuelve el mejor puntaje alcanzable desde el tablero actual.

        depth: jugadas simuladas hasta aquí (premia ganar rápido).
        is_maximizing: True si juega la IA (MAX), False si el humano (MIN).
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

        if is_maximizing:
            best_score = -float('inf')
            for move in board.get_available_moves():
                board.make_move(move, self.ai_player)
                current_score = self.minimax(board, depth + 1, False)
                board.undo_move(move)  # BACKTRACKING
                best_score = max(current_score, best_score)
            return best_score

        best_score = float('inf')
        for move in board.get_available_moves():
            board.make_move(move, self.human_player)
            current_score = self.minimax(board, depth + 1, True)
            board.undo_move(move)  # BACKTRACKING
            best_score = min(current_score, best_score)
        return best_score

    def get_best_move(self, board):
        """Devuelve la mejor jugada (fila, col) de la IA en el tablero."""
        self.nodes_evaluated = 0
        best_score = -float('inf')
        best_move = None

        for move in board.get_available_moves():
            board.make_move(move, self.ai_player)
            score = self.minimax(board, 0, False)  # sigue el humano
            board.undo_move(move)  # BACKTRACKING

            if score > best_score:
                best_score = score
                best_move = move

        return best_move
