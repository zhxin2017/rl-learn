
import torch
import agent
import board
import model


class Game:
    def __init__(self, agent=None, next_turn='red', me_color='black'):
        self.board = board.Board(next_turn, me_color=me_color)
        self.agent = agent
        self.turn = 0


    def agent_step(self):
        (src_row, src_col, dst_row, dst_col), prob = self.agent.exploit(self.board)
        self.board.move(src_row, src_col, dst_row, dst_col)
        return prob

    def person_step(self):
        move_str = input('input move str:\n')
        src_row, src_col, dst_row, dst_col = (int(s) for s in move_str)
        self.board.move(src_row, src_col, dst_row, dst_col)

    def play(self, mode='aa'):
        while True:
            print('-------------------')
            if mode[0] == 'a':
                prob = self.agent_step()
                self.board.show_board()
                print(f'win prob {prob:.4f}')
                result = self.board.get_result()
            else:
                self.person_step()
                self.board.show_board()
                result = self.board.get_result()
            print('-------------------')
            if mode[1] == 'a':
                prob = self.agent_step()
                self.board.show_board()
                print(f'win prob {prob:.4f}')
                result = self.board.get_result()
            else:
                self.person_step()
                self.board.show_board()
                result = self.board.get_result()
            if result != 'going':
                print(result)
                break


if __name__ == '__main__':
    # first_turn = int(sys.argv[1])
    model_ = model.Evaluator(20, 512)
    device = torch.device('mps')
    model_.load_state_dict(torch.load('resources/evaluator.1.pt', map_location=device))
    model_.to(device)
    agent_ = agent.Agent(model_, device=device)
    game = Game(agent_, next_turn='red', me_color='black')
    game.board.show_board()
    game.play(mode='aa')