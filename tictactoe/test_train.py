from .agent import Bot
from .board import Board
from tqdm import tqdm

def run_train():
    game = Board(size=3)
    bot1 = Bot()
    bot2 = Bot()

    print("Start training..")

    bot1.train_mode(0.7)
    bot2.train_mode(0.7)
    for i in tqdm(range(10000)):
        game.run(bot1, bot2, verbose=False)
        if (i+1)%100 == 0:
            bot1.learning_rate *= 0.95
            bot2.learning_rate *= 0.95
        
    bot1.train_mode(0.5)
    bot2.train_mode(0.5)
    for _ in tqdm(range(10000)):
        game.run(bot1, bot2, verbose=False)

    print("End training..")
    print("Play dummy game..")

    bot1.fix_model()
    bot2.fix_model()

    bot1.serious_mode()
    bot2.serious_mode()

    game.run(bot1, bot2)

def test_train():
    run_train()