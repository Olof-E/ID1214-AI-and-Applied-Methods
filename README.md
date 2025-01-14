# ID1214 Main Project

This was the main project for the course ID1214 Artificial Intelligence and Applied Methods.

# Installation

Firstly, you need to have Python version 3.11.8 installed and it is also recommended that the projects packages are installed in a virtual python environment, as to avoid version collisions. Then to install all of the required packages that are listed in the file **requirments.txt**, run the following command:

    pip install -r requirements.txt

# Usage

To run the training program you can use the command:

    py agent.py

To test a pre-trained model use:

    py agent.py <path to model policy> <num of games>

To run just the Tetris environment and play it manually, run:

    py tetris.py

Manual Controls are:

- A - Move left
- D - Move right
- Q - Rotate clockwise
- E - Rotate counterclockwise
- S - Full drop piece
- SPACE - Keep piece (Save the piece to held slot)
- P - Pause/Unpause the game
- ESC - Quit the game
