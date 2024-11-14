# relative positions of the blocks within a 4x4 matrix of each block
# (0, 0) is the top left corner of the matrxi

from Vector2 import Vector2

tBlock = [
  [
    Vector2(0, 2),
    Vector2(1, 2),
    Vector2(2, 2),
    Vector2(1, 3),
  ],
  [
    Vector2(0, 2),
    Vector2(1, 1),
    Vector2(1, 2),
    Vector2(1, 3),
  ],
  [
    Vector2(0, 3),
    Vector2(1, 2),
    Vector2(1, 3),
    Vector2(2, 3),
  ],
  [
    Vector2(1, 1),
    Vector2(1, 2),
    Vector2(1, 3),
    Vector2(2, 2),
  ],
]

iBlock = [
  [
    Vector2(0, 2),
    Vector2(1, 2),
    Vector2(2, 2),
    Vector2(3, 2),
  ],
  [
    Vector2(2, 0),
    Vector2(2, 1),
    Vector2(2, 2),
    Vector2(2, 3),  
  ],
  [
    Vector2(0, 2),
    Vector2(1, 2),
    Vector2(2, 2),
    Vector2(3, 2), 
  ],
  [
    Vector2(1, 0),
    Vector2(1, 1),
    Vector2(1, 2),
    Vector2(1, 3),  
  ],
]

lBlock = [
  [
   Vector2(0, 2),
   Vector2(1, 2), 
   Vector2(2, 2), 
   Vector2(0, 3),  
  ],
  [
   Vector2(0, 1),
   Vector2(1, 1), 
   Vector2(1, 2), 
   Vector2(1, 3),  
  ],
  [
   Vector2(0, 3),
   Vector2(1, 3), 
   Vector2(2, 3), 
   Vector2(2, 2),  
  ],
  [
   Vector2(1, 1),
   Vector2(1, 2), 
   Vector2(1, 3), 
   Vector2(2, 3),  
  ],
]

sBlock = [
  [
    Vector2(0, 2),
    Vector2(1, 2),
    Vector2(1, 3),
    Vector2(2, 3),
  ],
  [
    Vector2(2, 1),
    Vector2(2, 2),
    Vector2(1, 2),
    Vector2(1, 3),
  ],
  [
    Vector2(0, 2),
    Vector2(1, 2),
    Vector2(1, 3),
    Vector2(2, 3),
  ],
  [
    Vector2(1, 1),
    Vector2(1, 2),
    Vector2(0, 2),
    Vector2(0, 3),
  ],
]

zBlock = [
  [
    Vector2(0, 3),
    Vector2(1, 3),
    Vector2(1, 2),
    Vector2(2, 2),
  ],
  [
    Vector2(1, 1),
    Vector2(1, 2),
    Vector2(2, 2),
    Vector2(2, 3),
  ],
  [
    Vector2(0, 3),
    Vector2(1, 3),
    Vector2(1, 2),
    Vector2(2, 2),
  ],
  [
    Vector2(0, 1),
    Vector2(0, 2),
    Vector2(1, 2),
    Vector2(1, 3),
  ],
]

oBlock = [
  [
    Vector2(1, 1),
    Vector2(2, 1),
    Vector2(1, 2),
    Vector2(2, 2),
  ]
]*4

blocks = [tBlock, iBlock, lBlock, sBlock, zBlock, oBlock,]
