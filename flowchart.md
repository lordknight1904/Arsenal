
---
title: Animal example
---
```mermaid
classDiagram
    class Input {
        +Pillow image
        -from_disk()
    }
    class TrainableInput {
        +Tensor prediction
        +from_input()
    }
    class Sample {
        +numpy label
        +from_trainable_input()
        +from_input()
    }
    
    class Sample {
        +numpy label
        +from_trainable_input()
        +from_input()
    }

    TrainableInput <|-- Input
    Sample <|-- TrainableInput
    Sample <|-- Input
```
