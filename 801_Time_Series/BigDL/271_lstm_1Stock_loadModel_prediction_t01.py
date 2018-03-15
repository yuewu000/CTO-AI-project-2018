def build_model(layers):
    model = Sequential()
    recurrent1 = Recurrent()
    recurrent1.add(LSTM(layers[0], layers[1]))
    drop = Dropout(0.2)
    recurrent2 = Recurrent()
    recurrent2.add(LSTM(layers[1], layers[1]))

    model.add(InferReshape([-1, layers[0]], True))
    model.add(recurrent1)
    model.add(drop)
    model.add(Echo())
    model.add(recurrent2)
    model.add(drop)
    model.add(Echo())
    model.add(Select(2, -1))
    model.add(Linear(layers[1], layers[2]))
    return model

