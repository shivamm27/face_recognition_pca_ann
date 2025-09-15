def train_ann(X, y, num_classes, epochs=30):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import to_categorical

    y_cat = to_categorical(y, num_classes)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[0],)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X.T, y_cat, epochs=epochs, validation_split=0.2, verbose=1)

    return model, history
