# X_train contains word indices (single int between 0 and max_words)
# Y_train0 contains class indices (single int between 0 and nb_classes)

# X_train = sequence.pad_sequences(X_train, maxlen=maxlen, padding='post')
# X_test = sequence.pad_sequences(X_test, maxlen=maxlen, padding='post')
#
# Y_train = np.zeros((batchSize,globvars.nb_classes))#,dtype=np.float32)
# for t in range(batchSize):
#     Y_train[t][Y_train0[t]]=1
# Y_test = np.zeros((len(Y_test0),globvars.nb_classes))#,dtype=np.float32)
# for t in range(len(Y_test)):
#     Y_test[t][Y_test0[t]]=1
#
# model = Sequential()
# model.add(Embedding(globvars.max_words, globvars.embedsize, input_length=maxlen))
# model.add(LSTM(globvars.hidden,return_sequences=False))
# model.add(Dropout(0.5))
# model.add(Dense(globvars.nb_classes))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adam')
#
# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=3)
# score, acc = model.evaluate(X_test, Y_test,
#                             batch_size=batch_size,
#                             show_accuracy=True)