from LogisticRegr import *

if __name__ == '__main__':

    head = ["card", "reports", "age","income","share","expenditure","owner","selfemp","dependents","months","majorcards","active"]
    df = pd.read_csv("AER_credit_card_data.csv", delimiter = ",")
    itemsReplaced = {"yes": 1, "no": 0}
    df = df.replace(itemsReplaced)
    print(df.head())
    pd.columns = head
    print(pd)

    card = pd["card"]

    share = pd["share"]
    income = pd["income"]

    train_size = int(len(card) * 0.75)

    train_x = np.array(card[:train_size])

    train_z = np.array(share[:train_size])
    train_y = np.array(income[:train_size])
    train_x.reshape(-1,1)
    train_y.reshape(-1,1)
    train_z.reshape(-1,1)

    test_x = np.array(card[train_size:len(card)])
    test_y = np.array(income[train_size:len(income)])
    test_z = np.array(share[train_size:len(share)])

    learningRate = 0.0001
    epochs = 50

    model = LogisticRegr(2, 1)

    losses = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    for epoch in range(epochs):
        for i in range(len(train_x)):
        # Converting inputs and labels to Variable
            if torch.cuda.is_available():
                inputs = Variable(torch.from_numpy(np.array([train_y[i],train_z[i]])).cuda())
                labels = Variable(torch.from_numpy(np.array([train_x[i]])).cuda())
            else:
                inputs = Variable(torch.from_numpy(np.array([train_y[i],train_z[i]])))
                labels = Variable(torch.from_numpy(np.array([train_x[i]])))

            optimizer.zero_grad()

        # get output from the model, given the inputs
            outputs = model(inputs)

        # get loss for the predicted output
            loss = losses(outputs, labels)
            print(loss)
        # get gradients w.r.t to parameters
            loss.backward()

        # update parameters
            optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))
    predicted = []
    for epoch in range(len(test_x)):
        with torch.no_grad():  # we don't need gradients in the testing phase
            if torch.cuda.is_available():
                predicted.append( model(Variable(torch.from_numpy(np.array([test_x[epoch],test_z[epoch]])).cuda())).cpu().data.numpy())
            else:
                predicted.append( model(Variable(torch.from_numpy(np.array([test_x[epoch],test_z[epoch]])))).data.numpy() )
            print(predicted)

    plt.clf()
    plt.scatter(test_x, np.array(predicted))

    plt.grid()


    plt.scatter(test_x,test_y)
    plt.scatter(test_z, np.array(predicted))
    plt.legend(["Test_x/Predicted","Truth"])
    #loss = losses(Variable(torch.from_numpy(np.array([93.6,5.6]))), Variable(torch.from_numpy(np.array(test_y[30]))))
    #print(loss)

    plt.show()

    #window = Tk()
    #window.mainloop()
    # #window.title("Application for Calc procent")
