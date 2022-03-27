from LogisticRegr import *


if __name__ == '__main__':

    df = pd.read_csv('clean_data.csv', delimiter=',')
    # head = ['card', 'reports', 'age', 'income', 'expenditure', 'owner', 'selfemp', 'dependents', 'months',
    #         'majorcards', 'active']
    # itemsReplaced = {'yes': 1, 'no': 0}
    # df = df.replace(itemsReplaced)
    # print(df.head())
    # pd.columns = head
    # print(pd)

    card = df['card']  #
    reports = df['reports']
    age = df['age']
    income = df['income']
    expenditure = df['expenditure']
    owner = df['owner']
    selfemp = df['selfemp']
    dependents = df['dependents']
    majorcards = df['majorcards']
    active = df['active']

    train_size = int(len(card) * .75)

    train_x = np.array(card[:train_size])
    train_a = np.array(reports[:train_size])
    train_b = np.array(age[:train_size])
    train_c = np.array(income[:train_size])
    train_d = np.array(expenditure[:train_size])
    train_e = np.array(owner[:train_size])
    train_f = np.array(selfemp[:train_size])
    train_g = np.array(dependents[:train_size])
    train_i = np.array(majorcards[:train_size])
    train_j = np.array(active[:train_size])

    test_x = np.array(card[train_size:len(card)])
    test_a = np.array(reports[train_size:len(reports)])
    test_b = np.array(age[train_size:len(age)])
    test_c = np.array(income[train_size:len(income)])
    test_d = np.array(expenditure[train_size:len(expenditure)])
    test_e = np.array(owner[train_size:len(owner)])
    test_f = np.array(selfemp[train_size:len(selfemp)])
    test_g = np.array(dependents[train_size:len(dependents)])
    test_i = np.array(majorcards[train_size:len(majorcards)])
    test_j = np.array(active[train_size:len(active)])

    learningRate = .0001
    epochs = 50

    model = LogisticRegr(9, 1)

    losses = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    for epoch in range(epochs):
        for i in range(len(train_x)):
            # Converting inputs and labels to Variable
            if torch.cuda.is_available():
                inputs = Variable(torch.from_numpy(np.array([train_a[i], train_b[i], train_c[i],
                                                             train_d[i], train_e[i], train_f[i],
                                                             train_g[i], train_i[i],
                                                             train_j[i]])).cuda()).double()
                labels = Variable(torch.from_numpy(np.array([train_x[i]])).cuda()).double()
            else:
                inputs = Variable(torch.from_numpy(np.array([train_a[i], train_b[i], train_c[i],
                                                             train_d[i], train_e[i], train_f[i],
                                                             train_g[i], train_i[i],
                                                             train_j[i]]))).double()
                labels = Variable(torch.from_numpy(np.array([train_x[i]]))).double()

            optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = model(inputs)

            # get loss for the predicted output
            loss = losses(outputs, labels)
            # print(loss)
            # get gradients w.r.t to parameters
            loss.backward()

            # update parameters
            optimizer.step()

        print(f'epoch {epoch}, loss {loss.item()}')
    predicted = []
    for epoch in range(len(test_x)):
        with torch.no_grad():  # we don't need gradients in the testing phase
            if torch.cuda.is_available():
                predicted.append(model(
                    Variable(torch.from_numpy(np.array([train_a[epoch], train_b[epoch], train_c[epoch],
                                                        train_d[epoch], train_e[epoch], train_f[epoch],
                                                        train_g[epoch], train_i[epoch],
                                                        train_j[epoch]])).cuda()).double()).cpu().data.numpy())
            else:
                predicted.append(
                    model(Variable(torch.from_numpy(np.array([train_a[epoch], train_b[epoch], train_c[epoch],
                                                              train_d[epoch], train_e[epoch], train_f[epoch],
                                                              train_g[epoch], train_i[epoch],
                                                              train_j[epoch]]))).double()).data.numpy())
    print(predicted[-1])

    plt.clf()
    plt.bar([int(_) for _ in range(len(predicted))], predicted[-1])

    plt.grid()

    # plt.scatter(test_x, test_a)
    # plt.scatter(test_b, np.array(predicted))
    plt.legend(['Test_x/Predicted', 'Truth'])
    # loss = losses(Variable(torch.from_numpy(np.array([93.6,5.6]))), Variable(torch.from_numpy(np.array(test_x[30]))))
    # print(loss)

    plt.show()

    # print(*model.parameters())

    # window = Tk()
    # window.mainloop()
    # window.title('Application for Calc procent')
