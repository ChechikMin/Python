from include.classification import *


model = Classification(23, 1)
model.extract_data('src/UCI_Credit_Card.csv', aim_par='default.payment.next.month')
# model.load_state_dict(torch.load('bin/linear_sigmoid-UCI.pt'))
model.train_predict(num_epochs=200)
model.predict()
model.show_results()

print(*model.parameters(), sep='\n\n')
