from include.classification import *


model = Classification(23, 1)
model.extract_data('src/UCI_Credit_Card.csv', aim_par='default.payment.next.month')
model.train_predict(num_epochs=200)
# model.load_state_dict(torch.load('bin/log_regr-UCI.pt'))
# model.predict()
model.show_results()

print(*model.parameters(), sep='\n\n')
torch.save(model.state_dict(), 'bin/nonlinear-UCI.pt')
