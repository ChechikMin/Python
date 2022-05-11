import torch

from include.classification import *


model = Classification(12, 1)
model.extract_data('src/UCI_13.csv', aim_par='default.payment.next.month')
model.train_predict(num_epochs=100)
# model.load_state_dict(torch.load('bin/non_lin-UCI_13.pt'))
# model.predict()
model.show_results()

print(*model.parameters(), sep='\n\n')
torch.save(model.state_dict(), 'bin/non_lin-UCI_13.pt')
